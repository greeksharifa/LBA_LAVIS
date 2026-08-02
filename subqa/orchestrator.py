"""Hierarchical Sub-QA orchestration over common model/dataset protocols."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

from model.protocol import GenerationResult
from subqa.parsing import parse_answer, parse_questions
from subqa.prompts import (
    build_decomposition_prompt,
    build_direct_answer_prompt,
    build_question_repair_prompt,
    build_refined_answer_prompt,
)
from subqa.schema import (
    HierarchyConfig,
    SUPPORTED_CONFIDENCE_TYPE_ORDER,
    preorder_nodes,
    project_depth_one_questions,
    stable_node_id,
    validate_suba_artifact_entry,
    validate_subq_tree,
)
from subqa.selection import evidence_subsets, select_best_candidate
from util.artifacts import STAGE_DEPENDENCIES, mark_stage_complete, mark_stage_started
from util.confidence import normalize_confidence
from util.logger import get_logger
from util.path import get_output_dir, get_output_filename


class HierarchyStageIncompleteError(RuntimeError):
    """A required semantic result stayed invalid after configured repairs."""


_REQUIRED_SAMPLE_FIELDS = ("qid", "main_q", "question_type", "data_type")
def _validated_samples(samples: Sequence[Mapping]) -> list[dict]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        samples = list(samples)
    if not samples:
        raise ValueError("hierarchy dataset has no samples")
    validated = []
    seen_qids = set()
    for index, original in enumerate(samples):
        if not isinstance(original, Mapping):
            raise ValueError(f"hierarchy sample {index} must be a mapping")
        sample = dict(original)
        missing = [field for field in _REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"hierarchy sample {index} missing fields: {missing}")
        qid = str(sample["qid"])
        if not qid or qid in seen_qids:
            raise ValueError(f"hierarchy qid must be unique and non-empty: {qid!r}")
        for field in ("main_q", "question_type", "data_type"):
            if not isinstance(sample[field], str) or not sample[field].strip():
                raise ValueError(
                    f"hierarchy sample {qid} field {field} must be non-empty"
                )
        sample["qid"] = qid
        validated.append(sample)
        seen_qids.add(qid)
    return validated


def _validate_generation_result(
    result,
    label: str,
    required_metrics: Sequence[str],
) -> GenerationResult:
    if not isinstance(result, GenerationResult):
        raise ValueError(f"{label} must be a GenerationResult")
    for metric in required_metrics:
        if metric not in result.confidence:
            raise ValueError(f"{label} missing confidence metric {metric!r}")
    for metric in SUPPORTED_CONFIDENCE_TYPE_ORDER:
        if metric not in result.confidence:
            continue
        normalize_confidence(result.confidence[metric], metric)
    return result


def _generate_requests(
    model,
    requests: list[dict],
    hierarchy: HierarchyConfig,
) -> list:
    if not requests:
        return []
    results = []
    batch_size = hierarchy.generation_batch_size
    for offset in range(0, len(requests), batch_size):
        chunk = requests[offset : offset + batch_size]
        prompts = [
            model.apply_chat_template(
                request["text_prompt"],
                vision=request["sample"].get("vision"),
                mm_uuids=request["sample"]["qid"],
            )
            for request in chunk
        ]
        chunk_results = model.generate_results(prompts)
        if not isinstance(chunk_results, list) or len(chunk_results) != len(chunk):
            actual_count = (
                len(chunk_results) if isinstance(chunk_results, list) else "non-list"
            )
            raise ValueError(
                "generated result count mismatch: "
                f"expected {len(chunk)}, got {actual_count}"
            )
        results.extend(
            _validate_generation_result(
                result,
                f"generation result {offset + index}",
                hierarchy.required_confidence_types,
            )
            for index, result in enumerate(chunk_results)
        )
    return results


def _expanded_questions(
    model,
    requests: list[dict],
    hierarchy: HierarchyConfig,
) -> list[dict]:
    initial_results = _generate_requests(model, requests, hierarchy)
    states = []
    for request, result in zip(requests, initial_results):
        questions = parse_questions(
            result.text, expected_count=request["expected_count"]
        )
        states.append(
            {
                "questions": questions,
                "confidence": dict(result.confidence),
                "last_text": result.text,
            }
        )

    for _ in range(hierarchy.repair_attempts):
        repair_requests = []
        repair_state_indices = []
        for index, (request, state) in enumerate(zip(requests, states)):
            if len(state["questions"]) >= request["expected_count"]:
                continue
            repair_request = dict(request)
            repair_request["text_prompt"] = build_question_repair_prompt(
                request["sample"],
                child_count=request["expected_count"],
                existing_questions=state["questions"],
                parent_question=request["parent_question"],
                ancestor_questions=request["ancestor_questions"],
            )
            repair_requests.append(repair_request)
            repair_state_indices.append(index)
        if not repair_requests:
            break
        repair_results = _generate_requests(model, repair_requests, hierarchy)
        for state_index, repair_result in zip(repair_state_indices, repair_results):
            request = requests[state_index]
            state = states[state_index]
            combined = parse_questions(
                f"{state['last_text']}\n{repair_result.text}",
                expected_count=request["expected_count"],
            )
            if len(combined) > len(state["questions"]):
                state["confidence"] = dict(repair_result.confidence)
            state["questions"] = combined
            state["last_text"] = "\n".join(combined)
    return states


def _ancestor_questions(node_id: str, node_by_id: Mapping[str, Mapping]) -> tuple:
    ancestors = []
    parent_id = node_by_id[node_id]["parent_id"]
    while parent_id != "root":
        parent = node_by_id[parent_id]
        ancestors.append(parent["question"])
        parent_id = parent["parent_id"]
    return tuple(reversed(ancestors))


def _new_node(node_id, parent_id, depth, sibling_index, question, max_depth):
    is_leaf = depth == max_depth
    return {
        "id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "sibling_index": sibling_index,
        "question": question,
        "child_ids": [],
        "expansion_status": "leaf" if is_leaf else "failed",
        "expansion_confidence": None,
        "failure_reason": None if is_leaf else "node expansion not completed",
    }


def generate_subq_artifacts(samples, model, hierarchy: HierarchyConfig):
    """Expand all sample question trees top-down and return canonical artifacts."""
    if not isinstance(hierarchy, HierarchyConfig) or not hierarchy.enabled:
        raise ValueError("hierarchical SubQ generation requires depth > 1")
    samples = _validated_samples(samples)
    states = OrderedDict()
    root_requests = []
    for sample in samples:
        qid = sample["qid"]
        tree = {
            "schema_version": hierarchy.schema_version,
            "root_id": "root",
            "max_depth": hierarchy.depth,
            "nodes": [],
        }
        states[qid] = {
            "sample": sample,
            "tree": tree,
            "node_by_id": {},
            "root_confidence": None,
        }
        root_requests.append(
            {
                "sample": sample,
                "parent_id": "root",
                "parent_question": None,
                "ancestor_questions": (),
                "expected_count": hierarchy.branching[0],
                "text_prompt": build_decomposition_prompt(
                    sample, child_count=hierarchy.branching[0]
                ),
            }
        )

    root_expansions = _expanded_questions(model, root_requests, hierarchy)
    for request, expansion in zip(root_requests, root_expansions):
        sample = request["sample"]
        qid = sample["qid"]
        questions = expansion["questions"]
        expected = request["expected_count"]
        if len(questions) != expected:
            raise HierarchyStageIncompleteError(
                f"root expansion for qid {qid} expected {expected} questions, "
                f"got {len(questions)} after repairs"
            )
        state = states[qid]
        state["root_confidence"] = expansion["confidence"]
        for sibling_index, question in enumerate(questions):
            node_id = stable_node_id("root", sibling_index)
            node = _new_node(
                node_id,
                "root",
                1,
                sibling_index,
                question,
                hierarchy.depth,
            )
            state["tree"]["nodes"].append(node)
            state["node_by_id"][node_id] = node

    for depth in range(1, hierarchy.depth):
        requests = []
        for state in states.values():
            sample = state["sample"]
            parents = sorted(
                (
                    node
                    for node in state["node_by_id"].values()
                    if node["depth"] == depth
                ),
                key=lambda node: tuple(int(part) for part in node["id"].split(".")),
            )
            for parent in parents:
                ancestors = _ancestor_questions(parent["id"], state["node_by_id"])
                child_count = hierarchy.branching[depth]
                requests.append(
                    {
                        "sample": sample,
                        "parent_id": parent["id"],
                        "parent_question": parent["question"],
                        "ancestor_questions": ancestors,
                        "expected_count": child_count,
                        "text_prompt": build_decomposition_prompt(
                            sample,
                            child_count=child_count,
                            parent_question=parent["question"],
                            ancestor_questions=ancestors,
                        ),
                    }
                )
        expansions = _expanded_questions(model, requests, hierarchy)
        for request, expansion in zip(requests, expansions):
            qid = request["sample"]["qid"]
            state = states[qid]
            parent = state["node_by_id"][request["parent_id"]]
            questions = expansion["questions"]
            expected = request["expected_count"]
            parent["expansion_confidence"] = expansion["confidence"]
            if len(questions) == expected:
                parent["expansion_status"] = "expanded"
                parent["failure_reason"] = None
            elif questions:
                parent["expansion_status"] = "partial"
                parent["failure_reason"] = (
                    f"expected {expected} child questions, got {len(questions)} "
                    "after repairs"
                )
            else:
                parent["expansion_status"] = "failed"
                parent["failure_reason"] = (
                    f"expected {expected} child questions, got 0 after repairs"
                )
            for sibling_index, question in enumerate(questions):
                child_id = stable_node_id(parent["id"], sibling_index)
                child = _new_node(
                    child_id,
                    parent["id"],
                    depth + 1,
                    sibling_index,
                    question,
                    hierarchy.depth,
                )
                parent["child_ids"].append(child_id)
                state["tree"]["nodes"].append(child)
                state["node_by_id"][child_id] = child

    artifacts = OrderedDict()
    for qid, state in states.items():
        tree = state["tree"]
        tree["nodes"] = [dict(node) for node in preorder_nodes(tree)]
        validate_subq_tree(tree, branching=hierarchy.branching)
        artifacts[qid] = {
            "subq_list": project_depth_one_questions(
                tree, expected_count=hierarchy.branching[0]
            ),
            "conf_subq": state["root_confidence"],
            "subq_tree": tree,
        }
    return artifacts


def _answer_record(result: GenerationResult) -> dict:
    answer = parse_answer(result.text)
    return {
        "answer": answer,
        "confidence": dict(result.confidence),
        "status": "valid" if answer is not None else "invalid",
        "failure_reason": None if answer is not None else "blank answer",
    }


def _selected_direct(direct: Mapping, source: str) -> dict:
    return {
        "answer": direct["answer"],
        "confidence": dict(direct["confidence"]),
        "status": direct["status"],
        "failure_reason": direct["failure_reason"],
        "source": source,
        "support_node_ids": [],
    }


def _answer_ancestor_questions(node: Mapping, node_by_id: Mapping) -> tuple:
    return _ancestor_questions(node["id"], node_by_id)


def generate_suba_artifacts(samples, model, hierarchy: HierarchyConfig):
    """Answer all nodes direct-first, then refine internal nodes bottom-up."""
    if not isinstance(hierarchy, HierarchyConfig) or not hierarchy.enabled:
        raise ValueError("hierarchical SubA generation requires depth > 1")
    samples = _validated_samples(samples)
    states = OrderedDict()
    direct_requests = []
    for sample in samples:
        qid = sample["qid"]
        tree = sample.get("subq_tree")
        validate_subq_tree(tree, branching=hierarchy.branching)
        nodes = preorder_nodes(tree)
        node_by_id = {node["id"]: node for node in nodes}
        answers = OrderedDict(
            (
                node["id"],
                {"direct": None, "candidates": [], "selected": None},
            )
            for node in nodes
        )
        states[qid] = {
            "sample": sample,
            "tree": tree,
            "nodes": nodes,
            "node_by_id": node_by_id,
            "answers": answers,
        }
        for node in nodes:
            direct_requests.append(
                {
                    "sample": sample,
                    "qid": qid,
                    "node_id": node["id"],
                    "text_prompt": build_direct_answer_prompt(
                        sample,
                        target_question=node["question"],
                        ancestor_questions=_answer_ancestor_questions(
                            node, node_by_id
                        ),
                    ),
                }
            )

    direct_results = _generate_requests(model, direct_requests, hierarchy)
    for request, result in zip(direct_requests, direct_results):
        states[request["qid"]]["answers"][request["node_id"]][
            "direct"
        ] = _answer_record(result)

    for _ in range(hierarchy.repair_attempts):
        repair_requests = []
        for request in direct_requests:
            direct = states[request["qid"]]["answers"][request["node_id"]][
                "direct"
            ]
            if direct["status"] == "valid":
                continue
            repair = dict(request)
            repair["text_prompt"] = (
                f"{request['text_prompt']}\n"
                "The previous response was blank. Return a non-empty answer only."
            )
            repair_requests.append(repair)
        if not repair_requests:
            break
        repair_results = _generate_requests(model, repair_requests, hierarchy)
        for request, result in zip(repair_requests, repair_results):
            states[request["qid"]]["answers"][request["node_id"]][
                "direct"
            ] = _answer_record(result)

    for qid, state in states.items():
        for node in state["nodes"]:
            direct = state["answers"][node["id"]]["direct"]
            if node["depth"] == 1 and direct["status"] != "valid":
                raise HierarchyStageIncompleteError(
                    f"depth-1 direct answer for qid {qid} node {node['id']} "
                    "remained invalid after repairs"
                )
            if node["depth"] == hierarchy.depth:
                state["answers"][node["id"]]["selected"] = _selected_direct(
                    direct, "leaf_direct"
                )

    for depth in range(hierarchy.depth - 1, 0, -1):
        candidate_requests = []
        internal_nodes = []
        for qid, state in states.items():
            for node in state["nodes"]:
                if node["depth"] != depth:
                    continue
                internal_nodes.append((qid, node["id"]))
                valid_child_ids = [
                    child_id
                    for child_id in node["child_ids"]
                    if state["answers"][child_id]["selected"] is not None
                    and state["answers"][child_id]["selected"]["status"] == "valid"
                ]
                subsets = evidence_subsets(
                    valid_child_ids, hierarchy.suba_m, hierarchy.suba_k
                )
                for support_node_ids in subsets:
                    child_qa = [
                        (
                            state["node_by_id"][child_id]["question"],
                            state["answers"][child_id]["selected"]["answer"],
                        )
                        for child_id in support_node_ids
                    ]
                    direct = state["answers"][node["id"]]["direct"]
                    use_direct = (
                        hierarchy.condition_on_direct_suba
                        and direct["status"] == "valid"
                    )
                    candidate_index = len(
                        state["answers"][node["id"]]["candidates"]
                    )
                    state["answers"][node["id"]]["candidates"].append(None)
                    candidate_requests.append(
                        {
                            "sample": state["sample"],
                            "qid": qid,
                            "node_id": node["id"],
                            "candidate_index": candidate_index,
                            "support_node_ids": list(support_node_ids),
                            "text_prompt": build_refined_answer_prompt(
                                state["sample"],
                                target_question=node["question"],
                                child_qa=child_qa,
                                direct_draft=(
                                    direct["answer"] if use_direct else None
                                ),
                                condition_on_direct_suba=use_direct,
                            ),
                        }
                    )

        candidate_results = _generate_requests(model, candidate_requests, hierarchy)
        for request, result in zip(candidate_requests, candidate_results):
            candidate = _answer_record(result)
            candidate["support_node_ids"] = request["support_node_ids"]
            states[request["qid"]]["answers"][request["node_id"]][
                "candidates"
            ][request["candidate_index"]] = candidate

        for qid, node_id in internal_nodes:
            entry = states[qid]["answers"][node_id]
            selected = select_best_candidate(
                entry["candidates"], hierarchy.confidence_type
            )
            if selected is None:
                entry["selected"] = _selected_direct(
                    entry["direct"], "direct_fallback"
                )
                continue
            candidate, normalized_score, candidate_index = selected
            entry["selected"] = {
                "answer": candidate["answer"],
                "confidence": dict(candidate["confidence"]),
                "status": "valid",
                "failure_reason": None,
                "source": "confidence",
                "support_node_ids": list(candidate["support_node_ids"]),
                "candidate_index": candidate_index,
                "normalized_confidence": normalized_score,
            }

    artifacts = OrderedDict()
    for qid, state in states.items():
        depth_one_ids = [str(index) for index in range(hierarchy.branching[0])]
        selected = [state["answers"][node_id]["selected"] for node_id in depth_one_ids]
        if any(value is None or value["status"] != "valid" for value in selected):
            raise HierarchyStageIncompleteError(
                f"qid {qid} has an invalid depth-1 selected answer"
            )
        projected_metrics = [
            metric
            for metric in SUPPORTED_CONFIDENCE_TYPE_ORDER
            if all(metric in value["confidence"] for value in selected)
        ]
        artifact = {
            "suba_list": [value["answer"] for value in selected],
            "conf_suba": {
                metric: [value["confidence"][metric] for value in selected]
                for metric in projected_metrics
            },
            "answers_by_node": state["answers"],
        }
        validate_suba_artifact_entry(artifact, state["tree"], hierarchy)
        artifacts[qid] = artifact
    return artifacts


def run_hierarchical_stage(
    cfg,
    model,
    hierarchy: HierarchyConfig,
    *,
    dataset_loader,
    prepare_manifest,
    atomic_writer,
):
    """Run one hierarchical stage through the existing manifest transaction."""
    mode = str(cfg.runner_cfg.mode)
    if mode not in ("subq", "suba"):
        raise ValueError(f"unsupported hierarchy stage: {mode}")
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_loader(cfg)
    samples = _validated_samples(list(dataset))
    sample_qids = [sample["qid"] for sample in samples]
    manifest_path = prepare_manifest(cfg, sample_qids, output_dir)
    dependencies = STAGE_DEPENDENCIES[mode]
    parent_generations = getattr(dataset, "dependency_generations", None)
    if dependencies and parent_generations is None:
        raise ValueError(
            f"dataset loader for {mode} must provide dependency_generations"
        )
    started_manifest = mark_stage_started(
        manifest_path,
        mode,
        parent_generations=parent_generations or {},
    )
    generation_id = started_manifest["stages"][mode]["generation_id"]
    logger = get_logger()
    logger.info("Hierarchy stage %s: prepared %d samples", mode, len(samples))
    if mode == "subq":
        artifacts = generate_subq_artifacts(samples, model, hierarchy)
    else:
        artifacts = generate_suba_artifacts(samples, model, hierarchy)
    output_path = output_dir / get_output_filename(cfg)
    mark_stage_complete(
        manifest_path,
        mode,
        generation_id,
        artifact_writer=lambda: atomic_writer(output_path, artifacts),
    )
    logger.info("Saved hierarchical %s outputs to %s", mode, output_path)
    return artifacts

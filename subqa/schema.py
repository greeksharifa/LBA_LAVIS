"""Canonical hierarchy configuration and tree schema helpers."""

from dataclasses import asdict, dataclass
import math
from math import comb
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from subqa.selection import evidence_subsets, select_best_candidate
from util.confidence import normalize_confidence


SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_CONFIDENCE_TYPE_ORDER = ("seq_ppl", "token_min_prob")
SUPPORTED_CONFIDENCE_TYPES = frozenset(SUPPORTED_CONFIDENCE_TYPE_ORDER)
FALLBACK_POLICY = "direct_on_insufficient_children_or_no_valid_candidate"


def _get(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class HierarchyConfig:
    depth: int
    branching: Tuple[int, ...]
    suba_m: int
    suba_k: int
    confidence_type: str
    runner_confidence_type: str
    condition_on_direct_suba: bool
    max_nodes: int
    repair_attempts: int
    generation_batch_size: int
    schema_version: int
    fallback_policy: str = FALLBACK_POLICY

    @property
    def enabled(self) -> bool:
        return self.depth > 1

    @property
    def node_count(self) -> int:
        total = 0
        level_count = 1
        for child_count in self.branching:
            level_count *= child_count
            total += level_count
        return total

    def to_manifest_dict(self) -> dict:
        value = asdict(self)
        value.pop("runner_confidence_type")
        value["branching"] = list(self.branching)
        return value

    @property
    def required_confidence_types(self) -> Tuple[str, ...]:
        required = {self.runner_confidence_type, self.confidence_type}
        return tuple(
            metric
            for metric in SUPPORTED_CONFIDENCE_TYPE_ORDER
            if metric in required
        )


def normalize_hierarchy_config(runner_cfg: Any) -> HierarchyConfig:
    """Validate and canonicalize all hierarchy semantics in one boundary."""
    n = _positive_integer(_get(runner_cfg, "N", 4), "runner.N")
    _positive_integer(_get(runner_cfg, "M", 2), "runner.M")
    _positive_integer(_get(runner_cfg, "K", 4), "runner.K")
    depth = _positive_integer(
        _get(runner_cfg, "subqa_depth", 1), "runner.subqa_depth"
    )
    configured_branching: Optional[Any] = _get(
        runner_cfg, "branching_by_depth", None
    )
    if configured_branching is None:
        if depth != 1:
            raise ValueError(
                "runner.branching_by_depth must be explicit when subqa_depth > 1"
            )
        branching = (n,)
    else:
        if isinstance(configured_branching, (str, bytes)):
            raise ValueError("runner.branching_by_depth must be a sequence")
        try:
            branching = tuple(configured_branching)
        except TypeError as error:
            raise ValueError(
                "runner.branching_by_depth must be a sequence"
            ) from error
        if len(branching) != depth:
            raise ValueError(
                "runner.branching_by_depth length must equal runner.subqa_depth"
            )
        branching = tuple(
            _positive_integer(value, "runner.branching_by_depth branching value")
            for value in branching
        )
        if branching[0] != n:
            raise ValueError(
                "runner.branching_by_depth[0] must equal runner.N"
            )

    suba_m = _positive_integer(_get(runner_cfg, "suba_M", 2), "runner.suba_M")
    suba_k = _positive_integer(_get(runner_cfg, "suba_K", 3), "runner.suba_K")
    for child_count in branching[1:]:
        if suba_m > child_count:
            raise ValueError(
                "runner.suba_M must not exceed any internal child count"
            )
        if suba_k > comb(child_count, suba_m):
            raise ValueError(
                "runner.suba_K must not exceed available child combinations"
            )

    max_nodes = _positive_integer(
        _get(runner_cfg, "subqa_max_nodes", 64), "runner.subqa_max_nodes"
    )
    repair_attempts = _non_negative_integer(
        _get(runner_cfg, "subqa_repair_attempts", 1),
        "runner.subqa_repair_attempts",
    )
    generation_batch_size = _positive_integer(
        _get(runner_cfg, "subqa_generation_batch_size", 64),
        "runner.subqa_generation_batch_size",
    )
    confidence_type = str(
        _get(runner_cfg, "suba_confidence_type", "token_min_prob")
    )
    if confidence_type not in SUPPORTED_CONFIDENCE_TYPES:
        raise ValueError(
            f"unsupported runner.suba_confidence_type: {confidence_type!r}"
        )
    runner_confidence_type = str(
        _get(runner_cfg, "confidence_type", "token_min_prob")
    )
    if depth > 1 and runner_confidence_type not in SUPPORTED_CONFIDENCE_TYPES:
        raise ValueError(
            f"unsupported runner.confidence_type for hierarchy: "
            f"{runner_confidence_type!r}"
        )
    condition_on_direct_suba = _get(
        runner_cfg, "condition_on_direct_suba", True
    )
    if not isinstance(condition_on_direct_suba, bool):
        raise ValueError("runner.condition_on_direct_suba must be a boolean")
    schema_version = _get(
        runner_cfg, "subqa_schema_version", SUPPORTED_SCHEMA_VERSION
    )
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SUPPORTED_SCHEMA_VERSION
    ):
        raise ValueError(
            f"runner.subqa_schema_version must be supported schema version "
            f"{SUPPORTED_SCHEMA_VERSION}"
        )

    hierarchy = HierarchyConfig(
        depth=depth,
        branching=branching,
        suba_m=suba_m,
        suba_k=suba_k,
        confidence_type=confidence_type,
        runner_confidence_type=runner_confidence_type,
        condition_on_direct_suba=condition_on_direct_suba,
        max_nodes=max_nodes,
        repair_attempts=repair_attempts,
        generation_batch_size=generation_batch_size,
        schema_version=schema_version,
    )
    if hierarchy.node_count > max_nodes:
        raise ValueError(
            "hierarchy question node count exceeds runner.subqa_max_nodes: "
            f"{hierarchy.node_count} > {max_nodes}"
        )
    return hierarchy


NODE_EXPANSION_STATUSES = frozenset({"leaf", "expanded", "partial", "failed"})
REQUIRED_NODE_FIELDS = frozenset(
    {
        "id",
        "parent_id",
        "depth",
        "sibling_index",
        "question",
        "child_ids",
        "expansion_status",
        "expansion_confidence",
        "failure_reason",
    }
)


def validate_confidence_mapping(
    confidence: Any,
    label: str,
    required_metrics: Sequence[str] = (),
) -> None:
    """Validate raw confidence without requiring unrelated adapter metrics."""
    if not isinstance(confidence, Mapping):
        raise ValueError(f"{label} must be an object")
    for metric in required_metrics:
        if metric not in confidence:
            raise ValueError(f"{label} missing {metric}")
    for metric, value in confidence.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"{label} metric names must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label}.{metric} must be a finite number")
        if metric in SUPPORTED_CONFIDENCE_TYPES:
            try:
                normalize_confidence(value, metric)
            except ValueError as error:
                raise ValueError(f"{label}.{metric}: {error}") from error


def stable_node_id(parent_id: str, sibling_index: int) -> str:
    """Return the path-based ID for a child of ``parent_id``."""
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("parent_id must be a non-empty string")
    if (
        isinstance(sibling_index, bool)
        or not isinstance(sibling_index, int)
        or sibling_index < 0
    ):
        raise ValueError("sibling_index must be a non-negative integer")
    if parent_id == "root":
        return str(sibling_index)
    return f"{parent_id}.{sibling_index}"


def build_complete_subq_tree(
    branching: Sequence[int],
    question_factory: Optional[Callable[[str], str]] = None,
) -> dict:
    """Build a complete canonical question tree, primarily for pure orchestration use."""
    if isinstance(branching, (str, bytes)) or not isinstance(branching, Sequence):
        raise ValueError("branching must be a non-empty sequence")
    canonical_branching = tuple(
        _positive_integer(value, "branching value") for value in branching
    )
    if not canonical_branching:
        raise ValueError("branching must be a non-empty sequence")
    if question_factory is None:
        question_factory = lambda node_id: node_id
    if not callable(question_factory):
        raise ValueError("question_factory must be callable")

    max_depth = len(canonical_branching)
    nodes = []

    def append_subtree(parent_id: str, depth: int, sibling_index: int) -> None:
        node_id = stable_node_id(parent_id, sibling_index)
        question = question_factory(node_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question for node {node_id!r} must be non-empty")
        is_leaf = depth == max_depth
        child_count = 0 if is_leaf else canonical_branching[depth]
        child_ids = [stable_node_id(node_id, index) for index in range(child_count)]
        nodes.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "depth": depth,
                "sibling_index": sibling_index,
                "question": question.strip(),
                "child_ids": child_ids,
                "expansion_status": "leaf" if is_leaf else "expanded",
                "expansion_confidence": None,
                "failure_reason": None,
            }
        )
        for child_index in range(child_count):
            append_subtree(node_id, depth + 1, child_index)

    for root_child_index in range(canonical_branching[0]):
        append_subtree("root", 1, root_child_index)

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "root_id": "root",
        "max_depth": max_depth,
        "nodes": nodes,
    }


def preorder_nodes(tree: Mapping[str, Any]) -> list:
    """Return nodes in canonical preorder without mutating their serialization."""
    raw_nodes = tree.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("subq tree nodes must be a list")
    node_by_id = {}
    for node in raw_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("each subq tree node must be a mapping")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node id must be a non-empty string")
        if node_id in node_by_id:
            raise ValueError(f"duplicate id: {node_id!r}")
        node_by_id[node_id] = node

    root_id = tree.get("root_id")
    root_children = sorted(
        (
            node
            for node in raw_nodes
            if node.get("parent_id") == root_id
        ),
        key=lambda node: node.get("sibling_index", -1),
    )
    ordered = []
    visited = set()

    def visit(node: Mapping[str, Any]) -> None:
        node_id = node["id"]
        if node_id in visited:
            raise ValueError(f"child cycle or duplicate reference at {node_id!r}")
        visited.add(node_id)
        ordered.append(node)
        child_ids = node.get("child_ids")
        if not isinstance(child_ids, list):
            raise ValueError(f"node {node_id!r} child_ids must be a list")
        for child_id in child_ids:
            child = node_by_id.get(child_id)
            if child is None:
                raise ValueError(
                    f"node {node_id!r} child references missing id {child_id!r}"
                )
            visit(child)

    for root_child in root_children:
        visit(root_child)
    if len(visited) != len(raw_nodes):
        unreachable = sorted(set(node_by_id) - visited)
        raise ValueError(f"nodes have a missing parent or are unreachable: {unreachable}")
    return ordered


def validate_subq_tree(
    tree: Mapping[str, Any], branching: Optional[Sequence[int]] = None
) -> None:
    """Fail closed when a serialized SubQ tree violates canonical invariants."""
    if not isinstance(tree, Mapping):
        raise ValueError("subq tree must be a mapping")
    if tree.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"subq tree schema_version must be {SUPPORTED_SCHEMA_VERSION}"
        )
    root_id = tree.get("root_id")
    if root_id != "root":
        raise ValueError("subq tree root_id must be 'root'")
    max_depth = tree.get("max_depth")
    _positive_integer(max_depth, "subq tree max_depth")

    canonical_branching = None
    if branching is not None:
        if isinstance(branching, (str, bytes)) or not isinstance(
            branching, Sequence
        ):
            raise ValueError("branching must be a sequence")
        canonical_branching = tuple(
            _positive_integer(value, "branching value") for value in branching
        )
        if len(canonical_branching) != max_depth:
            raise ValueError("branching length must equal subq tree max_depth")

    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("subq tree nodes must be a list")
    node_by_id = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("each subq tree node must be a mapping")
        missing_fields = REQUIRED_NODE_FIELDS - node.keys()
        if missing_fields:
            raise ValueError(
                f"subq tree node missing fields: {sorted(missing_fields)}"
            )
        node_id = node["id"]
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node id must be a non-empty string")
        if node_id in node_by_id:
            raise ValueError(f"duplicate id: {node_id!r}")
        node_by_id[node_id] = node

    root_children = []
    for node in nodes:
        node_id = node["id"]
        parent_id = node["parent_id"]
        depth = node["depth"]
        sibling_index = node["sibling_index"]
        question = node["question"]
        child_ids = node["child_ids"]
        status = node["expansion_status"]
        confidence = node["expansion_confidence"]
        failure_reason = node["failure_reason"]

        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"node {node_id!r} parent_id must be non-empty")
        if parent_id != root_id and parent_id not in node_by_id:
            raise ValueError(
                f"node {node_id!r} references missing parent {parent_id!r}"
            )
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise ValueError(f"node {node_id!r} depth must be an integer")
        if depth < 1 or depth > max_depth:
            raise ValueError(
                f"node {node_id!r} depth exceeds max_depth {max_depth}"
            )
        expected_depth = (
            1 if parent_id == root_id else node_by_id[parent_id]["depth"] + 1
        )
        if depth != expected_depth:
            raise ValueError(
                f"node {node_id!r} depth is inconsistent with its parent"
            )
        if (
            isinstance(sibling_index, bool)
            or not isinstance(sibling_index, int)
            or sibling_index < 0
        ):
            raise ValueError(
                f"node {node_id!r} sibling_index must be non-negative"
            )
        if node_id != stable_node_id(parent_id, sibling_index):
            raise ValueError(
                f"node {node_id!r} id is inconsistent with parent/sibling"
            )
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"node {node_id!r} question must be non-empty")
        if not isinstance(child_ids, list) or not all(
            isinstance(child_id, str) for child_id in child_ids
        ):
            raise ValueError(f"node {node_id!r} child_ids must be a string list")
        if len(set(child_ids)) != len(child_ids):
            raise ValueError(f"node {node_id!r} has duplicate child ids")
        if status not in NODE_EXPANSION_STATUSES:
            raise ValueError(f"node {node_id!r} has invalid expansion status")
        if confidence is not None:
            validate_confidence_mapping(
                confidence,
                f"node {node_id!r} expansion_confidence",
            )
        if status in {"leaf", "expanded"}:
            if failure_reason is not None:
                raise ValueError(
                    f"node {node_id!r} failure_reason must be null for {status}"
                )
        elif not isinstance(failure_reason, str) or not failure_reason.strip():
            raise ValueError(
                f"node {node_id!r} failure_reason must be non-empty for {status}"
            )
        if parent_id == root_id:
            root_children.append(node)

    children_by_parent = {node_id: [] for node_id in node_by_id}
    for node in nodes:
        if node["parent_id"] != root_id:
            children_by_parent[node["parent_id"]].append(node)

    for node in nodes:
        node_id = node["id"]
        depth = node["depth"]
        status = node["expansion_status"]
        actual_children = sorted(
            children_by_parent[node_id], key=lambda child: child["sibling_index"]
        )
        actual_sibling_indices = [
            child["sibling_index"] for child in actual_children
        ]
        if actual_sibling_indices != list(range(len(actual_children))):
            raise ValueError(
                f"node {node_id!r} child sibling indices must be contiguous"
            )
        actual_child_ids = [child["id"] for child in actual_children]
        if node["child_ids"] != actual_child_ids:
            raise ValueError(
                f"node {node_id!r} child order does not match parent links"
            )

        child_count = len(actual_children)
        if depth == max_depth:
            if child_count:
                raise ValueError(f"leaf node {node_id!r} cannot have children")
            if status != "leaf":
                raise ValueError(f"node {node_id!r} at max_depth must be leaf")
            continue
        if status == "leaf":
            raise ValueError(f"internal node {node_id!r} cannot have leaf status")
        target_count = (
            canonical_branching[depth] if canonical_branching is not None else None
        )
        if target_count is not None and child_count > target_count:
            raise ValueError(
                f"node {node_id!r} child count exceeds branching target"
            )
        if status == "failed" and child_count != 0:
            raise ValueError(f"failed node {node_id!r} must not have children")
        if status == "partial" and (
            child_count == 0
            or (target_count is not None and child_count >= target_count)
        ):
            raise ValueError(
                f"partial node {node_id!r} must have fewer than target children"
            )
        if status == "expanded" and (
            child_count == 0
            or (target_count is not None and child_count != target_count)
        ):
            raise ValueError(
                f"expanded node {node_id!r} must have the target child count"
            )

    ordered_root_children = sorted(
        root_children, key=lambda child: child["sibling_index"]
    )
    root_sibling_indices = [
        child["sibling_index"] for child in ordered_root_children
    ]
    if root_sibling_indices != list(range(len(ordered_root_children))):
        raise ValueError("depth-1 sibling indices must be contiguous")
    if (
        canonical_branching is not None
        and len(root_children) != canonical_branching[0]
    ):
        raise ValueError("depth-1 node count does not match root branching")

    canonical_nodes = preorder_nodes(tree)
    if [node["id"] for node in nodes] != [
        node["id"] for node in canonical_nodes
    ]:
        raise ValueError("subq tree nodes must use canonical preorder")


def _validate_answer_record(
    record: Any,
    label: str,
    required_metrics: Sequence[str],
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be an object")
    required = {"answer", "confidence", "status", "failure_reason"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"{label} missing keys {sorted(missing)}")

    status = record["status"]
    if status not in ("valid", "invalid"):
        raise ValueError(f"{label}.status must be valid or invalid")
    answer = record["answer"]
    failure_reason = record["failure_reason"]
    if status == "valid":
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{label}.answer must be non-empty when valid")
        if failure_reason is not None:
            raise ValueError(f"{label}.failure_reason must be null when valid")
    elif not isinstance(failure_reason, str) or not failure_reason.strip():
        raise ValueError(
            f"{label}.failure_reason must be non-empty when invalid"
        )
    validate_confidence_mapping(
        record["confidence"],
        f"{label}.confidence",
        required_metrics,
    )


def _support_node_ids(record: Mapping, label: str, child_ids: set) -> list:
    support_node_ids = record.get("support_node_ids")
    if not isinstance(support_node_ids, list) or not all(
        isinstance(node_id, str) for node_id in support_node_ids
    ):
        raise ValueError(f"{label}.support_node_ids must be a string list")
    if len(support_node_ids) != len(set(support_node_ids)):
        raise ValueError(f"{label}.support_node_ids contains duplicates")
    if any(node_id not in child_ids for node_id in support_node_ids):
        raise ValueError(
            f"{label}.support_node_ids must reference immediate children"
        )
    return support_node_ids


def validate_suba_artifact_entry(
    entry: Mapping[str, Any],
    tree: Mapping[str, Any],
    hierarchy: HierarchyConfig,
) -> None:
    """Validate one canonical SubA record, including deterministic selection."""
    if not isinstance(hierarchy, HierarchyConfig) or not hierarchy.enabled:
        raise ValueError("hierarchical SubA validation requires depth > 1")
    if not isinstance(entry, Mapping):
        raise ValueError("SubA artifact entry must be an object")
    validate_subq_tree(tree, branching=hierarchy.branching)
    answers_by_node = entry.get("answers_by_node")
    if not isinstance(answers_by_node, Mapping):
        raise ValueError("answers_by_node must be an object")
    expected_node_ids = [node["id"] for node in tree["nodes"]]
    if set(answers_by_node) != set(expected_node_ids):
        raise ValueError(
            "answers_by_node keys must exactly match subq_tree node IDs"
        )

    required_metrics = hierarchy.required_confidence_types
    for node in tree["nodes"]:
        node_id = node["id"]
        label = f"answers_by_node[{node_id!r}]"
        answer_entry = answers_by_node[node_id]
        if not isinstance(answer_entry, Mapping):
            raise ValueError(f"{label} must be an object")
        missing = {"direct", "candidates", "selected"} - set(answer_entry)
        if missing:
            raise ValueError(f"{label} missing keys {sorted(missing)}")

        direct = answer_entry["direct"]
        _validate_answer_record(direct, f"{label}.direct", required_metrics)
        candidates = answer_entry["candidates"]
        if not isinstance(candidates, list):
            raise ValueError(f"{label}.candidates must be a list")
        if len(candidates) > hierarchy.suba_k:
            raise ValueError(f"{label}.candidates exceeds configured runner.suba_K")

        child_id_set = set(node["child_ids"])
        for index, candidate in enumerate(candidates):
            candidate_label = f"{label}.candidates[{index}]"
            _validate_answer_record(candidate, candidate_label, required_metrics)
            support = _support_node_ids(candidate, candidate_label, child_id_set)
            if len(support) != hierarchy.suba_m:
                raise ValueError(
                    f"{candidate_label}.support_node_ids must contain "
                    "runner.suba_M children"
                )

        selected = answer_entry["selected"]
        _validate_answer_record(selected, f"{label}.selected", required_metrics)
        selected_support = _support_node_ids(
            selected, f"{label}.selected", child_id_set
        )
        source = selected.get("source")
        if source not in ("leaf_direct", "confidence", "direct_fallback"):
            raise ValueError(f"{label}.selected.source is invalid")

        is_leaf = node["depth"] == tree["max_depth"]
        if is_leaf and source != "leaf_direct":
            raise ValueError(f"{label}.selected source must be leaf_direct")
        if not is_leaf and source == "leaf_direct":
            raise ValueError(
                f"{label}.selected leaf_direct source is only valid for leaves"
            )
        if source in ("leaf_direct", "direct_fallback"):
            if selected_support:
                raise ValueError(
                    f"{label}.selected direct source cannot have support nodes"
                )
            for key in ("answer", "confidence", "status", "failure_reason"):
                if selected[key] != direct[key]:
                    raise ValueError(
                        f"{label}.selected direct source differs from direct"
                    )
        else:
            candidate_index = selected.get("candidate_index")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index < 0
                or candidate_index >= len(candidates)
            ):
                raise ValueError(
                    f"{label}.selected.candidate_index is out of range"
                )
            candidate = candidates[candidate_index]
            if candidate["status"] != "valid":
                raise ValueError(f"{label}.selected candidate must be valid")
            for key in ("answer", "confidence", "status", "failure_reason"):
                if selected[key] != candidate[key]:
                    raise ValueError(
                        f"{label}.selected differs from its candidate"
                    )
            if selected_support != candidate["support_node_ids"]:
                raise ValueError(
                    f"{label}.selected support differs from its candidate"
                )

    for node in reversed(tree["nodes"]):
        node_id = node["id"]
        label = f"answers_by_node[{node_id!r}]"
        answer_entry = answers_by_node[node_id]
        candidates = answer_entry["candidates"]
        selected = answer_entry["selected"]
        is_leaf = node["depth"] == tree["max_depth"]
        if is_leaf:
            if candidates:
                raise ValueError(f"{label} leaf candidate count must be zero")
            continue

        valid_child_ids = [
            child_id
            for child_id in node["child_ids"]
            if answers_by_node[child_id]["selected"]["status"] == "valid"
        ]
        expected_subsets = [
            list(subset)
            for subset in evidence_subsets(
                valid_child_ids, hierarchy.suba_m, hierarchy.suba_k
            )
        ]
        actual_subsets = [candidate["support_node_ids"] for candidate in candidates]
        for support in actual_subsets:
            if any(child_id not in valid_child_ids for child_id in support):
                raise ValueError(
                    f"{label}.candidate support must reference valid child "
                    "selected answers"
                )
        if len(actual_subsets) != len(expected_subsets):
            raise ValueError(
                f"{label} candidate count does not match deterministic subsets"
            )
        if actual_subsets != expected_subsets:
            raise ValueError(
                f"{label} candidates must use lexicographic support subsets"
            )

        best = select_best_candidate(candidates, hierarchy.confidence_type)
        if best is None:
            if selected["source"] != "direct_fallback":
                raise ValueError(
                    f"{label}.selected must use direct_fallback without a "
                    "valid candidate"
                )
            continue
        _, normalized_score, candidate_index = best
        if (
            selected["source"] != "confidence"
            or selected.get("candidate_index") != candidate_index
        ):
            raise ValueError(
                f"{label}.selected must use the confidence first maximum"
            )
        normalized_confidence = selected.get("normalized_confidence")
        if (
            isinstance(normalized_confidence, bool)
            or not isinstance(normalized_confidence, (int, float))
            or not math.isfinite(float(normalized_confidence))
            or float(normalized_confidence) != normalized_score
        ):
            raise ValueError(
                f"{label}.selected.normalized_confidence does not match "
                "the confidence first maximum"
            )

    depth_one_ids = [str(index) for index in range(hierarchy.branching[0])]
    selected_depth_one = [
        answers_by_node[node_id]["selected"] for node_id in depth_one_ids
    ]
    if any(record["status"] != "valid" for record in selected_depth_one):
        raise ValueError("depth-1 selected answers must all be valid")
    expected_answers = [record["answer"] for record in selected_depth_one]
    if entry.get("suba_list") != expected_answers:
        raise ValueError("suba_list is not the exact depth-1 answer projection")

    projected_metrics = [
        metric
        for metric in SUPPORTED_CONFIDENCE_TYPE_ORDER
        if all(metric in record["confidence"] for record in selected_depth_one)
    ]
    expected_confidence = {
        metric: [record["confidence"][metric] for record in selected_depth_one]
        for metric in projected_metrics
    }
    if entry.get("conf_suba") != expected_confidence:
        raise ValueError(
            "conf_suba is not the exact depth-1 confidence projection"
        )


def project_depth_one_questions(tree: Mapping[str, Any], expected_count: int) -> list:
    """Project canonical depth-1 questions for the legacy refined pipeline."""
    expected_count = _positive_integer(expected_count, "expected depth-1 count")
    validate_subq_tree(tree)
    depth_one = sorted(
        (node for node in tree["nodes"] if node["depth"] == 1),
        key=lambda node: node["sibling_index"],
    )
    expected_ids = [str(index) for index in range(expected_count)]
    if [node["id"] for node in depth_one] != expected_ids:
        raise ValueError(
            f"incomplete depth-1 projection: expected IDs {expected_ids}"
        )
    return [node["question"] for node in depth_one]

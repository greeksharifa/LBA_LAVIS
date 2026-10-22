#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OUT_ROOT="${OUT_ROOT:-ywjang_output_sq_epoch3_fraction}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPUS="${GPUS:-2}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.10}"
BASE_CKPT="${BASE_CKPT:-/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth}"
ANSWERER_CKPT="${ANSWERER_CKPT:-$BASE_CKPT}"
VICUNA_PATH="${VICUNA_PATH:-/model/lmsys/vicuna-7b-v1.1}"
ANN_TRAIN="${ANN_TRAIN:-/home/ywjang/data/VQA-Introspect/VQAIntrospect_trainv1.0.json}"
ANN_VAL="${ANN_VAL:-/home/ywjang/data/VQA-Introspect/VQAIntrospect_valv1.0.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/data/coco/images}"
EVAL_SAMPLE_FRACTION="${EVAL_SAMPLE_FRACTION:-0.1}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-42}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-1}"
ACCUM_GRAD_ITERS="${ACCUM_GRAD_ITERS:-128}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1}"
EVAL_DEVICES="${EVAL_DEVICES:-cuda:0 cuda:1}"
read -r -a EVAL_DEVICE_ARGS <<< "$EVAL_DEVICES"
EVAL_DEVICE="${EVAL_DEVICE:-${EVAL_DEVICE_ARGS[0]}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

versions=(small medium large full)
fractions=(0.125 0.25 0.5 1.0)
roles=(questioner reasoner)

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPUS=$GPUS OUT_ROOT=$OUT_ROOT max_epoch=3 eval_checkpoint=2"
echo "effective_train_batch=$(( BATCH_SIZE_TRAIN * GPUS * ACCUM_GRAD_ITERS ))"
echo "eval_sample_fraction=$EVAL_SAMPLE_FRACTION eval_sample_seed=$EVAL_SAMPLE_SEED"
echo "eval_devices=$EVAL_DEVICES"
echo "ann_train=$ANN_TRAIN ann_val=$ANN_VAL image_root=$IMAGE_ROOT"

run_or_print() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_train_one() {
  local role_lc="$1"
  local version="$2"
  local fraction="$3"
  local cfg="configs/sq_instructblip/${role_lc}.yaml"
  local role prompt_type

  case "$role_lc" in
    questioner)
      role="Questioner"
      prompt_type="Questioner_MultipleSubQ"
      ;;
    reasoner)
      role="Reasoner"
      prompt_type="Reasoner"
      ;;
    *)
      echo "Unknown role: $role_lc" >&2
      exit 2
      ;;
  esac

  local existing_dir
  if existing_dir="$(latest_checkpoint_dir "$role_lc" "$version" 2>/dev/null)"; then
    echo "Skip training ${role_lc}_${version}: found ${existing_dir}/checkpoint_2.pth"
    return 0
  fi

  run_or_print \
    uv run --python "$UV_PYTHON_VERSION" python -m torch.distributed.run \
      --nproc_per_node="$GPUS" \
      train.py \
      --cfg-path "$cfg" \
      --options \
      datasets.vqa_introspect_qar_caption.sample_fraction="$fraction" \
      datasets.vqa_introspect_qar_caption.sample_seed=42 \
      datasets.vqa_introspect_qar_caption.eval_max_samples=null \
      datasets.vqa_introspect_qar_caption.eval_sample_fraction=null \
      datasets.vqa_introspect_qar_caption.prompt_type="$prompt_type" \
      datasets.vqa_introspect_qar_caption.build_info.annotations.train.storage="[$ANN_TRAIN]" \
      datasets.vqa_introspect_qar_caption.build_info.annotations.val.storage="[$ANN_VAL]" \
      datasets.vqa_introspect_qar_caption.build_info.annotations.test.storage="[$ANN_VAL]" \
      datasets.vqa_introspect_qar_caption.build_info.images.storage="$IMAGE_ROOT" \
      model.role="$role" \
      model.finetuned="$BASE_CKPT" \
      model.llm_model="$VICUNA_PATH" \
      run.max_epoch=3 \
      run.batch_size_train="$BATCH_SIZE_TRAIN" \
      run.batch_size_eval="$BATCH_SIZE_EVAL" \
      run.accum_grad_iters="$ACCUM_GRAD_ITERS" \
      run.world_size="$GPUS" \
      run.valid_splits=[] \
      run.test_splits=[] \
      run.evaluate=False \
      run.output_dir="${OUT_ROOT}/${role_lc}_${version}"
}

latest_checkpoint_dir() {
  local role_lc="$1"
  local version="$2"
  local root="lavis/${OUT_ROOT}/${role_lc}_${version}"

  if [[ ! -d "$root" ]]; then
    return 1
  fi

  local dir
  while IFS= read -r dir; do
    if [[ -f "$dir/checkpoint_2.pth" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
  done < <(ls -dt "$root"/* 2>/dev/null)

  return 1
}

run_eval_one() {
  local version="$1"
  local fraction="$2"
  local q_dir r_dir eval_dir

  if ! q_dir="$(latest_checkpoint_dir questioner "$version")"; then
    if [[ "$DRY_RUN" == "1" ]]; then
      q_dir="lavis/${OUT_ROOT}/questioner_${version}/DRY_RUN_RUN_DIR"
    else
      echo "Missing questioner checkpoint_2 for $version" >&2
      exit 3
    fi
  fi
  if ! r_dir="$(latest_checkpoint_dir reasoner "$version")"; then
    if [[ "$DRY_RUN" == "1" ]]; then
      r_dir="lavis/${OUT_ROOT}/reasoner_${version}/DRY_RUN_RUN_DIR"
    else
      echo "Missing reasoner checkpoint_2 for $version" >&2
      exit 3
    fi
  fi
  eval_dir="lavis/${OUT_ROOT}/qar_${version}_ckpt2_eval"

  run_or_print \
    uv run --python "$UV_PYTHON_VERSION" python scripts/eval_qar_checkpoint_sweep.py \
      --cfg-path configs/sq_instructblip/reasoner.yaml \
      --questioner-dir "$q_dir" \
      --reasoner-dir "$r_dir" \
      --answerer-checkpoint "$ANSWERER_CKPT" \
      --llm-model "$VICUNA_PATH" \
      --ann-val "$ANN_VAL" \
      --image-root "$IMAGE_ROOT" \
      --output-dir "$eval_dir" \
      --eval-sample-fraction "$EVAL_SAMPLE_FRACTION" \
      --eval-sample-seed "$EVAL_SAMPLE_SEED" \
      --device "$EVAL_DEVICE" \
      --devices "${EVAL_DEVICE_ARGS[@]}" \
      --checkpoint-start 2 \
      --checkpoint-end 2 \
      --summary-prefix summary_ckpt2
}

if [[ "$SKIP_TRAIN" != "1" ]]; then
  for i in "${!versions[@]}"; do
    for role_lc in "${roles[@]}"; do
      run_train_one "$role_lc" "${versions[$i]}" "${fractions[$i]}"
    done
  done
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  for i in "${!versions[@]}"; do
    run_eval_one "${versions[$i]}" "${fractions[$i]}"
  done
fi

if [[ "$DRY_RUN" != "1" && "$SKIP_EVAL" != "1" ]]; then
  python3 - <<'PY_SUMMARY'
import csv
import json
import os
from pathlib import Path

out_root = os.environ["OUT_ROOT"]
root = Path("lavis") / out_root
versions = ["small", "medium", "large", "full"]
fractions = ["0.125", "0.25", "0.5", "1.0"]

rows = []
for version, fraction in zip(versions, fractions):
    path = root / f"qar_{version}_ckpt2_eval" / "summary_ckpt2.json"
    if not path.is_file():
        raise SystemExit(f"missing evaluation summary: {path}")
    summary = json.load(open(path))
    if len(summary) != 1 or int(summary[0]["checkpoint"]) != 2:
        raise SystemExit(f"unexpected summary contents in {path}: {summary}")
    item = dict(summary[0])
    item["version"] = version
    item["train_fraction"] = fraction
    rows.append(item)

json_path = root / "summary_epoch3_ckpt2.json"
csv_path = root / "summary_epoch3_ckpt2.csv"
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)
with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "version",
        "train_fraction",
        "checkpoint",
        "num_examples",
        "accuracy",
        "questioner_checkpoint",
        "reasoner_checkpoint",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {json_path}")
print(f"wrote {csv_path}")
PY_SUMMARY
fi

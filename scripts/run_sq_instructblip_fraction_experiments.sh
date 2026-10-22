#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

GPUS="${GPUS:-4}"
BASE_CKPT="${BASE_CKPT:-/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth}"
VICUNA_PATH="${VICUNA_PATH:-/model/lmsys/vicuna-7b-v1.1}"
DRY_RUN="${DRY_RUN:-0}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.10}"

versions=(small medium large full)
fractions=(0.125 0.25 0.5 1.0)
roles=(questioner reasoner)

run_one() {
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

  local -a cmd=(
    uv run --python "$UV_PYTHON_VERSION" python -m torch.distributed.run
    --nproc_per_node="$GPUS"
    train.py
    --cfg-path "$cfg"
    --options
    datasets.vqa_introspect_qar_caption.sample_fraction="$fraction"
    datasets.vqa_introspect_qar_caption.sample_seed=42
    datasets.vqa_introspect_qar_caption.eval_max_samples=null
    datasets.vqa_introspect_qar_caption.prompt_type="$prompt_type"
    model.role="$role"
    model.finetuned="$BASE_CKPT"
    model.llm_model="$VICUNA_PATH"
    run.world_size="$GPUS"
    run.output_dir="ywjang_output_sq_fraction/${role_lc}_${version}"
  )

  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
}

for i in "${!versions[@]}"; do
  for role_lc in "${roles[@]}"; do
    run_one "$role_lc" "${versions[$i]}" "${fractions[$i]}"
  done
done

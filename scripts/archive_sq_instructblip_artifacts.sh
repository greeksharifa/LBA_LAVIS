#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/archive_sq_instructblip_artifacts.sh [--dry-run|--execute]

Moves pre-fraction SQ-InstructBLIP artifacts into archived/sq_instructblip_pre_fraction_<timestamp>/.
The default is --dry-run.
EOF
}

mode="dry-run"
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run) mode="dry-run" ;;
    --execute) mode="execute" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date '+%Y%m%d_%H%M%S')"
archive_dir="archived/sq_instructblip_pre_fraction_${timestamp}"

shopt -s nullglob
targets=(
  lavis/ywjang_output_*
  output
  thesis
  thesis_smoke
  temp
  subqa/self
  ywjang_log.txt
  baseline_prompts_single_subq.json
  prompts_backup.json
)
shopt -u nullglob

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No matching artifacts to archive."
  exit 0
fi

for target in "${targets[@]}"; do
  case "$target" in
    data|data/*|tests|tests/*|configs|configs/*|lavis/datasets/*|lavis/models/*|lavis/tasks/*|lavis/runners/*)
      echo "Refusing to archive source/data/test/config path: $target" >&2
      exit 1
      ;;
  esac
  echo "$target -> $archive_dir/$target"
done

if [[ "$mode" == "dry-run" ]]; then
  echo "Dry run only. Re-run with --execute to move these artifacts."
  exit 0
fi

mkdir -p "$archive_dir"
for target in "${targets[@]}"; do
  mkdir -p "$archive_dir/$(dirname "$target")"
  mv -- "$target" "$archive_dir/$target"
done

echo "Archived ${#targets[@]} artifact(s) under $archive_dir"

# Hierarchical Sub-QA — Completed

## What Shipped

- 전역 기본값 `N=4, M=2, K=4`와 depth/branching/internal-candidate 설정 검증
- model-independent `GenerationResult`/`generate_results()` protocol과 Qwen text/image/video template 경계
- stable preorder SubQ tree, top-down expansion, direct-first bottom-up SubA refinement, deterministic confidence selection과 fallback
- schema v2 tree/node-answer artifacts, depth-1 flat projection, hierarchy namespace/manifest/evaluator compatibility
- model/dataset 이름에 의존하지 않는 fake contract matrix와 실제 Qwen2.5-VL/MMMU 1문항 acceptance

## Why Closed

Focused 134 tests와 full discovery 190 tests가 통과했고, 변경 Python compile 및 `git diff --check`가 성공했다. 독립 리뷰에서 발견된 confidence union, deterministic SubA validation, qid ordering 세 항목을 수정한 뒤 새 blocker/high finding이 없음을 재확인했다.

GPU 6의 `output/smoke-hierarchical/MMMU/qwen2.5-vl-7b/val/N=4_M=2_K=4/D=2_H=c60e804cccb6` run은 네 stage를 current lineage로 완료했다. `dev_Accounting_1` tree는 depth-1 4개/depth-2 12개였고, 각 depth-1 node는 2개 child evidence를 사용하는 후보 3개 중 confidence first maximum을 선택했다. refined loader는 hierarchy detail 없이 flat SubQ/SubA 각 4개만 전달했다.

## Canonical References

- [Design](../../superpowers/specs/2026-08-01-hierarchical-subqa-design.md)
- [Implementation plan](../../superpowers/plans/2026-08-01-hierarchical-subqa-plan.md)
- [User commands and artifact layout](../../../README.md)

## Residual Risk

Qwen3, MMLU, DramaQA 실제 실행, full MMMU 150/900, accuracy improvement는 이 feature의 non-goal이다. 변경 전부터 존재한 `dataset/DramaQA.py:63` SyntaxError 때문에 repository-wide compileall은 실패하지만, 이번 변경 파일 compile과 전체 tests는 통과했다. GPU smoke의 vLLM profiling/process-group warnings는 stage completion이나 artifact integrity에 영향을 주지 않았다.


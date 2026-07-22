# Third-party local patches

The uncertainty-v2 checkout contains complete third-party clones with mostly
line-ending-only working-tree changes. Entire clones are intentionally not
vendored here. The small semantic changes are preserved as patches against
their original upstream revisions:

| Patch | Upstream | Base revision | Purpose |
| --- | --- | --- | --- |
| `ask-anything-videochat2-requirements.patch` | `https://github.com/OpenGVLab/Ask-Anything.git` | `0a79339cd4454f0584db0bca79d61737acc8166b` | Let the enclosing environment provide Torch/Torchvision. |
| `vllm-deepseek-vl2.patch` | `https://github.com/vllm-project/vllm.git` | `0f465ab53303fbd3c8ad32163db161cdb0cf8dad` | Select the DeepSeek-VL2 small multimodal example by default. |
| `deepseek-vl2-sentencepiece.patch` | `https://github.com/deepseek-ai/DeepSeek-VL2.git` | `ff23960c5cf9e6874b44be38af930cfb0ccbb620` | Remove the incompatible strict SentencePiece pin. |

Each patch is a forward patch from the exact base revision in the table. From
the root of a clean matching upstream checkout, verify and apply it with:

```bash
git checkout --detach <base-revision>
git apply --check /path/to/<patch>
git apply /path/to/<patch>
git apply --check --reverse /path/to/<patch>
```

The final command verifies that the applied tree contains the patch and can be
reversed; it is not a command to run against the untouched base. The
corresponding upstream licenses are copied under `licenses/`.

Environment-specific absolute model paths, notebook experiments, and
line-ending-only changes were not retained.

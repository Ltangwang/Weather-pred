# Examples

Short scripts to validate install and run a minimal probabilistic pipeline.
For the full experiment schedule, see [docs/WORKFLOW.md](../docs/WORKFLOW.md).

| Script | Time | Purpose |
|--------|------|---------|
| `smoke_probabilistic.sh` | ~5–15 min | 2-epoch CRPS train + limited val/test |
| `eval_only_demo.sh` | ~10 min | Full test eval only (`RUN_DIR` must contain `checkpoints/best.pth`) |

Set `DATA_ROOT` and `OUT_DIR` before running (defaults assume OpenSTL as a sibling repo).

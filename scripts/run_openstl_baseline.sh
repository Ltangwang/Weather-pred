#!/usr/bin/env bash
# Optional legacy wrapper — prefer Python: ``python scripts/run_deterministic.py ...``
# Run OpenSTL deterministic WeatherBench (t2m 5.625°) baseline with full console logs
# plus a plaintext copy under Weather-pred/results/baseline_logs/.
#
# Prerequisites:
#   - OpenSTL clone at OPENSTL_ROOT with pip install -e .
#   - Data: OPENSTL_ROOT/data/weather_5_625deg/2m_temperature/*.nc
#
# View output:
#   - Live: printed to stdout (stderr merged)
#   - File: RESULTS_ROOT/baseline_logs/<ex_name>_YYYYMMDD_HHMMSS.log
#   - Per epoch summary from OpenSTL: "Epoch ... | Train Loss | Vali Loss" (normalized MSE loss)
#   - After testing: prints denormalized MAE / RMSE (Kelvin if mean/std applied in metric)
#
# Typical full runs (TASK 3):
#   bash scripts/run_openstl_baseline.sh simvp full
#   bash scripts/run_openstl_baseline.sh convlstm full
#
# Quick smoke (~1 epoch):
#   bash scripts/run_openstl_baseline.sh simvp smoke
set -euo pipefail

BACKBONE="${1:?usage: run_openstl_baseline.sh <simvp|convlstm> [smoke|full]}"
MODE="${2:-full}"

OPENSTL_ROOT="${OPENSTL_ROOT:-/root/autodl-tmp/OpenSTL}"
PROJ="${WEATHER_PRED_ROOT:-/root/autodl-tmp/Weather-pred}"
RESULTS_ROOT="$PROJ/results"
LOGDIR="$RESULTS_ROOT/baseline_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORKDIR="$RESULTS_ROOT/openstl_work_dirs"

export PYTHONUNBUFFERED=1

EPOCHS=50
POSTFIX="det_baseline"
case "$MODE" in
  smoke) EPOCHS=1; POSTFIX="smoke" ;;
  full)  EPOCHS=50; POSTFIX="det_baseline" ;;
  *) echo "Unknown mode: $MODE (use smoke|full)"; exit 2 ;;
 esac

case "$BACKBONE" in
  simvp)
    METHOD=SimVP
    CFG="$OPENSTL_ROOT/configs/weather/t2m_5_625/SimVP_gSTA.py"
    EX_NAME="simvp_${POSTFIX}_${STAMP}"
    ;;
  convlstm)
    METHOD=ConvLSTM
    CFG="$OPENSTL_ROOT/configs/weather/t2m_5_625/ConvLSTM.py"
    EX_NAME="convlstm_${POSTFIX}_${STAMP}"
    ;;
  *)
    echo "Unknown backbone: $BACKBONE (use simvp|convlstm)"; exit 2 ;;
 esac

mkdir -p "$LOGDIR" "$WORKDIR"
LOG_FILE="$LOGDIR/${EX_NAME}.log"

TRAIN_HELPER="$PROJ/scripts/train_openstl_no_tensorboard.py"
if [[ ! -f "$TRAIN_HELPER" ]]; then
  echo "Missing $TRAIN_HELPER"; exit 1
fi

echo "=============================================="
echo "OpenSTL deterministic baseline"
echo "  backbone:       $METHOD"
echo "  cfg:             $CFG"
echo "  dataname:        weather_t2m_5_625"
echo "  data_root:       $OPENSTL_ROOT/data"
echo "  epochs:          $EPOCHS"
echo "  ex_name:         $EX_NAME"
echo "  work_dir:        $WORKDIR/$EX_NAME"
echo "  console+log:     tee $LOG_FILE"
echo "  - Live follow-up:  tail -f $LOG_FILE"
echo "  - Per-epoch line:  EpochEndCallback prints Lr, Train Loss, Vali Loss (standardized MSE, not Kelvin)."
echo "  - After testing:   one line \"mse:, rmse:, mae:\" in physical units (t2m = Kelvin)."
echo "  - CSV Lightning:    <work_dir>/<ex_name>/lightning_csv/version_*/metrics.csv"
echo "  - Text copy:        <work_dir>/<ex_name>/train_*.log  (SetupCallback)"
echo "  Important: Do not pipe this command to \\`tail\\`; keep full stdout/file so metrics flush."
echo "=============================================="

cd "$OPENSTL_ROOT"
export OPENSTL_ROOT
PYTHON_CMD=(python -u "$TRAIN_HELPER" \
  -d weather_t2m_5_625 \
  --data_root ./data \
  -m "$METHOD" \
  -c "$CFG" \
  --ex_name "$EX_NAME" \
  --res_dir "$WORKDIR" \
  -e "$EPOCHS" \
  --batch_size 16 \
  --val_batch_size 16 \
  --num_workers 4 \
  --seed 42 \
  --log_step 1)

if command -v stdbuf >/dev/null 2>&1; then
  PYTHON_CMD=("stdbuf" "-oL" "-eL" "${PYTHON_CMD[@]}")
fi
set -o pipefail
"${PYTHON_CMD[@]}" 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"

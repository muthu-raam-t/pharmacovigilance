#!/usr/bin/env bash
# Trains and evaluates backbones with OOM-safe settings (batch_size=4, max_length=384, grad_accum=2).
# Fits comfortably inside 6GB VRAM for all models including RoBERTa.
set -u
cd "$(dirname "$0")"
REPO="${WORKSPACE_DIR:-$(cd .. && pwd)}"
CKPT_DIR="$REPO/models/checkpoints"
LOG_DIR="$REPO/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$REPO/results"

if [ ! -f "$REPO/results/backbones.json" ]; then
  python check_backbones.py || { echo "Backbone check failed - fix the missing models first."; exit 1; }
fi

if [ "$#" -gt 0 ]; then
  KEYS="$*"
else
  KEYS="bert_base roberta scibert biobert pubmedbert"
fi

for KEY in $KEYS; do
  HF_ID=$(python -c "import json,sys; d=json.load(open('$REPO/results/backbones.json')); print(d.get('$KEY',{}).get('hf_id',''))")
  NAME=$(python -c "import json,sys; d=json.load(open('$REPO/results/backbones.json')); print(d.get('$KEY',{}).get('name',''))")
  if [ -z "$HF_ID" ]; then
    echo "Skipping $KEY - not listed in results/backbones.json"
    continue
  fi
  echo "=================================================================="
  echo "[$(date '+%H:%M:%S')] Training $NAME ($HF_ID)"
  echo "=================================================================="
  if python train_joint_model.py --backbone "$HF_ID" --model_name "$NAME" \
        --batch_size 4 --grad_accum 2 --max_length 384 \
        --output_dir "$CKPT_DIR/$KEY" > "$LOG_DIR/train_$KEY.log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] Evaluating $NAME"
    python evaluate_model.py --backbone "$HF_ID" --model_name "$NAME" --model_key "$KEY" \
        --batch_size 4 --max_length 384 \
        --checkpoint "$CKPT_DIR/$KEY/model.pt" > "$LOG_DIR/eval_$KEY.log" 2>&1 \
        && tail -n 5 "$LOG_DIR/eval_$KEY.log" \
        || echo "Evaluation of $NAME FAILED - see $LOG_DIR/eval_$KEY.log"
  else
    echo "Training of $NAME FAILED - see $LOG_DIR/train_$KEY.log (continuing with the next model)"
  fi
done

echo "[$(date '+%H:%M:%S')] Generating comparison figures"
python generate_metrics_images.py || echo "Figure generation failed"
echo "[$(date '+%H:%M:%S')] ALL DONE"
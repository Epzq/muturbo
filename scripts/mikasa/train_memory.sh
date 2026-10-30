#!/usr/bin/env bash
# muTurboVLA: train TurboVLA with muVLA-style recurrent memory on the five MIKASA training tasks.
# Same recipe as the TurboVLA 256px MIKASA baseline (global 256 samples / update, 40k steps).
#
# Usage (any directory, turbovla-libero env):
#   bash scripts/mikasa/train_memory.sh
#   MEMORY_TOKENS=64 TBPTT=2 GPUS=2,3 STREAMS=32 GRAD_ACCUM=2 bash scripts/mikasa/train_memory.sh
#   MEMORY_UPDATE=ema EMA_ALPHA=0.1 bash scripts/mikasa/train_memory.sh
# Re-running resumes from the latest checkpoint in CKPT_DIR (--resume_mode all).

set -euo pipefail

MEMORY_TOKENS=${MEMORY_TOKENS:-32}
MEMORY_UPDATE=${MEMORY_UPDATE:-tbptt}   # tbptt | ema
TBPTT=${TBPTT:-2}                       # consecutive steps per window (K)
EMA_ALPHA=${EMA_ALPHA:-0.1}             # ema only
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-29511}
STREAMS=${STREAMS:-64}                  # parallel episode streams per GPU (= --batch_size)
GRAD_ACCUM=${GRAD_ACCUM:-1}             # windows per optimizer update
MAX_STEPS=${MAX_STEPS:-40000}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-4000}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
DATA_ROOT=${DATA_ROOT:-/data/nrf3/usrdata/eric/dataset/mikasa-robo-vla-rlds}
OUT_ROOT=${OUT_ROOT:-/home/eric/usrdata/llm_weights/muTurboVLA}
TAG=${TAG:-mem${MEMORY_TOKENS}_${MEMORY_UPDATE}${TBPTT}}
CKPT_DIR=${CKPT_DIR:-${OUT_ROOT}/mikasa5_${TAG}}
TASKS=(shell_game_push_vla_v0 intercept_medium_vla_v0 remember_color_5_vla_v0 take_it_back_vla_v0 remember_shape_and_color_3x3_vla_v0)

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # use muTurboVLA's turbovla, not the TurboVLA install
python -c "import turbovla, sys; assert turbovla.__file__.startswith('$REPO_ROOT'), turbovla.__file__"

NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
DIRS=$(printf "${DATA_ROOT}/%s/1.0.0," "${TASKS[@]}")
mkdir -p "$CKPT_DIR"
echo "[muTurboVLA] $(date '+%F %T') -> $CKPT_DIR | memory_tokens=$MEMORY_TOKENS update=$MEMORY_UPDATE K=$TBPTT streams/gpu=$STREAMS gpus=$GPUS"

CUDA_VISIBLE_DEVICES="$GPUS" NCCL_P2P_DISABLE=1 torchrun --nproc_per_node="$NPROC" --master_port "$PORT" \
  experiments/mikasa/train_memory.py \
  --dataset_dirs "${DIRS%,}" \
  --stats_path experiments/mikasa/configs/mikasa5_stats.json --stats_key mikasa5 \
  --text_layout_path experiments/mikasa/configs/online_text_layout.json --text_padding_length 44 \
  --resize_images_to 256 --chunk_size 12 \
  --dinov3_path "$REPO_ROOT/pretrained/dinov3" \
  --bert_path google-bert/bert-base-uncased --allow_hf_download \
  --pretrained_init_ckpt "$REPO_ROOT/pretrained/groundingdino/groundingdino_swint_ogc.pth" \
  --batch_size "$STREAMS" --grad_accum_steps "$GRAD_ACCUM" \
  --max_steps "$MAX_STEPS" --warmup_steps "$WARMUP_STEPS" --save_steps "$SAVE_STEPS" \
  --head_lr 1e-4 --dinov3_lr 1e-4 --min_lr_ratio 0.1 \
  --shuffle_buffer 1250 --num_workers "$NUM_WORKERS" --seed "$SEED" --resume_mode all \
  --memory_tokens "$MEMORY_TOKENS" --memory_update "$MEMORY_UPDATE" --tbptt_length "$TBPTT" --memory_ema_alpha "$EMA_ALPHA" \
  --checkpoint_dir "$CKPT_DIR" \
  2>&1 | tee -a "$CKPT_DIR/train.log"

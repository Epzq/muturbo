#!/usr/bin/env bash
# MIKASA history ablations, run in sequence with identical settings:
#   1. image history only   (--obs_history)
#   2. state history only   (--state_history)
#   3. image + state history
# All use the same window (K slots, stride S) and both cameras (base + wrist) for past slots.
#
# Usage (from anywhere, in the turbovla-libero env):
#   bash scripts/mikasa/run_history_ablations.sh
#   K=3 STRIDE=2 GPUS=2,3 bash scripts/mikasa/run_history_ablations.sh
#
# Each run writes to outputs/mikasa5_<name>_k<K>s<S>/ with a train.log. Re-running the script
# resumes an interrupted run from its latest intact checkpoint and skips finished runs.

set -euo pipefail

# ---- settings (override via environment) -------------------------------------------------
K=${K:-3}                         # past window slots
STRIDE=${STRIDE:-1}               # steps between slots
FIRST_FRAME=${FIRST_FRAME:-0}     # 1 = also add episode step 0 as a slot
HISTORY_DROPOUT=${HISTORY_DROPOUT:-0.0}
TIME_EMB=${TIME_EMB:-0}           # 1 = time-ordered views + camera/time embeddings (image-history runs only)
HIST_TOKENS=${HIST_TOKENS:-0}     # 1 = original TurboVLA path + pooled history tokens in the action decoder
HIST_POOL=${HIST_POOL:-1}         # history tokens per past view = HIST_POOL^2
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-29501}
BATCH_SIZE=${BATCH_SIZE:-16}      # per GPU
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-40000}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
SEED=${SEED:-42}
DATA_ROOT=${DATA_ROOT:-/data/nrf3/usrdata/eric/dataset/mikasa-robo-vla-rlds}
GROUNDINGDINO=${GROUNDINGDINO:-/home/eric/TurboVLA/pretrained/groundingdino/groundingdino_swint_ogc.pth}
OUT_ROOT=${OUT_ROOT:-/home/eric/usrdata/llm_weights/TurboVLA}
TASKS=(
  shell_game_push_vla_v0
  intercept_medium_vla_v0
  remember_color_5_vla_v0
  take_it_back_vla_v0
  remember_shape_and_color_3x3_vla_v0
)
# -------------------------------------------------------------------------------------------

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
DIRS=$(printf "${DATA_ROOT}/%s/1.0.0," "${TASKS[@]}")
DIRS=${DIRS%,}

COMMON_ARGS=(
  --dataset_dirs "$DIRS"
  --stats_path experiments/mikasa/configs/mikasa5_stats.json
  --stats_key mikasa5
  --text_layout_path experiments/mikasa/configs/online_text_layout.json
  --text_padding_length 44
  --resize_images_to 256
  --chunk_size 12
  --dinov3_path /home/eric/TurboVLA/pretrained/dinov3
  --bert_path google-bert/bert-base-uncased
  --allow_hf_download
  --pretrained_init_ckpt "$GROUNDINGDINO"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM"
  --max_steps "$MAX_STEPS"
  --warmup_steps "$WARMUP_STEPS"
  --head_lr 1e-4 
  --dinov3_lr 1e-4 
  --min_lr_ratio 0.1
  --save_steps 4000
  --shuffle_buffer 1250
  --seed "$SEED"
  --resume_mode all
  --history_size "$K"
  --history_stride "$STRIDE"
  --history_obs_views both
  --history_dropout "$HISTORY_DROPOUT"
)
TAG="k${K}s${STRIDE}"
if [[ "$FIRST_FRAME" == "1" ]]; then
  COMMON_ARGS+=(--history_first_frame)
  TAG="${TAG}ff"
fi
TIME_EMB_ARGS=()
[[ "$TIME_EMB" == "1" ]] && TIME_EMB_ARGS=(--history_time_embedding)
[[ "$HIST_TOKENS" == "1" ]] && TIME_EMB_ARGS+=(--history_tokens --history_pool "$HIST_POOL")

# Move aside a latest checkpoint that was cut off mid-save, so --resume_mode all can't pick it.
quarantine_truncated_checkpoint() {
  local ckpt_dir=$1 latest
  latest=$(ls "$ckpt_dir"/turbovla_step_*.pth 2>/dev/null | sort -V | tail -1 || true)
  [[ -z "$latest" ]] && return 0
  if ! python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).testzip()" "$latest" 2>/dev/null; then
    echo "[ablations] $latest is truncated; moving it to ${latest}.broken"
    mv "$latest" "${latest}.broken"
    quarantine_truncated_checkpoint "$ckpt_dir"
  fi
}

run_ablation() {
  local name=$1; shift
  local suffix=""
  [[ " $* " == *" --history_time_embedding "* ]] && suffix="te"
  [[ " $* " == *" --history_tokens "* ]] && suffix="${suffix}ht${HIST_POOL}"
  local ckpt_dir="${OUT_ROOT}/mikasa5_${name}_${TAG}${suffix}"
  mkdir -p "$ckpt_dir"
  if [[ -f "${ckpt_dir}/turbovla_step_${MAX_STEPS}.pth" ]]; then
    echo "[ablations] ${name}: finished (turbovla_step_${MAX_STEPS}.pth exists), skipping"
    return 0
  fi
  quarantine_truncated_checkpoint "$ckpt_dir"
  echo "[ablations] ${name}: $(date '+%F %T') -> ${ckpt_dir} (history: $*)"
  CUDA_VISIBLE_DEVICES="$GPUS" NCCL_P2P_DISABLE=1 torchrun --nproc_per_node="$NPROC" --master_port "$PORT" \
    experiments/mikasa/train.py "${COMMON_ARGS[@]}" "$@" --checkpoint_dir "$ckpt_dir" \
    2>&1 | tee -a "${ckpt_dir}/train.log"
  echo "[ablations] ${name}: done $(date '+%F %T')"
}

run_ablation hist_obs       --obs_history "${TIME_EMB_ARGS[@]}"
run_ablation hist_state     --state_history
run_ablation hist_obs_state --obs_history --state_history "${TIME_EMB_ARGS[@]}"

echo "[ablations] all runs finished"

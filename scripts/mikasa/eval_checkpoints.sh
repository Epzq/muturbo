#!/usr/bin/env bash
# Evaluate MIKASA checkpoints on the five training tasks with the muVLA protocol:
# 100 episodes per task, seeds 4242424242 + i, success_once. muVLA reports two execution modes:
#   OPEN_LOOP=1   receding horizon: re-plan every step, execute only the first action
#   OPEN_LOOP=12  open loop: execute the whole predicted chunk before re-planning
# Each (step, open-loop) job runs on its own free GPU, concurrently; jobs wait when no GPU is free.
# Run from the TurboVLA repo root in the turbovla-mikasa env:
#   conda activate turbovla-mikasa
#   bash scripts/mikasa/eval_checkpoints.sh

CKPT_DIR=/home/eric/usrdata/llm_weights/TurboVLA/mikasa5_hist_obs_k0s1ffht1
STEPS="4000 8000"
OPEN_LOOP_STEPS="4 8"
HISTORY_EVAL_MODE=normal          # normal | current | shuffled (current/shuffled: history runs only)
GPUS="0 1 2 3"                    # GPUs this script may use
FREE_MEM_MB=30000                  # a GPU counts as free below this much used memory

OUT_DIR=outputs/mikasa_eval/$(basename ${CKPT_DIR})
declare -A GPU_PID                # GPU index -> PID of the eval job this script started on it

free_gpu() {
  local idx mem
  while IFS=', ' read -r idx mem; do
    [[ " $GPUS " == *" $idx "* ]] || continue
    [[ -n "${GPU_PID[$idx]}" ]] && kill -0 "${GPU_PID[$idx]}" 2>/dev/null && continue
    (( mem < FREE_MEM_MB )) && { echo "$idx"; return 0; }
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  return 1
}

for STEP in $STEPS; do
  for OPEN_LOOP in $OPEN_LOOP_STEPS; do
    until GPU=$(free_gpu); do sleep 60; done
    RUN=step_${STEP}/open_loop_${OPEN_LOOP}_${HISTORY_EVAL_MODE}
    mkdir -p ${OUT_DIR}/${RUN}
    echo "$(date '+%F %T') GPU ${GPU}: ${RUN} (log: ${OUT_DIR}/${RUN}/eval.log)"
    CUDA_VISIBLE_DEVICES=${GPU} python experiments/mikasa/evaluate.py \
      --ckpt_path ${CKPT_DIR}/turbovla_step_${STEP}.pth \
      --dinov3_path /home/eric/TurboVLA/pretrained/dinov3 \
      --bert_path google-bert/bert-base-uncased \
      --stats_path experiments/mikasa/configs/mikasa5_stats.json \
      --stats_key mikasa5 \
      --task ShellGamePush-VLA-v0 \
      --task InterceptMedium-VLA-v0 \
      --task RememberColor5-VLA-v0 \
      --task TakeItBack-VLA-v0 \
      --task RememberShapeAndColor3x3-VLA-v0 \
      --num_episodes 100 \
      --start_seed 4242424242 \
      --num_open_loop_steps ${OPEN_LOOP} \
      --history_eval_mode ${HISTORY_EVAL_MODE} \
      --precision bf16 \
      --seed 7 \
      --no_ui \
      --output_dir ${OUT_DIR}/${RUN} \
      > ${OUT_DIR}/${RUN}/eval.log 2>&1 < /dev/null &
    GPU_PID[$GPU]=$!
  done
done

wait
echo "$(date '+%F %T') all evaluations finished"
grep -E 'SR=.*±|^overall' ${OUT_DIR}/step_*/open_loop_*_${HISTORY_EVAL_MODE}/eval.log

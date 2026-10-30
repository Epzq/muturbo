#!/usr/bin/env bash
# muTurboVLA: evaluate recurrent-memory checkpoints on the five MIKASA training tasks (muVLA protocol:
# 100 episodes per task, seeds 4242424242 + i, success_once, render after every step).
# Memory settings come from the checkpoint. The memory is updated at every env step regardless of
# OPEN_LOOP; muVLA evaluates memory models with re-planning every step (OPEN_LOOP=1).
#   HISTORY_EVAL_MODE=current   -> memory disabled (M_init every step): the "does it use memory" test.
# Each (step, open-loop) job runs on its own free GPU, concurrently; jobs wait when no GPU is free.
#
# Usage (turbovla-mikasa env):
#   bash scripts/mikasa/eval_memory.sh
#   CKPT_DIR=... STEPS="20000 40000" OPEN_LOOP_STEPS="1" HISTORY_EVAL_MODE=current bash scripts/mikasa/eval_memory.sh

CKPT_DIR=${CKPT_DIR:-/home/eric/usrdata/llm_weights/muTurboVLA/mikasa5_mem32_tbptt2}
STEPS=${STEPS:-"12000 20000 28000 36000 40000"}
OPEN_LOOP_STEPS=${OPEN_LOOP_STEPS:-"8"}
HISTORY_EVAL_MODE=${HISTORY_EVAL_MODE:-normal}   # normal | current
GPUS=${GPUS:-"0 1 2 3"}                          # GPUs this script may use
FREE_MEM_MB=${FREE_MEM_MB:-4000}                 # a GPU counts as free below this much used memory
NUM_EPISODES=${NUM_EPISODES:-100}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # use muTurboVLA's turbovla, not the TurboVLA install
python -c "import turbovla; assert turbovla.__file__.startswith('$REPO_ROOT'), turbovla.__file__" || exit 1

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
      --task ShellGamePush-VLA-v0 \
      --task InterceptMedium-VLA-v0 \
      --task RememberColor5-VLA-v0 \
      --task TakeItBack-VLA-v0 \
      --task RememberShapeAndColor3x3-VLA-v0 \
      --num_episodes ${NUM_EPISODES} \
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

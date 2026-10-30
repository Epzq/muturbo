"""Evaluate a TurboVLA checkpoint on the MIKASA-Robo-VLA benchmark.

Uses the official runner (mikasa_robo_suite.vla.benchmarking): canonical env construction,
50 seeded episodes per task starting at seed 4242424242, success_once latching, and per-task
JSON plus summary.json. Run inside an env with ManiSkill + MIKASA-Robo installed
(e.g. turbovla-mikasa), from the TurboVLA repo root.

Default tasks are the five MIKASA tasks used for training; use --task or --split to change.

Examples::

    CUDA_VISIBLE_DEVICES=2 python experiments/mikasa/evaluate.py \\
        --ckpt_path outputs/mikasa5_baseline/turbovla_step_15000.pth

    # history ablation: replace history slots with the current step
    CUDA_VISIBLE_DEVICES=2 python experiments/mikasa/evaluate.py \\
        --ckpt_path outputs/mikasa5_hist/turbovla_step_15000.pth --history_eval_mode current

    # smoke test with videos
    CUDA_VISIBLE_DEVICES=2 python experiments/mikasa/evaluate.py \\
        --ckpt_path ... --task ShellGamePush-VLA-v0 --num_episodes 2 --save_videos
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import mikasa_robo_suite.vla.benchmarking as mikasa_benchmarking
from mikasa_robo_suite.vla.benchmarking import (
    NUM_EPISODES_PER_TASK,
    START_SEED,
    BenchmarkConfig,
    RichBenchmarkUI,
    evaluate_benchmark,
    make_run_dir,
    select_benchmark_tasks,
)

from turbovla.evaluation.mikasa_policy import HISTORY_EVAL_MODES, TurboVLAMikasaPolicy

class RenderEachStep(gym.Wrapper):
    """Call env.render() after every env.step(), as muVLA's MIKASA eval loop does.

    On the GPU sim, scene changes that a task makes inside evaluate() -- e.g. ShellGamePush
    lowering the cups when the cue phase ends -- reach the camera observations only after a
    render update. MIKASA's runner renders only when saving videos, so without this the policy
    never sees the cups come down (ShellGamePush then scores 0). The policy still receives the
    unchanged MIKASA observation stream; this only refreshes what the cameras see.
    """

    def __init__(self, env):
        super().__init__(env)
        self.max_episode_steps = env.max_episode_steps

    def step(self, action):
        result = self.env.step(action)
        self.env.render()
        return result


DEFAULT_TASKS = (
    "ShellGamePush-VLA-v0",
    "InterceptMedium-VLA-v0",
    "RememberColor5-VLA-v0",
    "TakeItBack-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_path", type=str, required=True)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--task",
        action="append",
        dest="tasks",
        metavar="ENV_ID",
        help="Env ID to evaluate; repeat for several. Default: the five training tasks.",
    )
    selection.add_argument("--split", choices=("short", "medium", "long", "all"), help="Evaluate a whole horizon split.")

    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES_PER_TASK)
    parser.add_argument("--start_seed", type=int, default=START_SEED)
    parser.add_argument(
        "--num_open_loop_steps",
        type=int,
        default=None,
        help="Actions executed per prediction before re-planning (default: the model's chunk size).",
    )
    parser.add_argument("--history_eval_mode", choices=HISTORY_EVAL_MODES, default="normal")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--dinov3_path", type=str, default=None, help="Override the checkpoint's DINOv3 path.")
    parser.add_argument("--bert_path", type=str, default=None, help="Override the checkpoint's BERT path.")
    parser.add_argument("--stats_path", type=str, default=None, help="Override the checkpoint's stats file.")
    parser.add_argument("--stats_key", type=str, default=None)
    parser.add_argument("--allow_hf_download", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --history_eval_mode shuffled.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Base directory; results go into a timestamped subdirectory. "
        "Default: outputs/mikasa_eval/<run>_<ckpt>/<history_eval_mode>.",
    )
    parser.add_argument("--resume", type=Path, default=None, metavar="RUN_DIR", help="Resume an interrupted run directory.")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save each episode's predicted chunks, executed actions and proprio to "
        "<run_dir>/predictions/<env_id>/episode_XXXX.npz.",
    )
    parser.add_argument("--sim_backend", default="gpu")
    parser.add_argument(
        "--render_each_step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call env.render() after every step like muVLA's eval (default). Needed for ShellGamePush's "
        "cups to appear in observations; --no-render_each_step reproduces MIKASA's plain runner.",
    )
    parser.add_argument("--no_ui", action="store_true", help="Plain log lines instead of the rich progress display.")
    return parser.parse_args()


def _default_output_dir(args: argparse.Namespace) -> Path:
    ckpt = Path(args.ckpt_path)
    return Path("outputs") / "mikasa_eval" / f"{ckpt.parent.name}_{ckpt.stem}" / args.history_eval_mode


def _load_completed(run_dir: Path) -> list[dict]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--resume directory not found: {run_dir}")
    completed = []
    for path in sorted(run_dir.glob("*.json")):
        if path.stem != "summary":
            completed.append(json.loads(path.read_text(encoding="utf-8")))
    return completed


class _PlainUI:
    def __init__(self, n_episodes: int):
        self.n_episodes = n_episodes

    def on_task_start(self, index, task):
        print(f"[task {index}] {task.env_id} ({task.split}, {task.memory_type}): {task.language_instruction}", flush=True)

    def on_episode_done(self, index, result):
        print(
            f"  episode {index + 1}/{self.n_episodes} seed={result.seed} "
            f"success={result.success_once} steps={result.n_steps}",
            flush=True,
        )

    def on_task_done(self, index, result):
        print(f"[task {index}] {result['env_id']} SR={result['sr']:.2%}", flush=True)


def main() -> None:
    args = parse_args()

    if args.split:
        tasks = select_benchmark_tasks(split=args.split)
    else:
        tasks = select_benchmark_tasks(env_ids=list(args.tasks or DEFAULT_TASKS))

    completed: list[dict] = []
    if args.resume is not None:
        completed = _load_completed(args.resume)
        done = {result["env_id"] for result in completed}
        tasks = [task for task in tasks if task.env_id not in done]
        run_dir = args.resume
    else:
        run_dir = make_run_dir(args.output_dir or _default_output_dir(args))

    if args.render_each_step:
        make_env = mikasa_benchmarking.make_benchmark_env
        mikasa_benchmarking.make_benchmark_env = lambda env_id, config: RenderEachStep(make_env(env_id, config))

    policy = TurboVLAMikasaPolicy(
        ckpt_path=args.ckpt_path,
        dinov3_path=args.dinov3_path,
        bert_path=args.bert_path,
        stats_path=args.stats_path,
        stats_key=args.stats_key,
        precision=args.precision,
        num_open_loop_steps=args.num_open_loop_steps,
        history_eval_mode=args.history_eval_mode,
        seed=args.seed,
        allow_hf_download=args.allow_hf_download,
    )
    config = BenchmarkConfig(
        start_seed=args.start_seed,
        n_episodes=args.num_episodes,
        sim_backend=args.sim_backend,
        save_videos=args.save_videos,
    )

    def run(ui):
        # The runner never resets the policy: start each task with its instruction and clear
        # the history buffer after every episode.
        current = {}

        def on_task_start(index, task):
            current["task"] = task
            policy.start_task(task.language_instruction)
            ui.on_task_start(index, task)

        def on_episode_done(index, result):
            if args.save_predictions:
                task = current["task"]
                path = Path(run_dir) / "predictions" / task.env_id / f"episode_{index:04d}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    **policy.episode_record(),
                    seed=result.seed,
                    success_once=result.success_once,
                    n_steps=result.n_steps,
                    instruction=task.language_instruction,
                )
            policy.reset()
            ui.on_episode_done(index, result)

        return evaluate_benchmark(
            tasks,
            policy,
            config,
            output_dir=run_dir,
            model={**policy.describe(), "render_each_step": args.render_each_step},
            task_start_callback=on_task_start,
            episode_callback=on_episode_done,
            task_done_callback=ui.on_task_done,
            initial_results=completed or None,
        )

    if args.no_ui:
        results, summary = run(_PlainUI(config.n_episodes))
    else:
        with RichBenchmarkUI(tasks, config.n_episodes, initial_results=completed or None) as ui:
            results, summary = run(ui)

    # SR ± standard error over episodes (the muVLA reporting convention), then the task mean.
    print()
    for result in results:
        successes = np.asarray(result["successes"], dtype=np.float64)
        stderr = successes.std(ddof=1) / np.sqrt(len(successes)) if len(successes) > 1 else 0.0
        print(f"{result['env_id']:40s} SR={successes.mean():.2%} ± {stderr:.2%} (n={len(successes)})")
    print(f"{'overall':40s} SR={summary['sr_split']:.2%}")
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()

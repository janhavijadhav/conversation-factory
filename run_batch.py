"""Generate a batch of conversations.

Examples:
    python run_batch.py --n 10 --name steered --batch-size 5
    python run_batch.py --n 10 --name baseline --batch-size 5 --no-steer
    python run_batch.py --n 10 --name steered --overwrite     # replace an existing run
"""
import argparse
import asyncio
import sys
from pathlib import Path

from src.factory import config
from src.factory.llm import LLMClient
from src.factory.runner import BatchRunner
from src.factory.tools.registry import ToolRegistry


def main():
    p = argparse.ArgumentParser(description="Generate synthetic tool-calling conversations")
    p.add_argument("--n", type=int, default=10, help="how many conversations")
    p.add_argument("--name", default="run", help="output folder name under output/")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-steer", action="store_true", help="turn off diversity steering (baseline)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing run with the same name")
    args = p.parse_args()

    out_dir = Path("output") / args.name
    existing = out_dir / "train.jsonl"
    if existing.exists() and existing.stat().st_size > 0 and not args.overwrite:
        print(f"output/{args.name}/ already has results. Use a different --name, "
              f"or add --overwrite to replace them.")
        sys.exit(1)

    assistant_llm = LLMClient()
    # Share one client if the simulator uses the same model, so pacing stays correct.
    simulator_llm = None if config.SIMULATOR_MODEL == config.MODEL_NAME \
        else LLMClient(model=config.SIMULATOR_MODEL)

    runner = BatchRunner(
        ToolRegistry.from_directory(),
        gen_llm=assistant_llm,
        judge_llm=LLMClient(model=config.JUDGE_MODEL),
        simulator_llm=simulator_llm,
        out_dir=out_dir,
        steer=not args.no_steer,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"Assistant: {config.MODEL_NAME} | Simulator: {config.SIMULATOR_MODEL} | Judge: {config.JUDGE_MODEL}")
    m = asyncio.run(runner.run(args.n, name=args.name))

    c = m["corpus"]
    print("\n" + "=" * 60)
    print(f"Run '{m['run']}' (steering={'on' if m['steering'] else 'off'}) in {m['minutes']} min")
    print(f"Passed {m['passed']}/{m['generated']}  (first try {m['first_try_pass_rate']:.0%})")
    print(f"Rewrites {m['rewrites']}, regenerations {m['regenerations']}, errors {len(m['errors'])}")
    print(f"Multi-step {c['multi_step_ratio']:.0%} | mean tool calls {c['mean_tool_calls']} "
          f"| category balance {c['category_balance']}")
    print(f"LLM calls: {m['llm_calls']}")
    print(f"Saved to output/{args.name}/")


if __name__ == "__main__":
    main()

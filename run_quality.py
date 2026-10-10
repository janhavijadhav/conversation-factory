"""Generate one conversation, judge it, and repair it if needed."""
import asyncio
import json
import sys
from pathlib import Path

from run_one import show
from src.factory import config
from src.factory.agents.orchestrator import ConversationGenerator
from src.factory.graph.sampler import ChainSampler
from src.factory.graph.tool_graph import ToolGraph
from src.factory.llm import LLMClient
from src.factory.quality.judge import LLMJudge
from src.factory.quality.repair import QualityPipeline, Rewriter
from src.factory.tools.registry import ToolRegistry


async def main(seed: int):
    registry = ToolRegistry.from_directory()
    chain = ChainSampler(ToolGraph(registry), seed=seed).sample()

    gen_llm = LLMClient()
    judge_llm = LLMClient(model=config.JUDGE_MODEL)
    pipeline = QualityPipeline(
        ConversationGenerator(registry, gen_llm),
        LLMJudge(registry, judge_llm),
        Rewriter(gen_llm),
    )

    print("Planned chain:", chain.pretty())
    print(f"Generator: {gen_llm.model} | Judge: {judge_llm.model}")
    conv = await pipeline.run(chain, seed=seed)

    print("=" * 70)
    show(conv)
    print("=" * 70)

    s = conv.scores
    if s:
        print(f"Scores  correctness={s.correctness} naturalness={s.naturalness} "
              f"groundedness={s.groundedness} completeness={s.completeness}  mean={s.mean:.2f}")
        print(f"Judge says: {s.feedback}")
    print(f"PASSED: {conv.passed} | repair attempts: {conv.repair_attempts}")

    print("\nQuality history:")
    for h in conv.metadata["quality_history"]:
        line = f"  attempt {h['attempt']} [{h['stage']}]"
        if h["stage"] == "judge":
            line += f" mean={h['mean']}"
        if h.get("action"):
            line += f" -> {h['action']}"
        print(line)
        for issue in h.get("issues", []) + [{"problem": p} for p in h.get("problems", [])]:
            where = f"msg {issue['message']}: " if "message" in issue else ""
            print(f"      - {where}{issue.get('problem')}")

    print(f"\nLLM calls: generator={gen_llm.calls} judge={judge_llm.calls}")

    record = conv.to_training_record()
    record["passed"] = conv.passed
    record["quality_history"] = conv.metadata["quality_history"]
    out = Path("output") / f"quality_{seed}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))

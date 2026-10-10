"""Generate one real conversation with the LLM and print it."""
import asyncio
import json
import sys
from pathlib import Path

from src.factory.agents.orchestrator import ConversationGenerator
from src.factory.graph.sampler import ChainSampler
from src.factory.graph.tool_graph import ToolGraph
from src.factory.llm import LLMClient
from src.factory.tools.registry import ToolRegistry


def show(conv):
    for m in conv.messages:
        if m.role == "user":
            print(f"\nUSER: {m.content}")
        elif m.role == "assistant":
            if m.content:
                print(f"\nASSISTANT: {m.content}")
            for tc in m.tool_calls:
                print(f"\nASSISTANT -> {tc.name}({json.dumps(tc.arguments)})")
        elif m.role == "tool":
            text = m.content if len(m.content) < 250 else m.content[:250] + "..."
            print(f"   TOOL <- {text}")


async def main(seed: int):
    registry = ToolRegistry.from_directory()
    sampler = ChainSampler(ToolGraph(registry), seed=seed)
    chain = sampler.sample()

    llm = LLMClient()
    gen = ConversationGenerator(registry, llm)

    print("Planned chain:", chain.pretty())
    conv, session, reason = await gen.generate(chain, seed=seed)

    s = conv.scenario
    print(f"\nPersona: {s.persona}\nGoal: {s.user_goal}\nFacts: {s.notes}")
    print("=" * 70)
    show(conv)
    print("=" * 70)
    print(f"Stop reason: {reason} | tool calls: {conv.num_tool_calls} | LLM calls: {llm.calls}")

    if conv.grounding_fixes:
        print("\nGrounding fixes:")
        for f in conv.grounding_fixes:
            print(f"  {f['tool']}.{f['param']}: {f['old']!r} -> {f['new']!r}  ({f['reason']})")
    else:
        print("Grounding fixes: none needed")

    if conv.validation_errors:
        print("\nBlocked by validator:")
        for v in conv.validation_errors:
            print(f"  [{v['rule']}] {v['message']}")
    else:
        print("Validator: no calls blocked")

    out = Path("output") / f"sample_{seed}.json"
    out.write_text(json.dumps(conv.to_training_record(), indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))

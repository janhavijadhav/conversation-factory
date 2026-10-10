"""Run the judge on a saved conversation file to check the judge itself."""
import asyncio
import json
import sys

from src.factory import config
from src.factory.llm import LLMClient
from src.factory.models import Conversation, Message, Scenario
from src.factory.quality.judge import LLMJudge
from src.factory.quality.repair import QualityPipeline
from src.factory.tools.registry import ToolRegistry


async def main(path: str):
    record = json.loads(open(path).read())
    messages = [Message(**m) for m in record["messages"]]
    first_user = next(m.content for m in messages if m.role == "user")
    tools_used = [tc.name for m in messages for tc in m.tool_calls]

    conv = Conversation(
        scenario=Scenario(user_goal=first_user, tool_chain=tools_used or ["search_flights"],
                          category=record["category"]),
        messages=messages,
    )

    registry = ToolRegistry.from_directory()
    judge = LLMJudge(registry, LLMClient(model=config.JUDGE_MODEL))
    result = await judge.judge(conv)
    s = result.scores

    print(f"File: {path}")
    print(f"correctness={s.correctness} naturalness={s.naturalness} "
          f"groundedness={s.groundedness} completeness={s.completeness}  mean={s.mean:.2f}")
    print(f"Judge says: {s.feedback}")
    print(f"Fixable by rewrite: {result.fixable_by_rewrite}")
    for i in result.issues:
        print(f"  - msg {i.get('message')}: {i.get('problem')}")

    passed = QualityPipeline(None, judge, None).passes(s)
    print(f"\nWould pass: {passed}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "output/sample_2.json"))

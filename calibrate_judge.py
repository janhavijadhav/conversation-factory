"""
Check the judge against conversations with known, injected flaws.

Usage:
    python calibrate_judge.py output/sample_3.json output/sample_4.json output/quality_2.json
"""
import asyncio
import json
import sys
from pathlib import Path

from src.factory import config
from src.factory.llm import LLMClient
from src.factory.quality.calibration import calibrate, load_conversations, summarize
from src.factory.quality.judge import LLMJudge
from src.factory.quality.repair import QualityPipeline
from src.factory.tools.registry import ToolRegistry


async def main(paths):
    bases = load_conversations(paths)
    if not bases:
        print("No usable conversations found in those files.")
        return
    print(f"Judge: {config.JUDGE_MODEL} | base conversations: {len(bases)}\n")

    judge = LLMJudge(ToolRegistry.from_directory(), LLMClient(model=config.JUDGE_MODEL))
    passes = QualityPipeline(None, judge, None).passes
    rows = await calibrate(judge, passes, bases)
    summary = summarize(rows)

    print(f"\n{'case type':<20}{'correct':>10}{'rate':>8}   lowest avg score")
    for kind, s in summary.items():
        lowest = min(s["avg_scores"], key=s["avg_scores"].get)
        label = "passed (good)" if kind == "control" else "caught"
        print(f"{kind:<20}{s['correct']:>4}/{s['cases']:<5}{s['rate']:>7.0%}   "
              f"{lowest}={s['avg_scores'][lowest]}   ({label})")

    corruptions = [r for r in rows if r["kind"] != "control"]
    controls = [r for r in rows if r["kind"] == "control"]
    caught = sum(r["correct_verdict"] for r in corruptions)
    false_alarms = sum(not r["passed"] for r in controls)
    print(f"\nOverall detection: {caught}/{len(corruptions)} corrupted cases caught")
    print(f"False alarms: {false_alarms}/{len(controls)} good conversations failed")

    out = Path("output") / "judge_calibration.json"
    out.write_text(json.dumps({"judge": config.JUDGE_MODEL, "summary": summary, "rows": rows}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["output/sample_3.json", "output/sample_4.json", "output/quality_2.json"]))

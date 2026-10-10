"""Show why conversations in a run failed: python inspect_run.py baseline"""
import json
import sys
from collections import Counter

name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
records = [json.loads(line) for line in open(f"output/{name}/rejected.jsonl") if line.strip()]
print(f"{len(records)} rejected conversations in '{name}'\n")

stages = Counter()
for r in records:
    actual = [tc["name"] for m in r["messages"] for tc in m.get("tool_calls", [])]
    print("=" * 70)
    print("Goal:   ", r["goal"][:160])
    print("Planned:", " -> ".join(r["planned_chain"]))
    print("Actual: ", " -> ".join(actual) or "(no tool calls)")
    print("Stop:   ", r.get("stop_reason"))
    for h in r["quality_history"]:
        stages[h["stage"]] += 1
        line = f"  attempt {h['attempt']} [{h['stage']}]"
        if "mean" in h:
            line += f" mean={h['mean']}"
        if h.get("action"):
            line += f" -> {h['action']}"
        print(line)
        for p in h.get("problems", []):
            print(f"      - {p}")
        for i in h.get("issues", []):
            print(f"      - msg {i.get('message')}: {i.get('problem')}")

print("\nWhere attempts failed:", dict(stages))

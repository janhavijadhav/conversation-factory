"""Compare two runs side by side: python compare_runs.py baseline steered"""
import json
import sys


def load(name):
    with open(f"output/{name}/metrics.json") as f:
        return json.load(f)


a_name, b_name = sys.argv[1], sys.argv[2]
a, b = load(a_name), load(b_name)

rows = [
    ("passed / generated", lambda m: f"{m['passed']}/{m['generated']}"),
    ("pass rate", lambda m: f"{m['pass_rate']:.0%}"),
    ("first-try pass rate", lambda m: f"{m['first_try_pass_rate']:.0%}"),
    ("multi-step ratio", lambda m: f"{m['corpus']['multi_step_ratio']:.0%}"),
    ("mean tool calls", lambda m: m["corpus"]["mean_tool_calls"]),
    ("long-chain ratio (3+)", lambda m: f"{m['corpus']['long_chain_ratio']:.0%}"),
    ("category balance", lambda m: m["corpus"]["category_balance"]),
    ("tool coverage", lambda m: f"{m['corpus']['tool_coverage']:.0%}"),
    ("repeated first names", lambda m: len(m["corpus"]["repeated_first_names"])),
    ("mean groundedness", lambda m: m["mean_scores"]["groundedness"]),
    ("mean naturalness", lambda m: m["mean_scores"]["naturalness"]),
    ("grounding fixes / conv", lambda m: m["grounding_fixes_per_conv"]),
    ("total LLM calls", lambda m: sum(m["llm_calls"].values())),
]

print(f"{'metric':<26}{a_name:>14}{b_name:>14}")
for label, fn in rows:
    print(f"{label:<26}{str(fn(a)):>14}{str(fn(b)):>14}")

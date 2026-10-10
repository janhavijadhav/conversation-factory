"""
Batch runner: generates many conversations with diversity steering and the
3-level quality system, then writes a JSONL corpus plus metrics.

output/<name>/train.jsonl     passed conversations (the training corpus)
output/<name>/rejected.jsonl  failed conversations (for analysis)
output/<name>/metrics.json    quality and diversity numbers for the run
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.factory import config
from src.factory.agents.orchestrator import ConversationGenerator
from src.factory.graph.sampler import ChainSampler, SampledChain
from src.factory.graph.tool_graph import ToolGraph
from src.factory.llm import DailyQuotaExceeded
from src.factory.models import Conversation
from src.factory.quality.diversity import DiversityTracker, SteeringPlan
from src.factory.quality.judge import DIMENSIONS, LLMJudge
from src.factory.quality.repair import QualityPipeline, Rewriter
from src.factory.tools.registry import ToolRegistry


class BatchRunner:
    def __init__(self, registry: ToolRegistry, gen_llm, judge_llm, out_dir: str | Path,
                 steer: bool = True, seed: int = 0, batch_size: int = config.BATCH_SIZE,
                 simulator_llm=None):
        self.registry = registry
        self.gen_llm = gen_llm
        self.judge_llm = judge_llm
        self.simulator_llm = simulator_llm
        self.out_dir = Path(out_dir)
        self.steer = steer
        self.seed = seed
        self.batch_size = batch_size

        self.sampler = ChainSampler(ToolGraph(registry), seed=seed)
        self.generator = ConversationGenerator(registry, gen_llm, simulator_llm=simulator_llm)
        self.pipeline = QualityPipeline(self.generator, LLMJudge(registry, judge_llm), Rewriter(gen_llm))
        self.tracker = DiversityTracker(registry)        # counts the accepted corpus only

        self.counts: Counter = Counter()
        self.score_sums: dict[str, float] = defaultdict(float)
        self.errors: list[dict[str, Any]] = []

    # ---- files -------------------------------------------------------------

    def _open_files(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self.out_dir / "train.jsonl"
        self.rejected_path = self.out_dir / "rejected.jsonl"
        for path in (self.train_path, self.rejected_path):
            path.write_text("")

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # ---- one conversation --------------------------------------------------

    def _sample(self, i: int, plan: SteeringPlan | None) -> SampledChain:
        chain = None
        if plan:
            chain = self.sampler.sample(min_len=plan.min_len, category=plan.category_for(i),
                                        boosts=plan.tool_boosts)
        return chain or self.sampler.sample()

    async def _one(self, index: int, i: int, plan: SteeringPlan | None,
                   semaphore: asyncio.Semaphore) -> Conversation:
        async with semaphore:
            chain = self._sample(i, plan)
            print(f"  #{index + 1} started: {' -> '.join(chain.tools)}", flush=True)
            start = time.time()
            conv = await self.pipeline.run(chain, seed=self.seed + index)
            verdict = "PASSED" if conv.passed else "failed"
            mean = f"{conv.scores.mean:.2f}" if conv.scores else "n/a"
            print(f"  #{index + 1} {verdict} in {time.time() - start:.0f}s "
                  f"(judge mean {mean}, {conv.num_tool_calls} tool calls, "
                  f"{conv.repair_attempts} repairs)", flush=True)
            return conv

    def _record(self, conv: Conversation) -> None:
        record = conv.to_training_record()
        record.update({
            "passed": conv.passed,
            "repair_attempts": conv.repair_attempts,
            "goal": conv.scenario.user_goal,
            "persona": conv.scenario.persona,
            "planned_chain": conv.scenario.tool_chain,
            "quality_history": conv.metadata.get("quality_history", []),
        })
        self._append(self.train_path if conv.passed else self.rejected_path, record)

        c = self.counts
        c["generated"] += 1
        c["grounding_fixes"] += len(conv.grounding_fixes)
        c["validation_errors"] += len(conv.validation_errors)
        for h in record["quality_history"]:
            if h.get("action") == "rewrite":
                c["rewrites"] += 1
            if h.get("action") == "regenerate":
                c["regenerations"] += 1
            if h.get("stage") == "structural":
                c["structural_failures"] += 1
        if conv.scores:
            c["judged"] += 1
            for d in DIMENSIONS:
                self.score_sums[d] += getattr(conv.scores, d)
        if conv.passed:
            c["passed"] += 1
            if conv.repair_attempts == 0:
                c["first_try"] += 1
            self.tracker.add(conv)

    # ---- the run -----------------------------------------------------------

    async def run(self, total: int, name: str = "run") -> dict[str, Any]:
        self._open_files()
        semaphore = asyncio.Semaphore(config.CONCURRENCY)
        start = time.time()
        index = 0
        stopped: str | None = None

        while index < total and not stopped:
            size = min(self.batch_size, total - index)
            plan = self.tracker.steering() if self.steer else None
            self.generator.planner.avoid_names = plan.avoid_names if plan else []

            tasks = [self._one(index + i, i, plan, semaphore) for i in range(size)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for k, result in enumerate(results):
                if isinstance(result, DailyQuotaExceeded):
                    stopped = str(result)
                elif isinstance(result, BaseException):
                    self.errors.append({"index": index + k,
                                        "error": f"{type(result).__name__}: {result}"[:300]})
                    print(f"  #{index + k + 1} error: {type(result).__name__}: {str(result)[:120]}")
                else:
                    self._record(result)

            index += size
            s = self.tracker.stats()
            print(f"[{name}] {index}/{total} | passed {self.counts['passed']}/{self.counts['generated']}"
                  f" | multi-step {s['multi_step_ratio']:.0%} | mean calls {s['mean_tool_calls']}")

        if stopped:
            print(f"[{name}] stopped early: {stopped}")
        metrics = self._metrics(name, total, time.time() - start, stopped)
        (self.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def _metrics(self, name: str, total: int, seconds: float, stopped: str | None) -> dict[str, Any]:
        c = self.counts
        generated = max(c["generated"], 1)
        judged = max(c["judged"], 1)
        llm_calls = {"assistant": getattr(self.gen_llm, "calls", 0),
                     "judge": getattr(self.judge_llm, "calls", 0)}
        if self.simulator_llm is not None:
            llm_calls["simulator"] = getattr(self.simulator_llm, "calls", 0)
        return {
            "run": name,
            "steering": self.steer,
            "requested": total,
            "generated": c["generated"],
            "passed": c["passed"],
            "pass_rate": round(c["passed"] / generated, 3),
            "first_try_pass_rate": round(c["first_try"] / generated, 3),
            "rewrites": c["rewrites"],
            "regenerations": c["regenerations"],
            "structural_failures": c["structural_failures"],
            "mean_scores": {d: round(self.score_sums[d] / judged, 2) for d in DIMENSIONS},
            "grounding_fixes_per_conv": round(c["grounding_fixes"] / generated, 2),
            "validation_errors_per_conv": round(c["validation_errors"] / generated, 2),
            "llm_calls": llm_calls,
            "minutes": round(seconds / 60, 1),
            "stopped_early": stopped,
            "errors": self.errors,
            "corpus": self.tracker.stats(),
        }

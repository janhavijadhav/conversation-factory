import asyncio
import json

from src.factory.llm import DailyQuotaExceeded
from src.factory.models import Conversation, Message, QualityScores, Scenario, ToolCall
from src.factory.runner import BatchRunner
from src.factory.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.from_directory()


def make_conv(passed: bool, n_calls: int = 2) -> Conversation:
    msgs = [Message(role="user", content="hi")]
    for k in range(n_calls):
        tc = ToolCall(id=f"c{k}", name="search_flights", arguments={})
        msgs += [Message(role="assistant", tool_calls=[tc]),
                 Message(role="tool", tool_call_id=f"c{k}", name="search_flights", content="{}")]
    msgs.append(Message(role="assistant", content="done"))
    conv = Conversation(
        scenario=Scenario(user_goal="g", tool_chain=["search_flights"], category="travel",
                          persona="Ines Duarte, 29"),
        messages=msgs,
        scores=QualityScores(correctness=5, naturalness=4, groundedness=5, completeness=5),
    )
    conv.passed = passed
    conv.metadata = {"quality_history": [{"attempt": 0, "stage": "judge", "mean": 4.75}]}
    return conv


class FakePipeline:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def run(self, chain, seed=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_runner(tmp_path, outcomes, batch_size=2):
    runner = BatchRunner(REGISTRY, gen_llm=None, judge_llm=None, out_dir=tmp_path,
                         steer=True, seed=0, batch_size=batch_size)
    runner.pipeline = FakePipeline(outcomes)
    return runner


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_writes_train_and_rejected(tmp_path):
    runner = make_runner(tmp_path, [make_conv(True), make_conv(False), make_conv(True, 3), make_conv(False)])
    m = asyncio.run(runner.run(4))
    assert len(lines(tmp_path / "train.jsonl")) == 2
    assert len(lines(tmp_path / "rejected.jsonl")) == 2
    assert m["pass_rate"] == 0.5
    assert m["corpus"]["conversations"] == 2
    assert m["corpus"]["mean_tool_calls"] == 2.5
    assert (tmp_path / "metrics.json").exists()


def test_one_error_does_not_stop_the_run(tmp_path):
    runner = make_runner(tmp_path, [make_conv(True), ValueError("bad json")])
    m = asyncio.run(runner.run(2))
    assert m["generated"] == 1
    assert len(m["errors"]) == 1


def test_stops_cleanly_on_daily_quota(tmp_path):
    runner = make_runner(tmp_path, [make_conv(True), make_conv(True),
                                    DailyQuotaExceeded("quota used"), make_conv(True)])
    m = asyncio.run(runner.run(6))
    assert m["generated"] == 3
    assert "quota used" in m["stopped_early"]
    assert (tmp_path / "metrics.json").exists()

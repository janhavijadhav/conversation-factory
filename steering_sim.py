"""Compare random sampling vs diversity-steered sampling. No LLM calls."""
from src.factory.graph.sampler import ChainSampler
from src.factory.graph.tool_graph import ToolGraph
from src.factory.quality.diversity import DiversityTracker
from src.factory.tools.registry import ToolRegistry

registry = ToolRegistry.from_directory()
graph = ToolGraph(registry)


def run(steer: bool, total: int = 400, batch: int = 10, seed: int = 0) -> dict:
    sampler = ChainSampler(graph, seed=seed)
    tracker = DiversityTracker(registry)
    for _ in range(total // batch):
        plan = tracker.steering() if steer else None
        for i in range(batch):
            chain = None
            if plan:
                chain = sampler.sample(min_len=plan.min_len, category=plan.category_for(i),
                                       boosts=plan.tool_boosts)
            chain = chain or sampler.sample()
            tracker.add_record(chain.main_category, chain.tools)
    return tracker.stats()


base, steered = run(steer=False), run(steer=True)
print(f"{'metric':<20}{'random':>12}{'steered':>12}")
for key in ["mean_tool_calls", "long_chain_ratio", "category_balance", "tool_coverage"]:
    print(f"{key:<20}{base[key]:>12}{steered[key]:>12}")
print("\ncategory share (random): ", base["category_share"])
print("category share (steered):", steered["category_share"])

from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.benchmark import Benchmark

if __name__ == '__main__':
    # RunAnalyzer('out', ['ObstacleEnv/mcts_320'])
    Benchmark.open_runs_summary('out/benchmark_mcts_iterations.json')



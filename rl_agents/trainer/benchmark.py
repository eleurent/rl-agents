import datetime
import json
import os

from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.simulation import Simulation


class Benchmark(object):
    SUMMARY_FILE = 'benchmark'

    def __init__(self, env, agents, num_episodes, train=False):
        self.env = env
        self.agents = agents
        self.num_episodes = num_episodes
        self.train = train

        self.summary_filename = os.path.join(Simulation.OUTPUT_FOLDER, '{}_{}.json'.format(
            self.SUMMARY_FILE, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

    def run(self):
        run_directories = []
        for agent in self.agents:
            run = Benchmark.evaluate_agent(self.env, agent, self.num_episodes, self.train)
            run_directories.append(run)

        self.write_runs_summary(run_directories)
        RunAnalyzer(run_directories)
        self.env.close()

    def write_runs_summary(self, run_directories):
        with open(self.summary_filename, 'w') as f:
            json.dump(run_directories, f, sort_keys=True, indent=4)

    @staticmethod
    def open_runs_summary(filename):
        with open(filename, 'r') as f:
            run_directories = json.loads(f.read())
        RunAnalyzer(run_directories)

    @staticmethod
    def evaluate_agent(env, agent, num_episodes, train=False):
        evaluation = Simulation(env, agent, num_episodes=num_episodes, close_env=False)
        run_directory = os.path.relpath(evaluation.monitor.directory)
        if train:
            evaluation.train()
        else:
            evaluation.test()
        return run_directory

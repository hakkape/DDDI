from ProblemData import ProblemData
from IntervalSolver import IntervalSolver, algorithm_option
from ExampleProblems import ExampleProblems
from gurobipy import Env
import signal
import sys

# Handle Ctrl-C to terminate cleanly
def _handle_sigint(signum, frame):
    print("\nCtrl-C received. Terminating...")
    sys.exit(130)

signal.signal(signal.SIGINT, _handle_sigint)

with Env() as env:
    for p in ExampleProblems.all_problems():
        print(f"Solving problem: {p[0]}")
        IntervalSolver(p[1], environment=env).solve()

    # solve a single instance with the IntervalSolver
    p = ProblemData.read_file(r'instances/timed_mtl_instances_1minute/c62_.3333_.25_1.txt')
    time_points = p.significant_time_points()

    problem = IntervalSolver(p, time_points=time_points, gap=0.01, algorithm=algorithm_option.reduced, environment=env)
    info = problem.solve()

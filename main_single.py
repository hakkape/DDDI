from ProblemData import ProblemData
from IntervalSolver import IntervalSolver
from ExampleProblems import ExampleProblems
from gurobipy import Env

with Env() as env:
    for p in ExampleProblems.all_problems():
        print(f"Solving problem: {p[0]}")
        IntervalSolver(p[1], environment=env).solve()

    # solve a single instance with the IntervalSolver
    p = ProblemData.read_file(r'instances/timed_mtl_instances_1minute/c62_.3333_.25_1.txt')
    problem = IntervalSolver(p, gap=0.01)
    info = problem.solve()


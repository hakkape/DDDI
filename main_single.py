from ProblemData import ProblemData
from IntervalSolver import IntervalSolver

# solve a single instance with the IntervalSolver
p = ProblemData.read_file(r'instances/timed_mtl_instances_1minute/c62_.3333_.25_1.txt')
problem = IntervalSolver(p, gap=0.01)
info = problem.solve()

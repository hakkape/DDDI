#!/usr/bin/env python3
# usage: python solve.py <instance_name> <output_path>
# where instance_name is the name of the instance file (without .txt extension)
# and output_path is the path to save the output as a json
from ProblemData import ProblemData
from IntervalSolver import IntervalSolver, algorithm_option
from gurobipy import Env
import json
import sys



if __name__ == "__main__":
    instance_name = sys.argv[1]  # get instance name from command line argument
    output_name = sys.argv[2]  # get output name from command line argument
    with Env() as env:
        # path of current file
        parent_path = __file__.rsplit('/', 1)[0] + '/'
        instances_folder = parent_path + 'instances/timed_mtl_instances_1minute/'
        instance_file = instances_folder + instance_name + '.txt'
        p = ProblemData.read_file(instance_file)

        problem = IntervalSolver(p, time_points=None, gap=0.01, algorithm=algorithm_option.reduced, environment=env)
        info = problem.solve()


        info_dict = {
            "n_iterations": len(info),
            "total_time": info[-1][2],
            "relaxation_solving_time": info[-1][3],
            "n_vars": info[-1][5],
            "n_constraints": info[-1][6],
            "lb": info[-1][0],
            "ub": info[-1][1],
        }
        with open(output_name, 'w') as f:
            json.dump(info_dict, f, indent=4)

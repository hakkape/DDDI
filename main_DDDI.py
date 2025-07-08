import csv
from os import listdir
from os.path import isdir, join, exists
from ProblemData import ProblemData
from IntervalSolver import IntervalSolver
from gurobipy import Env

def output_csv(path, file, instance, output, environment):
    csv_filename = output + file + ".csv"

    # create empty csv file, to stop other processes from trying to solve the same instance
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        pass

    try:
        print(csv_filename)
        p = ProblemData.read_file(path + file)
        problem = IntervalSolver(p, gap=0.0001, environment=environment)
        info = problem.solve()

        header = ['Instance','Id','LB','UB','total time','solve time','Added Time points','# Vars','# Cons','# Presolve Vars','# Presolve Cons','# Its','IP Gap']
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i in info:
                writer.writerow([file, instance] + list(i))
    
    except Exception as inst:
        print("Exception occurred:", type(inst))
        print(inst.args)


def run_all():
    path = r'instances/timed_mtl_instances_1minute/'
    instances = [f for f in listdir(path) if not isdir(join(path, f))]
    output = 'output/timed_mtl_instances_1minute/'

    # Create the output directory if it does not exist
    if not exists(output):
        from os import makedirs
        makedirs(output)

    env = Env("")

    for instance in instances:
        if not exists(output + instance + ".csv"):
            output_csv(path, instance, 1, output, env)


run_all()

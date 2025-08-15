import csv
from os import listdir
from os.path import isdir, join, exists
from ProblemData import ProblemData
from IntervalSolver import IntervalSolver
from gurobipy import Env
from instance_classification import InstanceClassification
from os import makedirs
from merge import merge_csv_files

def output_csv(path, file, instance, output, environment, output_last_iteration_only = True):
    csv_filename = output + file + ".csv"

    # create empty csv file, to stop other processes from trying to solve the same instance
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        pass

    try:
        print(csv_filename)
        p = ProblemData.read_file(path + file)
        problem = IntervalSolver(p, gap=0.01, environment=environment)
        info = problem.solve()
        
        if output_last_iteration_only:
            info = info[-1:]  # Get the last iteration only

        header = ["Instance", "Id", "LB", "UB", "total time", "solve time", "Added Time points", "# Vars", "# Cons", "# Presolve Vars", "# Presolve Cons", "# Its", "IP Gap"]

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i in info:
                writer.writerow([file, instance] + list(i))

    except Exception as inst:
        print("Exception occurred:", type(inst))
        print(inst.args)


def run_all(selected_instances, output_dir: str):
    path = r"instances/timed_mtl_instances_1minute/"
    instances = [f for f in listdir(path) if not isdir(join(path, f)) and f in selected_instances]
    output = f"output/{output_dir}/"

    # Create the output directory if it does not exist
    if not exists(output):
        makedirs(output)

    env = Env("")

    for instance in instances:
        if not exists(output + instance + ".csv"):
            output_csv(path, instance, 1, output, env)

    # merge output files for easier analysis
    merge_csv_files(output + output_dir + '.csv', *[output + instance + ".csv" for instance in instances])


run_all(InstanceClassification.LCLF, "LCLF")
run_all(InstanceClassification.LCHF, "LCHF")
run_all(InstanceClassification.HCLF, "HCLF")
run_all(InstanceClassification.HCHF, "HCHF")



import csv
from os import listdir
from os.path import isdir, join, exists
from ProblemData import ProblemData
from IntervalSolver import IntervalSolver
from gurobipy import Env
from os import makedirs
from merge import merge_csv_files

def output_csv(path, file, instance, output, environment):
    csv_filename = output + file + "_" + instance + ".csv"

    try:
        print(csv_filename)
        p = ProblemData.read_directory(path + file + "/" + instance)
        problem = IntervalSolver(p, fixed_paths=p.fixed_paths, gap=0.01, environment=environment)
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

def run_all(input_path, output_path, instance_type):
    env = Env("")

    path = f'{input_path}{instance_type}/'
    output = f'{output_path}{instance_type}/'

    # Create the output directory if it does not exist
    if not exists(output):
        makedirs(output)

    # List all instance directories in the path
    instances = [f for f in listdir(path) if isdir(join(path, f))]

    # Iterate over each instance
    for instance in instances:
        # List all subdirectories (ids) within each instance directory
        ids = [f for f in listdir(join(path, instance)) if isdir(join(path, instance, f))]

        # For each id, check if the output CSV already exists; if not, process and generate it
        for i in ids:
            if not exists(output + instance + "_" + i + ".csv"):
                output_csv(path, instance, i, output, env)

    # List all subdirectories (ids) within each instance directory
    files = [output + instance + "_" + id + ".csv" 
            for instance in listdir(path) if isdir(join(path, instance)) 
            for id in listdir(join(path, instance)) if isdir(join(path, instance, id))]

    merge_csv_files(output + instance_type + '.csv', *files)


if __name__ == "__main__":
    # Set the path to the instances and output directory based on the instance type
    # Assumes the SND-RR repo is in the parent directory
    input_path = '../SND-RR/Instances/'
    output_path = 'output/'

    run_all(input_path, output_path, 'critical_times')
    run_all(input_path, output_path, 'hub_and_spoke')
    run_all(input_path, output_path, 'designated_paths')

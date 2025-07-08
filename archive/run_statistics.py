# Solve an integer multicommodity fixed charge network flow problem
from BuildNetwork import *
from ProblemData import *
from DrawLaTeX import *
from ExampleProblems import *
from CheckSolution import *
from ConsolidationSolver import *
from IntervalSolver import *
import time
import csv

def get_statistics(stats, model, time, gurobi_time, iterations=0):
    return {'Avg Iv': round(stats[0]['avg_points'], 2),
            'Min Iv': stats[0]['min_points'],
            'Max Iv': stats[0]['max_points'],
            'Intervals': stats[0]['intervals'],
            'Arcs': stats[0]['arcs'],
            'Vars': len(model.getVars()),
            'LB': stats[0]['cost'],
            'IP LB': stats[0]['lower_bound'],
 #           'Gap \\%': round((100*(stats[1]['solution_cost'] - stats[0]['lower_bound']) / stats[1]['solution_cost']), 3),
            '\\# Its': iterations,
#            'LP UB': stats[1]['solution_cost'],
            'Time': round(time, 2),
            'Solve Time': round(gurobi_time, 2),
            'raw': stats
            }


##
## Gathers statistics for a given problem and discretization
##
def run_instance(p, time_points):
    t1 = time.clock()

    solver = ConsolidationSolver(p, time_points)
    solver.solve()

    t2 = time.clock()

    if solver.model.status != GRB.status.OPTIMAL:
        return None

    s = CheckSolution(solver)
    s.validate(solver.get_solution_paths(), solver.get_consolidations())

    return get_statistics((solver.get_statistics(), s.get_statistics()), solver.model, t2-t1)

def solve_instance(p):
    t1 = time.clock()
    solver = ConsolidationSolver(p)
    iterations, new_timepoint_count, gurobi_time = solver.solve()
    t2 = time.clock()

    s = CheckSolution(solver)
    s.validate(solver.get_solution_paths(), solver.get_consolidations())

    return (solver, get_statistics((solver.get_statistics(), s.get_statistics()), solver.model, t2-t1, iterations))


##
## Gathers statistics (different discretization) for a given problem
##
def run_statistics(p, custom_timepoints={}):
    n = BuildNetwork(p)

    results = {
                '1,D1': run_instance(p, n.discretization_network(1)), 
                '2,D2': run_instance(p, n.discretization_network(2)), 
                '3,D5': run_instance(p, n.discretization_network(5)), 
                '4,trivial': run_instance(p, n.trivial_network()), 
                '5,simple': run_instance(p, n.simple_network()), 
                '6,shortest': run_instance(p, n.shortest_path_network()), 
                '7,2nd shortest': run_instance(p, n.second_shortest_path_network()), 
                '8,all shortest': run_instance(p, n.all_shortest_path_network()), 
                '9,all 2nd shortest': run_instance(p, n.all_second_shortest_path_network())
              }

    results = dict(results.items() + custom_timepoints.items())

    output = []
    header = 'Discretization,Avg Iv,Min Iv,Max Iv,Intervals,Arcs,Vars,IP LB,LP UB,Gap \\%,\\# Its,Time'.split(',')
    data = []

    print('\n')
    print(','.join(header))
    
    for d in sorted(results):
        if results[d] == None:
            continue

        results[d]['Discretization'] = d.split(',')[1]
        data.append(results[d])

        print(','.join([str(results[d][h]) for h in header]))

    return (header, data, results)


##
## Creates pdf for problem statistics
##
def output_statistics(problem, name, scale=1, custom_timepoints={}, font='normalsize'):
    solver,trivial = solve_instance(problem)
    header,data,raw = run_statistics(problem, dict({'9.1,iterative': trivial}.items() + custom_timepoints.items()))

    print("\n\nSaving to LaTeX...\n")

    # draw model via pdf latex
    pdf = DrawLaTeX(problem.commodities, solver.network)
    pdf.draw_network(scale, position=problem.position, font=font)
    pdf.draw_commodities(scale, position=problem.position, font=font)

    full = raw['1,D1']
    compare =  {'N': len(solver.network.nodes()),
                'A': len(solver.network.edges()),
                'C': len(problem.commodities),
                'Horizon': str([min([k['a'][1] for k in problem.commodities]), max([k['b'][1] for k in problem.commodities])]),
                'Intervals': "{0}/{1} ({2}\\%)".format(trivial['Intervals'], full['Intervals'], round(trivial['Intervals'] * 100.0 / full['Intervals'], 2)),
                'Arcs': "{0}/{1} ({2}\\%)".format(trivial['Arcs'], full['Arcs'], round(trivial['Arcs'] * 100.0 / full['Arcs'], 2)),
                'Vars': "{0}/{1} ({2}\\%)".format(trivial['Vars'], full['Vars'], round(trivial['Vars'] * 100.0 / full['Vars'], 2)),
                'Time': "{0}/{1}".format(trivial['Time'], full['Time']),
                'Iterations': trivial['\\# Its'],
                'Gap': "{0}/{1}".format(trivial['Gap \\%'], full['Gap \\%']),
                'Value': "{0}/{1}".format(trivial['LP UB'], full['LP UB'])}

    pdf.draw_table('N,A,C,Horizon,Intervals,Arcs,Vars,Time,Iterations,Gap,Value'.split(','), [compare])
    pdf.draw_table([], []) # add vertical gap
    
    pdf.draw_table(header, data)

    pdf.draw_latex(False, 2)
    pdf.save(name)


##
## Compares a list of problems: full discretization vs our solution
##
def output_comparison(problems):
    # draw model via pdf latex
    pdf = DrawLaTeX(None, None)

    header = 'Instance,N,A,C,Horizon,Intervals,Arcs,Vars,Time,Iterations,Gap,Value'
    data = {}

    for name, p in problems.items():
        solver, trivial = solve_instance(p)
        full = run_instance(p, BuildNetwork(p).discretization_network(1))

        data[name] = {'Instance': name, 
                      'N': len(solver.network.nodes()),
                      'A': len(solver.network.edges()),
                      'C': len(p.commodities),
                      'Horizon': str([min([k['a'][1] for k in p.commodities]), max([k['b'][1] for k in p.commodities])]),
                      'Intervals': "{0}/{1} ({2}\\%)".format(trivial['Intervals'], full['Intervals'], round(trivial['Intervals'] * 100.0 / full['Intervals'], 2)),
                      'Arcs': "{0}/{1} ({2}\\%)".format(trivial['Arcs'], full['Arcs'], round(trivial['Arcs'] * 100.0 / full['Arcs'], 2)),
                      'Vars': "{0}/{1} ({2}\\%)".format(trivial['Vars'], full['Vars'], round(trivial['Vars'] * 100.0 / full['Vars'], 2)),
                      'Time': "{0}/{1}".format(trivial['Time'], full['Time']),
                      'Iterations': trivial['\\# Its'],
                      'Gap': "{0}/{1}".format(trivial['Gap \\%'], full['Gap \\%']),
                      'Value': "{0}/{1}".format(trivial['LP UB'], full['LP UB'])}

    pdf.draw_table(header.split(','), [data[d] for d in sorted(data)])
    pdf.draw_latex(False, 2)
    pdf.save('compare')



def output_csv(file, scheme):
    csv_filename = file + scheme[20:] + ".csv"
    header = ['name','rounding','Vars','Constraints','Presolve Vars','Presolve Constrs','Arcs','Intervals','Cost','IP LB','IP Gap','Solve Time','Time','Status','\\# Its']

    path = r'C:/data/Time-Expanded Networks Computations/gather_stats/tests/'
#    path = "../gather_stats/tests/"

    try:
        t1 = time.clock()
        problem = IntervalSolver(ProblemData.read_file(path + scheme + "/" + file))
        t = time.clock()-t1

        d = get_statistics([problem.get_statistics()], problem.model, t, 0, 0)
        d['Presolve Vars'] = d['raw'][0]['presolve_vars']
        d['Presolve Constrs'] = d['raw'][0]['presolve_cons']
        d['Constraints'] = len(problem.model.getConstrs())
        d['rounding'] = scheme[20:]
        d['name'] = file
        d['Cost'] = ''
        d['IP Gap'] = ''
        d['Status'] = 'Aborted'

        with open('output/time/' + csv_filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(map(lambda h: d[h], header))

        t1 = time.clock()
        iterations, new_timepoint_count, gurobi_time = problem.solve()
        t2 = time.clock()

        if problem.infeasible() == False:
            d.update(get_statistics([problem.get_statistics()], problem.model, t2-t1+t, gurobi_time, iterations))

            if (problem.model.status == GRB.status.OPTIMAL or problem.model.status == GRB.status.TIME_LIMIT):
                d['Cost'] = d['LB']
                d['IP Gap'] = 1-d['IP LB']/d['Cost']
                d['Status'] = 'Solved' if problem.model.status != GRB.status.TIME_LIMIT else "Time Limit"
                d['Presolve Vars'] = d['raw'][0]['presolve_vars']
                d['Presolve Constrs'] = d['raw'][0]['presolve_cons']
                d['Constraints'] = len(problem.model.getConstrs())
            else:
                d['Status'] = 'Infeasible'
        else:
            d['Status'] = 'Infeasible'

        with open('output/time/' + csv_filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(map(lambda h: d[h], header))
    
    except Exception as inst:
        print("Exception occurred:", type(inst))
        print(inst.args)

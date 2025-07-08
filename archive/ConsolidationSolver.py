from gurobipy import *
import itertools
from CheckSolution import *
from math import log10, ceil
from collections import Counter
from DrawLaTeX import *
from tools import * 
#import matplotlib.pyplot as plt

#from BuildNetwork import *
import time

def enum(**enums):
    return type('Enum', (), enums)

preprocessing_option = enum(node='node', arc='arc')
shortest_path_option = enum(commodity='commodity', shared='shared')

try:
    import pydot
    pydot_loaded = True
except ImportError:
    pydot_loaded = False

import networkx as nx

# zips a sequence on itself - "s -> (s0,s1), (s1,s2), (s2, s3), ..."
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

# itertools.accumulate is in python 3.x
def accumulate(iterator):
    total = 0
    for item in iterator:
        total += item
        yield total

# zips a sequence on itself - "s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ..."
def triple(iterable):
    a, b, c = itertools.tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return itertools.izip(a, b, c)

def quad(iterable):
    a, b, c, d = itertools.tee(iterable, 4)
    next(b, None)
    next(c, None)
    next(c, None)
    next(d, None)
    next(d, None)
    next(d, None)
    return itertools.izip(a, b, c, d)


#def pairwise(t):
#    it = iter(itertools.chain.from_iterable((x,x) for x in t))
#    next(it)
#    return itertools.izip(it,it)


MIP_GAP = 0.01

class ConsolidationSolver(object):
    """description of class"""
    __slots__ = ['problem', 'model', 'network', 'commodity_shortest_paths', 'shared_shortest_paths', 'S', 'T', 'x', 'z', 'commodities', 'intervals', 'arcs', 'origin_destination', 'constraint_consolidation', 'constraint_flow', 'constraint_cycle', 'constraint_path_length', 'solution_paths','consolidations','consolidation_groups', 'fixed_timepoints_model','redirected_arcs', 'initial_timepoints', 'incumbent', 'lower_bound', 'plot']

    def __init__(self, problem, time_points=None, full_solve=True, plot=False):
        self.problem = problem
        self.commodities = problem.commodities
        self.S = min(c['a'][1] for c in self.commodities)  # time horizon
        self.T = max(c['b'][1] for c in self.commodities) + 1  # time horizon

        self.incumbent = None  # store the lowest upper bound
        self.lower_bound = None
        self.solution_paths = None
        shouldEnforceCycles = False

        # build graph
        self.network = nx.DiGraph()

        for a, destinations in problem.network.items():
            for b, transit_time in destinations.items():
                shouldEnforceCycles = shouldEnforceCycles or problem.var_cost.get((a,b),0) == 0  # if all arcs have positive costs then we don't need to add extra constraints
                self.network.add_edge(a, b, weight=transit_time, capacity=problem.capacities.get((a,b), 1.0), fixed_cost=problem.fixed_cost.get((a,b), transit_time), var_cost=problem.var_cost.get((a,b), 0))

        self.create_shortest_paths()

        #print self.estimate_cost()

        ## interactive graphing! :)
        self.plot = None

        #if plot:
        #    self.plot = Abj(LB=[], it=[], UB=[], fig=plt.figure(), ax = None, lb_line = None, ub_line = None)
        #    self.plot.ax = self.plot.fig.add_subplot(111)
        #    plt.ion()
        #    plt.tight_layout()

        # create initial intervals/arcs
        self.redirected_arcs = {}
        self.intervals, self.arcs = self.create(self.trivial_network() if time_points == None else time_points)
        self.initial_timepoints = time_points
        self.fixed_timepoints_model = time_points != None and full_solve == False
        
        # find which interval contains the origin/destination
        self.origin_destination = {k: (self.find_interval(c['a']), self.find_interval(c['b'])) for k,c in enumerate(self.commodities)}


        ##
        ## Construct Gurobi model
        ## 
        model = self.model = Model("IMCFCNF", env=Env(""))
        model.modelSense = GRB.MINIMIZE

        if self.fixed_timepoints_model == False:
            model.setParam('OutputFlag', False)

    #    model.setParam(GRB.param.MIPGap, 0.01)  # 1% gap
        model.setParam(GRB.param.TimeLimit, 14400) # 4hr limit
        #model.setParam(GRB.param.MIPFocus, 2) # focus on proving optimality

        # only dispatch arcs (we don't care about storage arcs)
        K = len(self.commodities)
        all_arcs = sorted(set(a for k in range(K) for a in self.arcs[k] if a[0][0] != a[1][0]))
        
        ##
        ## decision variables
        ##

        # x - dispatch along an arc [k,a]
        x = self.x = {(k, a): model.addVar(obj=(self.network[a[0][0]][a[1][0]]['var_cost'] * self.commodities[k]['q'] if a[0][0] != a[1][0] else 0), lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(k) + ',' + str(a)) 
                        for k in range(K) for a in self.arcs[k]}

        # z - number of trailers required for arc [a]
        z = self.z = {(a,b): model.addVar(obj=(self.network[a[0]][b[0]]['fixed_cost']), lb=0, ub=GRB.INFINITY, name='z' + str((a,b)), vtype=GRB.INTEGER)
                        for a,b in all_arcs}
        
        model.update()

        ##
        ## Constraints
        ## 

        # flow constraints
        self.constraint_flow = {}

        for k in range(K):
            for n in self.intervals:
                i,o = self.inflow(k,n), self.outflow(k,n)

                if i or o:
                    self.constraint_flow[(k,n)] = model.addConstr(quicksum(i) - quicksum(o) == self.r(k,n), 'flow' + str((k,n)))

        # Consolidation constraints
        self.constraint_consolidation = {a: model.addConstr(quicksum(self.x[k,a] * self.commodities[k]['q'] for k in range(K) if (k,a) in self.x) <= self.z[a] * self.network[a[0][0]][a[1][0]]['capacity'], 'cons' + str(a)) for a in all_arcs }

        # Ensure no cycles
        self.constraint_cycle = None

        if shouldEnforceCycles:
            self.constraint_cycle = {}
    
            for k in range(K):
                for n in self.network.nodes():
                    outflow = [x[k,a] for a in self.arcs[k] if a[0][0] == n and a[1][0] != n]

                    if outflow:
                        self.constraint_cycle[(k,n)] = model.addConstr(quicksum(outflow) <= 1, 'cycle')

        # Ensure path length
        #self.constraint_path_length = {}

        #for k in range(K):
        #    path_length = self.commodities[k]['a'][1]

        #    for n in self.network.nodes():
        #        outflow = [x[k,a]*self.transit(a[0][0],a[1][0]) for a in self.arcs[k] if a[0][0] == n and a[1][0] != n]

        #        if outflow:
        #            path_length += quicksum(outflow)

        #    self.constraint_path_length[k] = model.addConstr(path_length <= self.commodities[k]['b'][1], 'path')
            

    # find a node/interval from a node/time
    def find_interval(self, node_time):
        node,time = node_time
        return next((n for n in self.intervals.select(node, '*', '*') if time >= n[1] and time < n[2]), None)

    def inflow(self,k,i):
        return [self.x[k,a] for a in self.arcs[k].select('*', i)]

    def outflow(self,k,i):
        return [self.x[k,a] for a in self.arcs[k].select(i, '*')]

    def r(self,k,i):
        # at origin
        if self.origin_destination[k][0] == i:
            return -1

        # at destination
        if self.origin_destination[k][1] == i:
            return 1

        return 0

    def infeasible(self):
        return len([c for k,c in enumerate(self.commodities) if c['a'][1] + self.shortest_path(k,c['a'][0],c['b'][0]) > c['b'][1]]) > 0

    ##
    ## Iterative Solve
    ##
    def solve(self, draw_filename='', start_time=time.time()):
        # feasibility check: shortest path cannot reach destination - gurobi doesn't pick it up, because no arcs exist
        if self.infeasible():
            print("INFEASIBLE")
            return 0, 0, 0

        gap_test = True
        iterations = -1
        new_timepoints = set()  # not required, but might be good for testing
        len_timepoints = 0      # used to halt processing if haven't added unique timepoints in a while (hack - this shouldn't happen if my code was good)
        it_timepoints = 0

        s = CheckSolution(self)

        solve_time = 0

        # output statistics
        sys.stdout.write('{0:>3}, {1:>10}, {2:>10}, {3:>7}, {4:>6}, {5:>6}, {6}'.format(*'##,LB,UB,Gap,Time,Solver,Type [TP]'.split(',')))

        ### TEST
        #n = BuildNetwork(self.problem)
        #intervals, arcs = n.create(n.trivial_network())

        #missing = {k: set(v).symmetric_difference(set(arcs[k])) for k,v in self.arcs.items()}

        #p = NetworkFlow(self.commodities, self.network, intervals, arcs)
        #p.solve()

        while True:
            iterations += 1


            ##
            ## Testing - model is different than rebuilding every time!
            ##
            #tp = self.trivial_network() if self.initial_timepoints == None else tuplelist(self.initial_timepoints)
            #tp.extend(new_timepoints)
            #tp = tuplelist(set(tp))

            #tmp = self.redirected_arcs
            #self.redirected_arcs = {}
            #intervals, arcs = self.create(tp)
            #self.redirected_arcs = tmp

            ## which arcs are missing
            #removed_arcs = {k: [a for a in v if a not in self.arcs[k]] for k,v in arcs.items()}
            #count1 = sum(len(i) for i in removed_arcs.values())

            ## which arcs are redundant
            #added_arcs = {k: [a for a in v if a not in arcs[k]] for k,v in self.arcs.items()}
            #count2 = sum(len(i) for i in added_arcs.values())

            #print (count1,count2)

            t0 = time.time()
            self.solve_lower_bound()  # Solve lower bound problem
            solve_time += time.time() - t0

            # return if not optimal
            if self.model.status != GRB.status.OPTIMAL:
                print("INFEASIBLE")

                #for c,v in sorted(self.arcs.items()):
                #    for i1,i2 in sorted(v):
                #        print c,i1,i2


                #print new_timepoints
                return iterations, len(new_timepoints), solve_time

            #for c,v in sorted(self.arcs.items()):
            #    for i1,i2 in sorted(v):
            #        print c,i1,i2

            last = self.solution_paths
            self.solution_paths = self.get_solution_paths()
            self.consolidations = self.get_consolidations()
            self.consolidation_groups = self.get_consolidation_groups()
            #self.consolidation_groups = map(frozenset, self.get_partitions())

            ### TESTING path changes - remove
            if last != None:
                path_changes += sum([1 for k in range(len(self.solution_paths)) if last[k] != self.solution_paths[k]])
            else:
                path_changes = 0

            ### interactive graphing! :)
            #if self.plot != None:
            #    self.plot.LB.append(self.lower_bound)
            #    self.plot.it.append(iterations)

            #    if iterations == 0:
            #        self.plot.lb_line, = self.plot.ax.plot(self.plot.LB)
            #        self.plot.ub_line, = self.plot.ax.plot(self.plot.UB)

            #    self.plot.lb_line.set_ydata(self.plot.LB)
            #    self.plot.lb_line.set_xdata(self.plot.it)
 
            #    self.plot.ax.relim()
            #    self.plot.ax.autoscale()

            #    if iterations == 0:
            #        plt.show()    

            #    plt.draw()
            #    plt.pause(0.001)


            # output statistics
            sys.stdout.write("{0:>3}, {1:10.1f}, ".format(iterations, self.lower_bound))

            # return if we are only doing one iteration
            if self.fixed_timepoints_model == True:
                return iterations, len(new_timepoints), solve_time

            # draw the timepoints if required
            portrait = False

            if draw_filename != '':
                scale = 2
                pdf = DrawLaTeX(self.commodities, self.network)
                pdf.draw_network(scale, position=self.problem.position, font='normal')
#                pdf.draw_commodities(scale, position=self.problem.position)
 
                data = [{'k': k, 
                         'Node': n[0],
                         'R1': self.shortest_path(k, c['a'][0], n[0]) < sum([self.transit(n1,n2) for n1,n2 in pairwise(self.solution_paths[k][:i+1])]),  
                         'R4': sum([self.transit(n1,n2) for n1,n2 in pairwise(self.solution_paths[k][:i+1])]) + self.shortest_path(k, n[1], c['b'][0]) <= c['b'][1]-c['a'][1],  
                         'TW': self.time_window(k, n), 
                         'SP': (c['a'][1] + self.shortest_path(k,c['a'][0], n[0]), c['b'][1] - self.shortest_path(k, n[0],c['b'][0])), 
                         r'$e^k+\sum_{i=1}^{n-1} \tau^k_{\left(n_i,n_{i+1}\right)}$': c['a'][1] + sum([self.transit(n1,n2) for n1,n2 in pairwise(self.solution_paths[k][:i+1])])}
                          for k,c in enumerate(self.commodities) 
                            for i,n in enumerate(pairwise(self.solution_paths[k]))]
                pdf.draw_table(r"k;Node;R1;R4;TW;SP;$e^k+\sum_{i=1}^{n-1} \tau^k_{\left(n_i,n_{i+1}\right)}$".split(';'), data)
 
                pdf.draw_table(r"k;$o^k\rightarrow d^k$;$[e^k, l^k]$".split(';'), 
                               [{'k': k, r'$o^k\rightarrow d^k$': (c['a'][0], c['b'][0]), '$[e^k, l^k]$': (c['a'][1], c['b'][1])} for k,c in enumerate(self.commodities)])              

                pdf.draw_timeline(self.intervals, self.arcs, 24 if not portrait else 15, self.get_solution())
                pdf.draw_latex(portrait)
                pdf.save(draw_filename + str(iterations))


            # 1. Path length is greater than time window  (choose latest node before destination and actual time it would dispatch - iterates if already in time points)
            path_length_timepoints = self.check_path_length()
            if path_length_timepoints:
                new_timepoints.update(path_length_timepoints)

                # output statistics
                sys.stdout.write("{0:>10}, {0:>7}, {1:6.2f}, {2:6.2f}, P[{3}]".format('-', time.time()-start_time, solve_time, len(path_length_timepoints)))

                ## interactive plot!
                #if self.plot != None:
                #    self.plot.UB.append(None)

                #print("Path length violated")
                continue

            # 2. Solve UB LP, check solution costs
            t0 = time.time()
            if s.validate(self.solution_paths, self.consolidations):
                solve_time += time.time() - t0

                stats = s.get_statistics()
                self.incumbent = min(self.incumbent, stats['solution_cost']) if self.incumbent != None else stats['solution_cost']


                ## interactive plot!
                #if self.plot != None:
                #    self.plot.UB.append(self.incumbent)
                #    self.plot.ub_line.set_ydata(self.plot.UB)
                #    self.plot.ub_line.set_xdata(self.plot.it)
 
                #    self.plot.ax.relim()
                #    self.plot.ax.autoscale()

                #    if iterations == 0:
                #        plt.show()    

                #    plt.draw()
                #    plt.pause(0.001)

                #test = s.get_broken_consolidations()

                ## Stop endless loop - hack for my bad code
                if len(new_timepoints) == len_timepoints:
                    it_timepoints += 1
                else:
                    len_timepoints = len(new_timepoints)
                    it_timepoints = 0

                # arbitrary number, I used # sort orders * 2
                if it_timepoints > 12:
                    sys.stdout.write(" Endless Loop\n")
                    break

                if gap_test == True and (self.incumbent - self.lower_bound) < self.incumbent*0.04:
                    self.model.setParam(GRB.param.MIPGap, MIP_GAP*0.9)  # slightly less than mip gap so to achieve desired gap
                    self.model.setParam(GRB.param.MIPFocus, 2) # focus on proving optimality
                    gap_test = False
                elif gap_test == True:
                    self.model.setParam(GRB.param.MIPGap, ((self.incumbent - self.lower_bound)/self.incumbent * .6))  # 1% gap


                #if gap_test == True and (self.incumbent - self.lower_bound) < self.incumbent*0.05:
                #    self.model.setParam(GRB.param.MIPGap, 0.01)  # 1% gap
                #    self.model.setParam(GRB.param.MIPFocus, 2) # focus on proving optimality
                #elif gap_test == True and (self.incumbent - self.lower_bound) < self.incumbent*0.15:
                #    self.model.setParam(GRB.param.MIPGap, 0.05)  # 1% gap
                #    gap_test = False
                


                #n = BuildNetwork(self.problem)
                #its,arcs = n.create_arcs(n.discretization_network(1), self.solution_paths)
                #nf = NetworkFlow(self.commodities, self.network, its, arcs)
                #nf.solve()

                sys.stdout.write("{0:10.1f}, {1:7.2%}, {2:6.2f}, {3:6.2f},".format(self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time))

                #print("UB: {0} - Gap: {1}%".format(self.incumbent, round((self.incumbent - self.lower_bound)*100/self.incumbent, 2)))

                # Checks gap of a lower bound / upper bound solutions - 1% gap               
                if (self.incumbent - self.lower_bound) < self.incumbent*MIP_GAP:
                    sys.stdout.write(" Optimal Solution\n")

                    #print new_timepoints
                    #print ""
                    #for k,p in enumerate(s.solution_paths):
                    #    print k, p
                    break

                ## 3. Consolidation is not possible due to direct time window (check sorted time windows for all commodities, choose arc[0] node and earliest time of the first/latest commodity that does not overlap)
                #consolidation_window_timepoints = self.check_time_windows()
                #if consolidation_window_timepoints:
                #    #new_timepoints.update(consolidation_window_timepoints)
                #    self.update_timepoints(new_timepoints, consolidation_window_timepoints)
                #    sys.stdout.write("Time window, {0}\n".format(len(consolidation_window_timepoints)))
                #    #print("Time window violated")
                #    continue


                ## Process timewindows FIRST!!!!!
                b = s.get_broken_consolidations()

                if b:
                    solution_times = s.get_solution_times() # store before destroying next step
                    broken = set((a,frozenset((k,solution_times[k][self.solution_paths[k].index(a[0][0])][1]) for k in itertools.chain.from_iterable(v))) for a,v in b.items())

                    # process early broken consolidations first, as they may change future consolidations
                    broken_consolidations_ordered_by_time = list(broken)
                    
                    ##
                    ## Try various sorting options to choose 'best' order to 'fix' broken consolidations

                    # earliest time
                    if iterations % 6 == 3:
                        broken_consolidations_ordered_by_time.sort(key=lambda (a,v): max(t for k,t in v), reverse=False)

                    # amount of time broken
                    if iterations % 6 == 1:
                        broken_consolidations_ordered_by_time.sort(key=lambda (a,v): sum((abs(n1[1]-n2[1]) for n1,n2 in pairwise(v))), reverse=True)
                    
                    # estimated cost
                    if iterations % 6 == 2:
                        broken_consolidations_ordered_by_time.sort(key=lambda (a,v): sum(self.commodities[k]['q'] for k,t in v)*self.network[a[0][0]][a[1][0]]['var_cost'] + ceil(sum(self.commodities[k]['q'] for k,t in v) / self.network[a[0][0]][a[1][0]]['capacity']) * self.network[a[0][0]][a[1][0]]['fixed_cost'], reverse=True)
                    
                    # Normalized early time & estimated cost
                    if iterations % 6 == 5:
                        broken_consolidations_ordered_by_time.sort(key=lambda (a,v): 1 - (max(t for k,t in v) - self.S)/float(self.T - self.S) + sum(self.commodities[k]['q'] for k,t in v)*self.network[a[0][0]][a[1][0]]['var_cost'] + ceil(sum(self.commodities[k]['q'] for k,t in v) / self.network[a[0][0]][a[1][0]]['capacity']) * self.network[a[0][0]][a[1][0]]['fixed_cost'], reverse=True)
                    
                    # Earliest time & estimated cost
                    if iterations % 6 == 4:
                        broken_consolidations_ordered_by_time.sort(key=lambda (a,v): self.T - 2*self.S - max(t for k,t in v) + sum(self.commodities[k]['q'] for k,t in v)*self.network[a[0][0]][a[1][0]]['var_cost'] + ceil(sum(self.commodities[k]['q'] for k,t in v) / self.network[a[0][0]][a[1][0]]['capacity']) * self.network[a[0][0]][a[1][0]]['fixed_cost'], reverse=True)
                    
                    # largest group first
                    if iterations % 6 == 0:
                        broken_consolidations_ordered_by_time.sort(key=lambda (a,v): len(v), reverse=True)




                    # process all irreducible commodities separately
                    processed = set()
                    arc = broken_consolidations_ordered_by_time[0] if broken_consolidations_ordered_by_time else None

                    # get the next arc that has a group not already processed
                    broken_arcs = itertools.ifilter(lambda (a,g): g not in processed, 
                                                    ((a,next((g for g in self.consolidation_groups if next(iter(a[1]))[0] in g), None)) for a in broken_consolidations_ordered_by_time)) 

                    ##
                    ## Time windows
                    while arc != None and len(processed) < len(self.consolidation_groups):
                        arc_group = next(broken_arcs, None)

                        if arc_group == None:
                            break

                        arc,group = arc_group

                        # keep removing consolidations until infeasible to discover problem arcs
                        if s.test_fix_and_resolve(arc[0]) != 3:
                            continue

                        processed.add(group)
                        tmp = self.testing_timewindows(arc[0], b[arc[0]], solution_times)
                        new_timepoints.update(tmp)
                        sys.stdout.write(" W[{0}]".format(len(tmp)))

                    ##
                    ## Mutual consolidations
                    
                    # must reset the model after playing with it above
                    s.model.update()
                    s.model.optimize()

                    arc = broken_consolidations_ordered_by_time[0] if broken_consolidations_ordered_by_time else None

                    # get the next arc that has a group not already processed
                    broken_arcs = itertools.ifilter(lambda (a,g): g not in processed, 
                                                    ((a,next((g for g in self.consolidation_groups if next(iter(a[1]))[0] in g), None)) for a in broken_consolidations_ordered_by_time)) 

                    while arc != None and len(processed) < len(self.consolidation_groups):
                        arc_group = next(broken_arcs, None)

                        if arc_group == None:
                            break

                        arc,group = arc_group
                        processed.add(group)
                        arc_times = [arc]

                        # keep removing consolidations until infeasible to discover problem arcs
                        while arc != None:
                            if s.test_remove_and_resolve(arc[0]) == 3:
                                break

                            # mutual exclusive!?
                            solution_times = s.get_solution_times()
                            b2 = s.get_broken_consolidations()
                            broken2 = set((a,frozenset((k,solution_times[k][self.solution_paths[k].index(a[0][0])][1]) for k in itertools.chain.from_iterable(v))) for a,v in b2.items())

                            count = len(broken2.difference(broken))
                            arc_times.extend(broken2.difference(broken))

                            if count == 0:
                                arc = None

                            b = b2

                            #remove all!
                            for a in broken2.difference(broken):
                                if s.test_remove_and_resolve(a[0]) == 3:
                                    arc = a
                                    break


                        if len(arc_times) == 1:
                            tmp = self.testing_timewindows(arc_times[0][0], b[arc_times[0][0]], solution_times)
                            new_timepoints.update(tmp)
                            sys.stdout.write(" T2[{0}]".format(len(tmp)))
                            #continue
                        else:
                            # choose first and last arcs
                            arc_times.sort(key=lambda x: max(t for k,t in x[1]))
                            first_arc = arc_times[0]
                            last_arc = arc_times[-1]

                                ## check timewindows also
                                #tmp = self.testing_timewindows(first_arc[0], [(k1,k2) for k1,t1 in first_arc[1] for k2,t2 in first_arc[1] if k1<k2], solution_times)

                                #if tmp:
                                #    continue

                                #tmp = self.testing_timewindows(last_arc[0], [(k1,k2) for k1,t1 in last_arc[1] for k2,t2 in last_arc[1] if k1<k2], solution_times)

                                #if tmp:
                                #    continue

                            cons = {key: list(itertools.imap(operator.itemgetter(1), group)) for key, group in itertools.groupby(sorted(self.consolidations.iteritems(), key=lambda a: (a[0][0][0],a[0][1][0])), key=lambda a: (a[0][0][0],a[0][1][0]))}
                            time_path, cycle_path = self.find_recursive_consolidation_k(cons, group)
                            
                            if cycle_path:
                                tmp = self.check_cycles(cycle_path)
                                self.update_timepoints(new_timepoints, tmp)
                                sys.stdout.write(" C[{0}]".format(len(tmp)))
                            else:
                                tmp = self.testing_mutual(first_arc, last_arc, solution_times, cons, time_path)
                                new_timepoints.update(tmp)
                                sys.stdout.write(" M[{0}]".format(len(tmp)))
                                #continue
                        break

                #self.testing_warm_start(solution_times)
                continue



            # 3. Consolidation is not possible due to direct time window (check sorted time windows for all commodities, choose arc[0] node and earliest time of the first/latest commodity that does not overlap)
            consolidation_window_timepoints = self.check_time_windows()
            if consolidation_window_timepoints:
                #new_timepoints.update(consolidation_window_timepoints)
                self.update_timepoints(new_timepoints, consolidation_window_timepoints)
                sys.stdout.write("Time window, {0}\n".format(len(consolidation_window_timepoints)))
                #print("Time window violated")
                continue

            # 4. Fix recursive or mutually exclusive Consolidation
            cons = {key: list(itertools.imap(operator.itemgetter(1), group)) for key, group in itertools.groupby(sorted(self.consolidations.iteritems(), key=lambda a: (a[0][0][0],a[0][1][0])), key=lambda a: (a[0][0][0],a[0][1][0]))}

            time_path, cycle_path = self.find_recursive_consolidation(cons)

            if cycle_path:
                #new_timepoints.update(self.check_cycles(cycle_path))
                tmp = self.check_cycles(cycle_path)
                self.update_timepoints(new_timepoints, tmp)
                sys.stdout.write(" Cyclic [{0}]".format(len(tmp)))
                #print("Recursive consolidation")
            else:
                #new_timepoints.update(self.check_consolidation(time_path, stats['consolidation'], cons))
                tmp = self.check_consolidation(time_path, stats['consolidation'], cons)
                self.update_timepoints(new_timepoints, tmp)
                sys.stdout.write(" Mutual [{0}]".format(len(tmp)))
                #print("Mutually exclusive consolidation")

        return iterations, len(new_timepoints), solve_time

    # testing only, will break if no new timepoints are added
    def update_timepoints(self, tp, new_tp):
        l = len(tp)
        tp.update(new_tp)
        #self.fixed_timepoints_model = l == len(tp)

    def find_recursive_consolidation_k(self, consolidation, group):
        cycle_path = []
        time_path = {}

        for k in group:
            path = self.solution_paths[k]
            forwards = {}
            self.rec_dispatch(forwards, k, (path[-2],path[-1]), consolidation, cycle_path)

            if cycle_path:
                return [], cycle_path

            backwards = {}

            t2 = self.commodities[k]['b'][1]
            for a in reversed(list(pairwise(path))):
                t2 -= self.transit(*a)
                backwards[(k,a)] = t2

            #self.prev_time(backwards, k, (path[0],path[1]), consolidation)

            time_path[k] = [(a[0], forwards[(k,a)], backwards[(k,a)]) for a in pairwise(path)]
            time_path[k].append((path[-1], forwards[(k, (path[-2],path[-1]))] + self.transit(path[-2], path[-1]), self.commodities[k]['b'][1]))

        return time_path, cycle_path


    # Sets feasible solution from LP result
    def testing_warm_start(self, solution_times):
        times = {k: {(a[0][0],a[1][0]): a[0][1] for a in pairwise(v)} for k,v in enumerate(solution_times)}

        for (k,a),x in self.x.items():
            x.start = 0

        # select the latest arc available, then use storage arcs to get to correct interval
        for k, v in times.items():
            (origin,dest) = self.origin_destination[k]
            c = self.commodities[k]

            tmp = sorted(v.items(),key=lambda x: x[1])
            tmp.append((None, c['b'][1]))

            for (i1,i2) in pairwise(tmp):
                # wait at origin till first dispatch
                if i1[0][0] == c['a'][0]:
                    while origin[2] <= i1[1]:
                        x = (origin, self.intervals.select(i1[0][0],origin[2],'*')[0])
                        self.x[(k,x)].start = 1
                        origin = x[1]
                        
                # find latest arc variable
                arcs = sorted([xa for xk,xa in self.x 
                                    if xk == k and                                  # commodity match
                                       i1[0][0] == xa[0][0] and i1[0][1] == xa[1][0] and    # arc nodes match
                                       xa[0][1] <= i1[1] and i1[1] < xa[0][2] and           # t1 <= t < t2
                                       xa[1][1] <= i2[1]], reverse=True)   # t3 <= t + tau

                x = arcs[0] if arcs else None
                self.x[(k,x)].start = 1

                # wait until next dispatch
                while x[1][2] <= i2[1]:
                    x = (x[1],self.intervals.select(i1[0][1], x[1][2], '*')[0])
                    self.x[(k,x)].start = 1

                # wait at destination until late time
                if i1[0][1] == c['b'][0]:
                    while x[1] != dest:
                        x = (x[1], self.intervals.select(i1[0][1],x[1][2],'*')[0])
                        self.x[(k,x)].start = 1

        self.model.update()

        for a,z in self.z.items():
            z.start = ceil(sum((self.commodities[k]['q'] for k in range(len(self.commodities)) if (k,a) in self.x and self.x[(k,a)].start > 0)) / self.network[a[0][0]][a[1][0]]['capacity'])
            
        self.model.update()
        return True

    def testing_timewindows(self, arc, broken_commodities, timepath):
        new_timepoints = set()

        # start with broken arc
        # foreach commodity, check 'path' length using consolidation times

        for k in broken_commodities:
            idx = (self.solution_paths[k[0]].index(arc[0][0]), self.solution_paths[k[1]].index(arc[0][0]))
            time = (timepath[k[0]][idx[0]][1], timepath[k[1]][idx[1]][1])

            dt = abs(time[1]-time[0])

            if time[0] > time[1]:
                k = (k[1],k[0])
                time = (time[1],time[0])
                idx = (idx[1],idx[0])

            # not sure - think about this... best time            
            #if not self.arc_rule_5(k[1], arc[0][0], time[0]):
            #    new_timepoints.add((arc[0][0], self.commodities[k[1]]['a'][1] + self.shortest_path(k[1], self.commodities[k[1]]['a'][0], arc[0][0])))

            new_timepoints.add((arc[0][0], self.commodities[k[1]]['a'][1] + self.shortest_path(k[1], self.commodities[k[1]]['a'][0], arc[0][0])))
            new_timepoints.add((arc[0][0], time[1]))

            # enforce first interval commodities cannot travel in second interval
            # if second interval is still valid: look at nodes from n2 -> destination until we break arc rule 7
            for i1,i2 in pairwise(timepath[k[0]][idx[0]:]):
                if not self.arc_rule_7(k[0], i2[0], i1[1] + dt + self.transit(i1[0],i2[0])):
                    break
                                
                new_timepoints.add((i2[0], i1[1] + dt + self.transit(i1[0],i2[0])))

            # enforce second interval commodities cannot travel in first interval
            # if first interval is still valid: look at nodes from n1 -> origin until we break arc rule 5
            for i1,i2 in reversed(list(pairwise(timepath[k[1]][:idx[1]+1]))):
                if not self.arc_rule_5(k[1], i1[0], i1[1] - dt):
                    break

                new_timepoints.add((i1[0], i1[1] - dt))

        ##
        ## update model
        ##
        if new_timepoints:
            new_arcs, del_arcs, new_intervals = self.add_timepoints(new_timepoints)
            self.update_model(new_arcs, del_arcs, new_intervals)

        return new_timepoints


    # trace path backwards (across commodities) to find conflicting consolidation(s)
    def testing_tracepath(self, stop_arc, time_path, k, node, cons, path, processed=set()):
        index = self.solution_paths[k].index(node)
        a = (node, self.solution_paths[k][index+1])

        if a == stop_arc:
            return True

        processed.add(k)

        # process the other commodities
        for k2 in set(next((g for g in cons[a] if k in g), [])).difference(processed):
            if k2 != k:
                index2 = self.solution_paths[k2].index(node)

                # find wait time at previous node
                if index2 > 0 and self.testing_tracepath(stop_arc, time_path, k2, self.solution_paths[k2][index2-1], cons, path):
                    path.append((time_path[k2][index2-1][:2]))
                    return True

       
        # prefer to follow path of current commodity
        # find wait time at previous node
        if index > 0 and self.testing_tracepath(stop_arc, time_path, k, self.solution_paths[k][index-1], cons, path):
            path.append((time_path[k][index-1][:2]))
            return True

        return False


    def testing_mutual(self, first_arc, last_arc, timepath, cons, time_path):
        new_timepoints = set()

        # trace backwards
        for k,t in last_arc[1]:
            path = [(last_arc[0][0][0], max(t for k,t in last_arc[1]))]
            self.testing_tracepath((first_arc[0][0][0], first_arc[0][1][0]), time_path, k, last_arc[0][0][0], cons, path, set())
            new_timepoints.update(path)

            #path = [(arc1[0][0], time_path[k][self.solution_paths[k].index(arc1[0][0])][1])]
            #self.find_consolidation(t1-t3, time_path, k, arc1[0][0], cons, path)
            #new_timepoints.update(path)
            

        ##
        ## update model
        ##
        if new_timepoints:
            new_arcs, del_arcs, new_intervals = self.add_timepoints(new_timepoints)
            self.update_model(new_arcs, del_arcs, new_intervals)

        return new_timepoints








    # solve the model
    def solve_lower_bound(self):
        sys.stdout.write('\n')

        def callback(model, where):
            if where == GRB.callback.MIP:
                objbst = model.cbGet(GRB.callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.callback.MIP_OBJBND)
                sys.stdout.write('{0:7.2%}\r'.format((objbst-objbnd)/objbst))
                sys.stdout.flush()


        self.model.update()
        #self.model.write('test.lp')
        self.model.optimize(callback)
        self.lower_bound = max(self.model.objBound, self.lower_bound) if self.lower_bound != None else self.model.objBound


    # print solution
    def printSolution(self):
        model = self.model

        if model.status == GRB.status.OPTIMAL:
            print('\nCost:', model.objVal)

            print("Path:", self.get_solution_paths())

            print("Consolidation: ")
            for c,v in self.get_consolidations().items():
                print(v)
        else:
            print('No solution')

    # get statistics
    def get_statistics(self):
        stats = {}
        stats['cost'] = self.model.objVal if self.model.status == GRB.status.OPTIMAL and self.incumbent == None else self.incumbent
        stats['nodes'] = len(self.network.nodes())
        stats['intervals'] = len(self.intervals)
        stats['arcs'] = len(self.x)
        stats['variables'] = len(self.model.getVars())
        stats['paths'] = self.get_solution_paths()
        stats['consolidation'] = self.get_consolidations()
        stats['commodities'] = len(self.commodities)
        stats['lower_bound'] = self.lower_bound

        try:
            m = self.model.presolve()
            stats['presolve_vars'] = len(m.getVars())
            stats['presolve_cons'] = len(m.getConstrs())
        except:
            stats['presolve_vars'] = 0
            stats['presolve_cons'] = 0

        time_points_per_node = [len(self.intervals.select(n, '*', '*')) for n in self.network.nodes()]
        stats['avg_points'] = sum(time_points_per_node)/float(len(time_points_per_node))
        stats['min_points'] = min(time_points_per_node)
        stats['max_points'] = max(time_points_per_node)

        return stats

    # print statistics
    def print_statistics(self):
        stats = self.get_statistics()

        print('')
        for k in sorted(stats):
            print("   {0}: {1}".format(k,stats[k]))
        print('')

    # get all arcs used for transport
    def get_solution(self):
        return tuplelist(k for k,v in self.x.items() if v.x > 0) if self.model.status == GRB.status.OPTIMAL else None

    ##
    ## From the solution, return all arcs used for a given commodity
    ##
    def get_solution_arcs(self):
        result = {}
        arcs = dict(((k,a[0]),a[1]) for (k,a),x in self.x.items() if x.x > 0)

        for k,(origin,dest) in self.origin_destination.items():
            i = origin
            result[k] = [i]

            while i != dest:
                i = arcs[(k,i)]
                result[k].append(i)
        
        return result

    ##
    ## From the solution, return the path chosen for each commodity
    ##
    def get_solution_paths(self):
        result = []

        if self.model.status == GRB.status.OPTIMAL:
            arcs = dict(((k,a[0]),a[1]) for (k,a),x in self.x.items() if x.x > 0)

            if self.model.status == GRB.status.OPTIMAL:
                for k, (origin,dest) in self.origin_destination.items():
                    i = origin
                    result.append([i[0]])
                
                    while i[0] != dest[0]:
                        i = arcs[(k,i)]
                
                        if i[0] != result[k][-1]:
                            result[k].append(i[0])
              
        return result

    ##
    ## From the solution, return the commodities that are consolidated together
    ##
    def get_consolidations(self):
        # ignore cycles that are not apart of actual path - can occur when x has 0 cost
        return {a: [k for k in range(len(self.commodities)) if (k,a) in self.x and self.x[k,a].x > 0 and a[0][0] in self.solution_paths[k]] 
                    for a, za in self.z.items() if za.x > 0} if self.model.status == GRB.status.OPTIMAL else None

    ##
    ## From the consolidations find the non-interacting commodities
    ##
    def get_consolidation_groups(self):
        consolidation_groups = map(set, self.consolidations.values())

        # repeat until irreducible        
        while True:
            l = len(consolidation_groups)
            consolidation_groups = set(map(frozenset, map(lambda x: reduce(lambda a,b: a if a.isdisjoint(b) else a.union(b), consolidation_groups, x), consolidation_groups)))

            if l == len(consolidation_groups):
                break            
         
        return consolidation_groups


    ## Partitions - same as consolidation groups, but uses network graph to find them (i.e. better!)
    def get_partitions(self):
        partitions = nx.Graph()
        partitions.add_nodes_from(range(len(self.commodities)))

        dispatch_intervals = {k: self.time_windows(k) for k in range(len(self.commodities))}

        def intersection(i1,i2):
            return i1 != None and i2 != None and i2[0] <= i1[1] and i1[0] <= i2[1]

        for k1 in range(len(self.commodities)):
            for k2 in range(k1 + 1, len(self.commodities)):
                if any(intersection(dispatch_intervals[k1][(a[0][0],a[1][0])], dispatch_intervals[k2][(a[0][0],a[1][0])]) for a,v in self.consolidations.items() if k1 in v and k2 in v and (a[0][0],a[1][0]) in dispatch_intervals[k1] and (a[0][0],a[1][0]) in dispatch_intervals[k2]):
                    partitions.add_edge(k1,k2)

        return nx.connected_components(partitions)

    # returns the min/max time window for a commodity that travels along arc a
    def time_windows(self, k):
        t1 = self.commodities[k]['a'][1]
        t2 = self.commodities[k]['b'][1]

        paths = list(pairwise(self.solution_paths[k]))

        early = [0] + list(accumulate((self.transit(*n) for n in paths)))
        late  = [0] + list(accumulate((self.transit(*n) for n in reversed(paths))))

        return {p: (int(t1 + e), int(t2 - l)) for (p,e,l) in zip(paths, early, reversed(late))}


    ## get the transit time between 2 nodes (looks nicer than direct access)
    def transit(self,n1,n2):
        return self.network[n1][n2]['weight']

    def shortest_path(self,k,n1,n2):
        return self.commodity_shortest_paths[k][n1].get(n2, None) if self.commodity_shortest_paths else self.shared_shortest_paths[n1].get(n2, None)

    def shortest_paths(self,k,n1):
        return self.commodity_shortest_paths[k][n1] if self.commodity_shortest_paths else self.shared_shortest_paths[n1]

    def triangle_check(self):
        for k in range(len(self.commodities)):
            for n1 in self.network.nodes():
                for n2 in self.network.nodes():
                    n = self.network[n1].get(n2, None)
                    s = self.shortest_path(k,n1,n2)

                    if n != None and s != None and n['weight'] > s:
                        raise Exception("Failed triangle inequality at {0}->{1}, {2} vs {3}".format(n1,n2,n['weight'],s))

        return True

    def create_shortest_paths(self, opt=shortest_path_option.shared):
        self.shared_shortest_paths = nx.shortest_path_length(self.network, weight='weight')
        self.commodity_shortest_paths = []

        if opt == shortest_path_option.commodity:
            # create shortest paths excluding destination node from calculations
            for k,c in enumerate(self.commodities):
                network_copy = self.network.copy()
                network_copy.remove_node(c['b'][0]) # remove destination
                network_copy.remove_node(c['a'][0]) # remove origin

                path = nx.shortest_path_length(network_copy, weight='weight')

                # merge destination path with all other paths
                # cluster server uses networkx version 1.7 which does not like a target without a source
                #dest = nx.shortest_path_length(self.network, source=None, target=c['b'][0], weight='weight')
                #for n,t in dest.items():
                #    if n in path:
                #        path[n][c['b'][0]] = t

                for n in self.network.nodes():
                    if n in path:
                        path[n][c['b'][0]] = nx.shortest_path_length(self.network, source=n, target=c['b'][0], weight='weight')

                network_copy = self.network.copy()
                network_copy.remove_node(c['a'][0]) # remove origin

                path[c['b'][0]] = nx.shortest_path_length(network_copy, source=c['b'][0], weight='weight')

                network_copy = self.network.copy()
                network_copy.remove_node(c['b'][0]) # remove destination

                path[c['a'][0]] = nx.shortest_path_length(network_copy, source=c['a'][0], weight='weight')
                path[c['a'][0]][c['b'][0]] = self.shared_shortest_paths[c['a'][0]][c['b'][0]]
            
                self.commodity_shortest_paths.append(path)


    ## creates arcs/nodes for time horizon
    def trivial_network(self):
        return tuplelist((n,t) for n in self.network.nodes() for t in [self.S,self.T])

    ##
    ## Creates intervals / arcs from time points
    ##
    def create(self, time_points):
        time_points.sort()
        
        intervals = tuplelist(((n, t1, t2) for n in self.network.nodes() for t1,t2 in pairwise(map(operator.itemgetter(1), time_points.select(n, '*')))))
        intervals.sort()

        return (intervals, self.create_arcs(intervals))


    ##
    ## Creates valid arcs (using lower bound) for each commodity and then shares 'redirected' arcs across all commodities
    ##
    def create_arcs(self, intervals):
        interval_cache = {n: intervals.select(n,'*', '*') for n in self.network.nodes()}

        redirected_arcs = set()
        arcs = {}

        # Create arcs ((n1, t1, t2), (n2, t3, t4)) pairs
        for k,c in enumerate(self.commodities):
            origin,dest = c['a'], c['b']

            # setup storage arcs
            origin_to_arc = self.shortest_paths(k, origin[0])
            arcs[k] = tuplelist(arc for n in self.network.nodes() if n in origin_to_arc and self.shortest_path(k, n, dest[0]) != None
                                    for arc in pairwise(interval_cache[n]) if self.is_arc_valid(arc, origin, dest, origin_to_arc.get(n, None), 0, self.shortest_path(k, n, dest[0])))

            # for each physical arc
            for e in self.network.edges():
                if self.shortest_path(k, origin[0], e[0]) != None and self.shortest_path(k, e[1], dest[0]) != None:
                    arcs[k].extend(self.create_arcs_between_nodes(k, e[0], e[1], origin, dest, interval_cache[e[0]], interval_cache[e[1]], redirected_arcs))    

        # add 'redirected' arcs to all commodities so we don't miss consolidation opportunities
        for k,c in enumerate(self.commodities):
            origin,dest = c['a'], c['b']
            origin_to_arc = self.shortest_paths(k, origin[0])
 
            missing_arcs = set(a for a in redirected_arcs 
                                 if a not in arcs[k] and self.is_arc_valid(a, origin, dest, origin_to_arc.get(a[0][0], None), self.transit(a[0][0], a[1][0]), self.shortest_path(k, a[1][0], dest[0])))

            # add missing arcs if valid for k
            arcs[k].extend(missing_arcs)

        return arcs
    
    def add_redirected_arc(self, k, a, redirected_arcs):
        redirected_arcs.add(a)

        # store redirected arcs for each commodity - when adding new timepoints they can cause issues
        if a not in self.redirected_arcs:
            self.redirected_arcs[a] = set([k])
        else:
            self.redirected_arcs[a].add(k)

    def remove_redirected_arc(self, k, a):
        if a in self.redirected_arcs:
            self.redirected_arcs[a].remove(k)

            # if removed all commodities remove entry
            if not self.redirected_arcs[a]:
                del self.redirected_arcs[a]


    ##
    ## Creates all arcs for a commodity for all given intervals between two nodes.  Also lists any 'redirected' arcs (using shortest paths)
    ##
    def create_arcs_between_nodes(self, k, n1, n2, origin, dest, origin_intervals, destination_intervals, redirected_arcs):
        new_arcs = []

        it = iter(destination_intervals) # for each destination interval
        i2 = next(it)

        transit_time = self.transit(n1,n2)

        origin_to_arc = self.shortest_path(k, origin[0], n1)
        arc_to_dest = self.shortest_path(k, n2, dest[0])
        arc_to_arc = transit_time

        # for each origin interval
        for i1 in origin_intervals:
            time = i1[1] + transit_time

            # the first / previously used interval is also valid for this interval
            if time < i2[2] and (i1 != None and i2 != None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                new_arcs.append((i1, i2))
            else:
                # skip to correct interval
                while i2 != None and time >= i2[2]:
                    i2 = next(it, None)

                # keep skipping if invalid, up until latest time
                while i2 != None and i1[2] + transit_time >= i2[2]:
                    if (i1 != None and i2 != None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                        new_arcs.append((i1,i2))

                        # redirected
                        if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                            #redirected_arcs.add((i1,i2))
                            self.add_redirected_arc(k, (i1,i2), redirected_arcs)

                        break

                    i2 = next(it, None)

                # we are done for this arc
                if i2 == None:
                    break

                # possibly the last transit time (from above loop) is valid
                if i1[2] + transit_time < i2[2] and (i1 != None and i2 != None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                    new_arcs.append((i1,i2))

                    # redirected
                    if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                        #redirected_arcs.add((i1,i2))
                        self.add_redirected_arc(k, (i1,i2), redirected_arcs)

        return new_arcs


    def is_arc_valid(self, arc, origin, dest, origin_to_arc, arc_to_arc, arc_to_dest):
        return not ((arc[0] == None or arc[1] == None) or                                   # is valid node
                    (origin_to_arc == None or arc_to_dest == None) or                       # 2. invalid path
                    (arc[1][0] == origin[0] and arc[0][0] != origin[0]) or                  # 1. no inflow into origin (except storage arc)
                    (arc[0][0] == dest[0] and arc[1][0] != dest[0]) or                      # 1. no outflow from destination (except storage arc)
                    (dest[1] - arc_to_dest < origin[1] + origin_to_arc + arc_to_arc) or     # 3. cannot route via this arc using shortest paths (assumes transit time in arc)
#                    (arc[0][1] + arc_to_arc >= arc[1][2]) or                                # 4. transit time within interval is valid?
                    (origin[1] + origin_to_arc >= arc[0][2]) or                             # 5. cannot reach arc in time for dispatch
                    (origin[1] + origin_to_arc + arc_to_arc >= arc[1][2]) or                # 6. arc is invalid due to actual dispatch time window
                    (arc[1][1] + arc_to_dest > dest[1]) or                                  # 8. cannot reach destination in time - from i2
                    (arc[0][1] + arc_to_arc + arc_to_dest > dest[1]))                       # 7. cannot reach destination in time - from i1

        #return (arc[0] != None and arc[1] != None and (arc[1][0] != origin[0] or arc[0][0] == origin[0]) and (arc[0][0] != dest[0] or arc[1][0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < arc[0][2]) and (origin[1] + origin_to_arc + arc_to_arc < arc[1][2]) and (arc[1][1] + arc_to_dest <= dest[1]) and (arc[0][1] + arc_to_arc + arc_to_dest <= dest[1])) 


    ##
    ## Ensure that path length for each commodity is less than it's time window
    ##
    def check_path_length(self):
        new_timepoints = set()
        start_end = {}

        for k,path in enumerate(self.solution_paths):
            time = self.commodities[k]['a'][1] + sum(self.transit(a[0],a[1]) for a in pairwise(path))
            first = True

            # Path is longer than time window
            if time > self.commodities[k]['b'][1]:
                # trivial, but easy guarantee - add all time points along path
                for a in reversed(list(pairwise(path))):
                    time -= self.transit(a[0],a[1])
                    new_timepoints.add((a[0], time))

#                t2 = time
#                t = 0
#                l = 0
#                a6 = None
#                a7 = None

#                c = self.commodities[k]

#                # test code
#                for a in triple(path):
#                    tmp = (c['b'][1] - self.transit(a[1],a[2]) - self.shortest_path(k, a[2], c['b'][0]), c['a'][1] + self.shortest_path(k, c['a'][0], a[0]) + self.transit(a[0],a[1]))

#                    if self.shortest_path(k, c['a'][0], a[0]) + self.transit(a[0],a[1]) + self.transit(a[1],a[2]) + self.shortest_path(k, a[2], c['b'][0]) > c['b'][1] - c['a'][1]:
#                        a7 = (a[1],self.shortest_path(k, c['a'][0], a[0]) + self.transit(a[0],a[1]) + c['a'][1])
#                        new_timepoints.add(a7)
#                        break

#                for a0,a1,a2,a3 in quad(path):
#                    tmp = (c['b'][1] - self.transit(a1,a2) - self.transit(a2,a3) - self.shortest_path(k, a3, c['b'][0]), c['a'][1] + self.shortest_path(k, c['a'][0], a0) + self.transit(a0,a1))

#                    if self.shortest_path(k, c['a'][0], a0) + self.transit(a0,a1) + self.transit(a1,a2) + self.transit(a2,a3) + self.shortest_path(k, a3, c['b'][0]) > c['b'][1] - c['a'][1]:
#                        a7 = (a1,self.shortest_path(k, c['a'][0], a0) + self.transit(a0,a1) + c['a'][1])
#                        new_timepoints.add(a7)

#                        a7 = (a2,self.shortest_path(k, c['a'][0], a1) + self.transit(a1,a2) + c['a'][1])
#                        new_timepoints.add(a7)

#                        break


#                t3 = c['b'][1]

#                for a in pairwise(path):
#                    # min arc 6
#                    #if a6 == None:
#                    #   if t > self.shortest_path(k, c['a'][0], a[0]):
#                    #        a6 = (a[0],l)

#                    t1 = (t3 - self.shortest_path(k, a[1], c['b'][0]) - self.transit(a[0],a[1]), self.shortest_path(k, c['a'][0], a[0]) + self.transit(a[0],a[1]) + c['a'][1])


#                    t += self.transit(a[0],a[1])
#                    l += self.shortest_path(k, a[0], a[1])

#                    if l > self.shortest_path(k, c['a'][0], a[0]) + self.transit(a[0],a[1]):
#                        a6 = (a[0],c['a'][1] + l-self.shortest_path(k, a[0], a[1]))

#                    # max arc 7
#                    if l + self.shortest_path(k, a[1], c['b'][0]) > c['b'][1] - c['a'][1]:
#                        break
#                    else:
#                        a7 = (a[1],self.shortest_path(k, c['a'][0], a[0]) + self.transit(a[0],a[1]) + c['a'][1])

##                new_timepoints.add(a7)


#                for a in reversed(list(pairwise(path))):
#                    # want first that breaks arc rule 7
#                    if first:
#                        if not self.arc_rule_7(k, a[1], time):
#                            time -= self.transit(a[0],a[1])
#                            continue
#                        else:
#                            new_timepoints.add((a[1], time))
#                            start_end[k] = (a[1],a[1])
#                            first = False

#                    time -= self.transit(a[0],a[1])

#                    # check arc rule 6
#                    if self.commodities[k]['a'][1] + self.shortest_path(k, self.commodities[k]['a'][0], a[0]) < time:
#                        new_timepoints.add((a[0], time))
#                    else:
#                        start_end[k] = (a[1],start_end[k][1])
#                        break
        ##
        ## update model
        ##
        if new_timepoints:
            #solution = self.get_solution_arcs()
            new_arcs, del_arcs, new_intervals = self.add_timepoints(new_timepoints)

            self.update_model(new_arcs, del_arcs, new_intervals)

        #for k,s in start_end.items():
        #    # turn off path to destination
        #    for i1,i2 in solution:
        #        if i2[0] == a[1]:
        #            break

        #        self.x[(k,(i1,i2))].start = 0

        #    # redirect path to shortest path
        #    shortest_path = nx.shortest_path(self.network, a[1], self.commodities[k]['b'][0])

        

        return new_timepoints

    def arc_rule_7(self, k, n2, t1_plus_transit):
        return t1_plus_transit + self.shortest_path(k, n2, self.commodities[k]['b'][0]) <= self.commodities[k]['b'][1]

    def arc_rule_6(self, k, n1, n2, t4):
        return self.commodities[k]['a'][1] + self.shortest_path(k, self.commodities[k]['a'][0], n1) + self.transit(n1,n2) < t4

    def arc_rule_5(self, k, n1, t2):
        return self.commodities[k]['a'][1] + self.shortest_path(k, self.commodities[k]['a'][0], n1) < t2


    ##
    ## Ensure that all commodities that consolidate on an arc have valid time windows
    ##
    def check_time_windows(self):
        new_timepoints = set()
        used_groups = set()

        # order by largest consolidations, only process one window per 'consolidation group'
        # TODO: is there a better choice for picking which timewindows to break?
        for arc, group in sorted(self.consolidations.items(), key=lambda x: len(x[1]), reverse=True):
            # we are sorted by group size, if we hit no consolidation then stop processing
            if len(group) == 1:
                break

            key = next(g for g in self.consolidation_groups if group[0] in g)

            if key not in used_groups:
                time_windows = sorted((self.time_window(k, arc[0]), k) for k in group)
                min_dispatch = time_windows[0][0][1]
                first_interval_commodities = [time_windows[0][1]]

                for t in time_windows[1:]:
                    min_dispatch = min(t[0][1], min_dispatch)

                    if t[0][0] > min_dispatch:

                        #c = self.commodities[t[1]]
                        #new_timepoints.add((arc[0][0], c['b'][1] - self.shortest_path(k, arc[1][0], c['b'][0])))

                        #c = self.commodities[time_windows[time_windows.index(t)][1]]
                        #new_timepoints.add((arc[0][0], c['a'][1] + self.shortest_path(k, c['a'][0], arc[0][0])))


                        new_timepoints.add((arc[0][0], t[0][0]))

                        # enforce first interval commodities cannot travel in second interval
                        for k in first_interval_commodities:
                            t1 = t[0][0]

                            # if second interval is still valid: look at nodes from n2 -> destination until we break arc rule 7
                            for n1,n2 in pairwise(self.solution_paths[k][self.solution_paths[k].index(arc[0][0]):]):
                                t1 += self.transit(n1,n2)

                                if not self.arc_rule_7(k, n2, t1):
                                    break
                                
                                new_timepoints.add((n2, t1))

                        # enforce second interval commodities cannot travel in first interval
                        for tw, k in time_windows[time_windows.index(t):]:
                            t2 = t[0][0]

                            # if first interval is still valid: look at nodes from n1 -> origin until we break arc rule 5
                            for n1,n2 in reversed(list(pairwise(self.solution_paths[k][:self.solution_paths[k].index(arc[1][0])]))):
                                t2 -= self.transit(n1,n2)

                                if not self.arc_rule_5(k, n1, t2):
                                    break

                                new_timepoints.add((n1, t2))

                        used_groups.add(key)  # don't process this consolidation group again
                        break

                    first_interval_commodities.append(t[1])
        
        if new_timepoints:
            new_arcs, del_arcs, new_intervals = self.add_timepoints(new_timepoints)
            self.update_model(new_arcs, del_arcs, new_intervals)

            return new_timepoints


    ##
    ## Check cycles
    ##
    def check_cycles(self, cycle_path):
        # get commodity withh shortest time window
        k = min(((self.time_window(k, cycle_path[0][1]), k) for k in cycle_path[0][0]), key=lambda x: x[0][1] - x[0][0])[1]
            
        # get latest start time (or last item if equal) arc for k
        start, i, arc = max((self.time_window(k, a)[0], i, a) for i,(g,a) in enumerate(cycle_path))

        cycle_groups = {a: g for g,a in cycle_path[:-1]} 
        cycle_arcs = [a for g,a in cycle_path[:-1]]

        i = cycle_arcs.index(arc)
        path = cycle_arcs[i:] + cycle_arcs[:i]

        new_timepoints = set()

        t = start
        for a in itertools.cycle(path):
            new_timepoints.add((a[0],t))
            t += self.transit(a[0],a[1])

            if t + self.shortest_path(k, a[1], self.commodities[k]['b'][0]) > self.commodities[k]['b'][1] and k in cycle_groups[a] or t >= self.commodities[k]['b'][1]:
                break

        if new_timepoints:
            new_arcs, del_arcs, new_intervals = self.add_timepoints(new_timepoints)
            self.update_model(new_arcs, del_arcs, new_intervals)

        return new_timepoints

    # trace path backwards (across commodities) to find conflicting consolidation(s)
    def find_consolidation(self, overstep, time_path, k, node, cons, path, processed=set()):
        if overstep <= 0:
            return True

        processed.add(k)
        index = self.solution_paths[k].index(node)
        a = (node, self.solution_paths[k][index+1])

        # process the other commodities
        for k2 in set(next((g for g in cons[a] if k in g), [])).difference(processed):
            if k2 != k:
                index2 = self.solution_paths[k2].index(node)

                # find wait time at previous node
                wait = (time_path[k2][index2-1][1] - (time_path[k2][index2-2][1] + self.transit(self.solution_paths[k2][index2-2], self.solution_paths[k2][index2-1]) if index2 > 1 else self.commodities[k2]['a'][1])) if index2 > 0 else 0

                if index2 > 0 and self.find_consolidation(overstep - wait, time_path, k2, self.solution_paths[k2][index2-1], cons, path):
                    path.append((time_path[k2][index2-1][:2]))
                    return True

       
        # prefer to follow path of current commodity
        # find wait time at previous node
        wait = (time_path[k][index-1][1] - (time_path[k][index-2][1] + self.transit(self.solution_paths[k][index-2], self.solution_paths[k][index-1]) if index > 1 else self.commodities[k]['a'][1])) if index > 0 else 0

        if index > 0 and self.find_consolidation(overstep - wait, time_path, k, self.solution_paths[k][index-1], cons, path):
            path.append((time_path[k][index-1][:2]))
            return True


        return False

    ##
    ## Check mutually exclusive consolidations
    ##
    def check_consolidation(self, time_path, feasible_consolidations, cons):
        new_timepoints = set()
        sols = {key: list(itertools.imap(operator.itemgetter(1), group)) for key, group in itertools.groupby(sorted(feasible_consolidations.iteritems()), key=lambda a: a[0][0])}

        broken = {k: [g for g in v if g not in sols[k]] for k,v in cons.items() if [g for g in v if g not in sols[k]]}

        for group in self.consolidation_groups:
            ## find first break (dispatch time > time window)
            #breaks = filter(lambda x: x[0] != None, [(next((t for t in time_path[k] if t[1] > t[2]), None),k) for k in group])

            #if breaks:
            #    b,k = min(breaks)

            #    # calculate overstep
            #    overstep = b[1]-b[2]

            #    # find another consolidated dispatch along path so that total wait time > overstep
            #    path = [b[:2]]
            #    self.find_consolidation(overstep, time_path, k, b[0], cons, path)
    
            #    new_timepoints.update(path)

            for k in group:
                # find first break (dispatch time > time window)
                b = next((t for t in time_path[k] if t[1] > t[2]), None)

                if b != None:
                    # calculate overstep
                    overstep = b[1]-b[2]

                    # find another consolidated dispatch along path so that total wait time > overstep
                    path = [b[:2]]
                    self.find_consolidation(overstep, time_path, k, b[0], cons, path)
    
                    new_timepoints.update(path)
                    break

                #test = map(lambda a: (a[0], self.time_window(k, a)), pairwise(self.solution_paths[k]))
                #test.append((self.solution_paths[k][-1], (test[-1][1][0] + self.transit(test[-1][0], self.solution_paths[k][-1]), self.commodities[k]['b'][1])))

                #new_timepoints.extend((t[0],t[1]) for t in time_path[k] if t[1] > t[2])

        if new_timepoints:
            new_arcs, del_arcs, new_intervals = self.add_timepoints(new_timepoints)
            self.update_model(new_arcs, del_arcs, new_intervals)

        return new_timepoints

        #new_timepoints = {}
        #sols = {key: list(itertools.imap(operator.itemgetter(1), group)) for key, group in itertools.groupby(sorted(feasible_consolidations.iteritems()), key=lambda a: a[0][0])}

        #for arc,groups in cons.items():
        #    for group in groups:
        #        # find a consolidation that is unbroken in LP2 that hasn't been already been processed (as a member in consolidation_groups)
        #        key = next(g for g in self.consolidation_groups if group[0] in g)

        #        if len(group) > 1 and not new_timepoints.has_key(key) and not group in sols[arc]:
        #            new_timepoints[key] = self.mutually_exclusive_timepoints(arc, cons, group, time_paths)


        #new_arcs, del_arcs, new_intervals = self.add_timepoints(list(itertools.chain.from_iterable(new_timepoints.values())))
        #self.update_model(new_arcs, del_arcs, new_intervals)

        #return list(itertools.chain.from_iterable(new_timepoints.values()))


    def mutually_exclusive_timepoints(self, arc, cons, group, time_paths):
        # find a previous consolidation
        for k in group:
            path = list(pairwise(self.solution_paths[k]))
            index = path.index(arc)

            #previous = filter(lambda x: x[1] < self.T, 
            #                  map(lambda a: (a[0], time_paths[k][a[0]][0]), 
            #                      itertools.takewhile(lambda a: time_paths[k][a[0]][0] > time_paths[k][a[0]][1], 
            #                                          reversed(path[:index]))))
            #if previous:
            #    return previous + [(arc[0], time_paths[k][arc[0]][0])]

            previous = []

            for a in reversed(path[:index]):
                if time_paths[k][a[0]][0] < self.T:
                    previous.append((a[0], time_paths[k][a[0]][0]))

                if time_paths[k][a[0]][0] <= time_paths[k][a[0]][1]:
                    break

            if previous:
                return previous + [(arc[0], time_paths[k][arc[0]][0])]



            #for a in reversed(path[:index]):
            #    if [g for g in cons[a] if len(g) > 1 and k in g]:
            #        tmp = [(p[1], time_paths[k][p[1]][0]) for p in path[path.index(a):index] if time_paths[k][p[1]][0] > time_paths[k][p[1]][1] and time_paths[k][p[1]][0] < self.T]

            #        if tmp:
            #            return tmp

        # find a previous consolidation
        for k in group:
            path = list(pairwise(self.solution_paths[k]))
            index = path.index(arc)

            # find a next consolidation
            future = filter(lambda x: x[1] < self.T, 
                            map(lambda a: (a[0], time_paths[k][a[0]][0]), 
                                itertools.takewhile(lambda a: time_paths[k][a[0]][0] > time_paths[k][a[0]][1], 
                                                    path[index:])))
            if future:
                return future

            #for a in path[index+1:]:
            #    if [g for g in cons[a] if len(g) > 1 and k in g]:
            #        tmp = [(p[0], time_paths[k][p[0]][0]) for p in path[index:path.index(a)+2] if time_paths[k][p[0]][0] > time_paths[k][p[0]][1] and time_paths[k][p[0]][0] < self.T]

            #        if tmp:
            #            return tmp
        return []


    def find_recursive_consolidation(self, consolidation):
        cycle_path = []
        time_path = {}

        for k,path in enumerate(self.solution_paths):
            forwards = {}
            self.rec_dispatch(forwards, k, (path[-2],path[-1]), consolidation, cycle_path)

            if cycle_path:
                return [], cycle_path

            backwards = {}

            t2 = self.commodities[k]['b'][1]
            for a in reversed(list(pairwise(path))):
                t2 -= self.transit(*a)
                backwards[(k,a)] = t2

            #self.prev_time(backwards, k, (path[0],path[1]), consolidation)

            time_path[k] = [(a[0], forwards[(k,a)], backwards[(k,a)]) for a in pairwise(path)]
            time_path[k].append((path[-1], forwards[(k, (path[-2],path[-1]))] + self.transit(path[-2], path[-1]), self.commodities[k]['b'][1]))

        return time_path, cycle_path

    # gets the dispatch times for path/consolidation.  If recursive consolidation exists it gets the cycle path
    def rec_dispatch(self, time_paths, k, next_arc, consolidation, path):
        if (k,next_arc) in time_paths:
            return time_paths[(k,next_arc)]

        max_dispatch = time_paths[(k,next_arc)] = None # stop recursive consolidations
        group = next((g for g in consolidation[next_arc] if k in g), None) # there will always be only one
        
        if group == None:
            return None #error shouldn't occur!!

        # dispatch time is maximum in group
        for k2 in group:
            i = self.solution_paths[k2].index(next_arc[0])

            if i == 0:
                max_dispatch = max(self.commodities[k2]['a'][1], max_dispatch)
            else:
                prev = (self.solution_paths[k2][i-1], next_arc[0])
                t1 = self.rec_dispatch(time_paths, k2, prev, consolidation, path)
            
                if t1 == None:
                    if len(path) < 2 or path[0][1] != path[-1][1]:  # stop collecting path when we have cycle
                        path.append((group,next_arc))  # get cycle of recursive consolidation
                    return None

                max_dispatch = max(t1 + self.network[prev[0]][prev[1]]['weight'], max_dispatch)
                    
        time_paths[(k,next_arc)] = max_dispatch
        return max_dispatch


    def prev_time(self, time_paths, k, prev, consolidation):
        if (k,prev) in time_paths:
            return time_paths[(k,prev)]

        i = self.solution_paths[k].index(prev[1])
        t1 = (self.prev_time(time_paths, k, (prev[1], self.solution_paths[k][i+1]), consolidation) if i + 1 < len(self.solution_paths[k]) else self.commodities[k]['b'][1]) - self.transit(prev[0],prev[1])

        time_paths[(k,prev)] = t1 # stop infinite recursion

        group = [g for g in consolidation[prev] if k in g][0] # there will always be only one            
        time_paths[(k,prev)] = min([t1] + [self.prev_time(time_paths, k2, prev, consolidation) for k2 in group if k2 != k])

        return time_paths[(k,prev)]


    # returns the min/max time window for a commodity that travels along arc a
    def time_window(self, k, a):
        path = self.solution_paths[k]

        t1 = self.commodities[k]['a'][1]
        t2 = self.commodities[k]['b'][1]

        start = True

        for n1,n2 in pairwise(path):
            if n1 == a[0]:
                start = False

            if start:
                t1 += self.transit(n1,n2)
            else:
                t2 -= self.transit(n1,n2)
        
        return (t1,t2)


    # split the current intervals and then map new/current arcs
    def split_intervals(self, new_timepoints):
        new_intervals = {}
        original_intervals = tuplelist(self.intervals)  # create copy for lookup

        for node,time in sorted(new_timepoints):
            # find current interval that gets split by new timepoint - should always succeed if time >= 0 and time <= T
            interval = next((i for i in self.intervals.select(node, '*', '*') if time > i[1] and time < i[2]), None)
            original_interval = next((i for i in original_intervals.select(node, '*', '*') if time >= i[1] and time < i[2]), None)

            # ignore timepoints that are already in the system
            if interval == None:
                continue

            # split interval
            split = [(node, interval[1], time), (node, time, interval[2])]

            if original_interval not in new_intervals:
                new_intervals[original_interval] = split
            else:
                new_intervals[original_interval].remove(interval)
                new_intervals[original_interval].extend(split)

            self.intervals.remove(interval)
            self.intervals.extend(split)

        self.intervals.sort()
        return new_intervals

    # splits current intervals and creates new arcs based on these splits (does not update the model!)
    def add_timepoints(self, new_timepoints):
        new_intervals = self.split_intervals(new_timepoints)
        interval_cache = {n: sorted(self.intervals.select(n,'*', '*')) for n in self.network.nodes()}  # force sort on the intervals - for some reason they stopped being in order even though they should be in the previous function call


        ##testing
        ##
        #t_arcs = self.get_solution_arcs()



        # create / rename / delete arcs based on the new intervals
        new_arcs, ren_arcs, del_arcs = ([],[],[])
        redirected_arcs = set()

        for k,c in enumerate(self.commodities):
            origin,dest = c['a'], c['b']

            new_arcs.append(set())
            ren_arcs.append({})
            del_arcs.append([])
            
            for original_interval, split_intervals in new_intervals.items():
                split_intervals.sort()  # enforce order

                origin_to_arc = self.shortest_path(k, origin[0], original_interval[0])
                arc_to_dest = self.shortest_path(k, original_interval[0], dest[0])

                ## new storage arcs
                storage_intervals = self.intervals.select(original_interval[0], '*', split_intervals[0][1]) + split_intervals + self.intervals.select(original_interval[0], split_intervals[-1][2], '*')  # connect to existing intervals

                storage_arcs = [arc for arc in zip(storage_intervals, storage_intervals[1:]) 
                                    if self.is_arc_valid(arc, origin, dest, origin_to_arc, 0, arc_to_dest)]

                new_arcs[k].update(storage_arcs)
                self.arcs[k].extend(storage_arcs)

                #
                # outflow arcs
                #
                outflow = self.arcs[k].select(original_interval, '*')
                for arc in outflow:
                    self.arcs[k].remove(arc)
                    del_arcs[k].append(arc)

                    # skip storage & redirected arcs
                    redirect_arc = self.redirected_arcs.get(arc)

                    if arc[0][0] != arc[1][0] and (redirect_arc == None or k in redirect_arc):
                        self.remove_redirected_arc(k,arc) # if redirect remove
                        new_interval_arcs = set(self.create_arcs_between_nodes(k, arc[0][0], arc[1][0], origin, dest, split_intervals, interval_cache[arc[1][0]], redirected_arcs))

                        self.arcs[k].extend(list(new_interval_arcs.difference(new_arcs[k])))
                        new_arcs[k].update(new_interval_arcs)

            #
            # inflow arcs
            #
            for original_interval, split_intervals in new_intervals.items():
                split_intervals.sort()  # enforce order

                inflow = self.arcs[k].select('*', original_interval)
                for arc in inflow:
                    self.arcs[k].remove(arc)
                    del_arcs[k].append(arc)

                    # skip storage & redirect arcs
                    redirect_arc = self.redirected_arcs.get(arc)

                    if arc[0][0] == arc[1][0] or not (redirect_arc == None or k in redirect_arc):
                        continue

                    self.remove_redirected_arc(k,arc) # if redirected remove it also

                    i1 = arc[0]
                    transit_time = self.transit(arc[0][0], arc[1][0])
                    time = i1[1] + transit_time

                    origin_to_arc = self.shortest_path(k, origin[0], arc[0][0])
                    arc_to_dest = self.shortest_path(k, arc[1][0], dest[0])
                    arc_to_arc = self.transit(arc[0][0], arc[1][0])

                    # find appropriate inflow interval
                    tmp_arc = next(((i1, i2) for i2 in split_intervals if time < i2[2] and (i1 != None and i2 != None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1]))), None)

                    if tmp_arc != None:
                        self.arcs[k].append(tmp_arc)
                        new_arcs[k].add(tmp_arc)

                        if tmp_arc[0][1] + arc_to_arc < tmp_arc[1][1]:
                            #redirected_arcs.add(tmp_arc)
                            self.add_redirected_arc(k, tmp_arc, redirected_arcs)


        # add non-standard arcs created by commodities time windows
        for k,c in enumerate(self.commodities):
            origin,dest = c['a'], c['b']
            origin_to_arc = self.shortest_paths(k, origin[0])
 
            missing_arcs = set(a for a in redirected_arcs 
                                 if a not in self.arcs[k] and self.is_arc_valid(a, origin, dest, origin_to_arc.get(a[0][0], None), self.transit(a[0][0], a[1][0]), self.shortest_path(k, a[1][0], dest[0])))

            # add missing arcs if valid for k
            self.arcs[k].extend(missing_arcs)
            new_arcs[k].update(missing_arcs)

        return new_arcs, del_arcs, new_intervals

    ##
    ## Update the model
    ##

    def update_model(self, new_arcs, del_arcs, new_intervals):
        origin_destination = {k: (self.find_interval(c['a']), self.find_interval(c['b'])) for k,c in enumerate(self.commodities)}

        # remove the arc by setting UB to 0
        for k,arcs in enumerate(del_arcs):
            #self.model.setAttr("UB", [self.x[(k,a)] for a in arcs], [0]*len(arcs))

            for a in set(arcs):
                self.model.setAttr("UB", [self.x[(k,a)]], [0])

                # dispatch arc
                if a[0][0] != a[1][0]:
                    self.model.chgCoeff(self.constraint_consolidation[a], self.x[k,a], 0)
                    
                    if self.constraint_cycle != None:
                        self.model.chgCoeff(self.constraint_cycle[(k,a[0][0])], self.x[k,a], 0)

                    #self.model.chgCoeff(self.constraint_path_length[k], self.x[k,a], 0)

                self.model.chgCoeff(self.constraint_flow[(k,a[1])], self.x[k,a], 0)
                self.model.chgCoeff(self.constraint_flow[(k,a[0])], self.x[k,a], 0)

                #test            
                self.model.remove(self.x[(k,a)])
                del self.x[(k,a)]

        ## remap origin/destination intervals
        #for k in range(len(self.commodities)):
        #    for old_interval, split_intervals in new_intervals.items():
        #        new_interval = None

        #        # the split interval is origin, so remap to new origin
        #        if self.origin_destination[k][0] == old_interval:
        #            new_interval = origin_destination[k][0]
               
        #        # the split interval is dest, so remap to new dest
        #        elif self.origin_destination[k][1] == old_interval:
        #            new_interval = origin_destination[k][1]

        #        if new_interval != None and (k,new_interval) not in self.constraint_flow and (k,old_interval) in self.constraint_flow:
        #            self.constraint_flow[(k,new_interval)] = self.constraint_flow[(k,old_interval)]
        #            self.constraint_flow[(k,new_interval)].constrName = 'flow' + str((k,new_interval))
        #            del self.constraint_flow[(k,old_interval)]

        self.origin_destination = origin_destination


        for old_interval in new_intervals:
            for k in range(len(self.commodities)):
                if (k,old_interval) in self.constraint_flow:
                    self.model.remove(self.constraint_flow[(k,old_interval)])
                    del self.constraint_flow[(k,old_interval)]

        # only dispatch arcs (we don't care about storage)
        all_arcs = set(a for k,arcs in enumerate(del_arcs) for a in arcs if a[0][0] != a[1][0])

        self.model.setAttr("UB", [self.z[a] for a in all_arcs], [0]*len(all_arcs))

        for a in all_arcs:
            self.model.remove(self.z[a])
            del self.z[a]

            #self.model.remove(self.constraint_consolidation[a])
            del self.constraint_consolidation[a]

        self.add_vars(new_arcs)
        self.update_constraints(new_arcs)


    # add new arc variables
    def add_vars(self, new_arcs):
        self.x.update({(k, a): self.model.addVar(obj=(self.network[a[0][0]][a[1][0]]['var_cost'] * self.commodities[k]['q'] if a[0][0] != a[1][0] else 0), lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(k) + ',' + str(a)) 
                               for k,arcs in enumerate(new_arcs) for a in arcs if (k,a) not in self.x})

        # only dispatch arcs (we don't care about storage)
        all_arcs = set(a for k,arcs in enumerate(new_arcs) for a in arcs if a[0][0] != a[1][0] and a not in self.z)

        self.z.update({(i1,i2): self.model.addVar(obj=(self.network[i1[0]][i2[0]]['fixed_cost']), lb=0, ub=GRB.INFINITY, name='z' + str((i1,i2)), vtype=GRB.INTEGER)
                                for i1,i2 in all_arcs})

        self.model.update()

    # adds a new variable, inserts itself into appropriate constraints
    def update_constraints(self, new_arcs):
        chg_coeff = []

        # update constraints
        for k,arcs in enumerate(new_arcs):
            for a in arcs:
                # arc is a dispatch
                if a[0][0] != a[1][0]:
                    # consolidation
                    if a not in self.constraint_consolidation:
                        self.constraint_consolidation[a] = self.model.addConstr(self.x[k,a] * self.commodities[k]['q'] <= self.z[a] * self.network[a[0][0]][a[1][0]]['capacity'], 'cons' + str(a))
                    else:
                        chg_coeff.append((self.constraint_consolidation[a], self.x[k,a], self.commodities[k]['q']))

                    # cycle
                    if self.constraint_cycle != None:
                        if (k,a[0][0]) not in self.constraint_cycle:
                            self.constraint_cycle[(k,a[0][0])] = model.addConstr(self.x[k,a] <= 1, 'cycle')
                        else:
                            chg_coeff.append((self.constraint_cycle[(k,a[0][0])], self.x[k,a], 1))

                    # path length
                    #chg_coeff.append((self.constraint_path_length[k], self.x[k,a], self.transit(a[0][0],a[1][0])))


                # inflow
                if (k,a[1]) not in self.constraint_flow:
                    self.constraint_flow[(k,a[1])] = self.model.addConstr(self.x[k,a] == self.r(k,a[1]), 'flow' + str((k,a[1])))
                else:
                    chg_coeff.append((self.constraint_flow[(k,a[1])], self.x[k,a], 1))

                # outflow
                if (k,a[0]) not in self.constraint_flow:
                    self.constraint_flow[(k,a[0])] = self.model.addConstr(-self.x[k,a] == self.r(k,a[0]), 'flow' + str((k,a[0])))
                else:
                    chg_coeff.append((self.constraint_flow[(k,a[0])], self.x[k,a], -1))

        # minimize the number of updates
        self.model.update()

        for c,x,q in chg_coeff:
            self.model.chgCoeff(c, x, q)





    ## Estimate the path cost if taking 'cheapest' paths for each commodity, then trying to consolidate
    def intersection(self,i1,i2):
        return i1 != None and i2 != None and i2[0] <= i1[1] and i1[0] <= i2[1]

    def estimate_cost(self):
        # get cheapest path for each commodity
        p = map(min, self.path_cost())

        self.solution_paths = [a[1] for a in p]

        path = [list(pairwise(a[1])) for a in p]
        arcs = set(itertools.chain(*path))

        consolidations = {a: [k for k in range(len(self.commodities)) if a in path[k]] for a in arcs}
        dispatch_intervals = {(k,a): self.time_window(k,a) for a,v in consolidations.items() for k in v }   

        cons = {}

        # Partitions
        for a, K in consolidations.items():
            if len(K) <= 1:
                continue

            partitions = nx.Graph()
            partitions.add_nodes_from(K)

            for k1 in K:
                for k2 in K:
                    if any(self.intersection(dispatch_intervals[(k1,a)], dispatch_intervals[(k1,a)]) for a,v in consolidations.items() if k1 in v and k2 in v):
                        partitions.add_edge(k1,k2)

            for i,c in enumerate(nx.connected_components(partitions)):
                if len(c) > 1:
                    cons[((a[0],i),(a[1],i))] = c


        # try to consolidate 
        s = CheckSolution(self)
        s.validate([a[1] for a in p], cons)

        path_cost = sum([a[0] for a in p]) # cost without consolidation
        return (path_cost, s.get_solution_cost())       # cost with some consolidation

    def path_cost(self):
        # all feasible paths (directed path from o^k to d^k within time window)
        paths = [[p for p in self.all_simple_paths(self.network, c['a'][0], c['b'][0], c['b'][1] - c['a'][1])] 
                for k,c in enumerate(self.commodities)]

        # calculate cost of path
        return [[(sum((self.network[a[0]][a[1]]['var_cost'] * self.commodities[k]['q']) + self.network[a[0]][a[1]]['fixed_cost'] * math.ceil(self.commodities[k]['q'] / self.network[a[0]][a[1]]['capacity']) for a in pairwise(P)), P) 
                    for P in Pk]
                        for k,Pk in enumerate(paths)]

    # source taken from networkx - modified to cutoff at weighted length
    def all_simple_paths(self, G, source, target, cutoff=None):
        length = 0
        visited = [source]
        stack = [iter(G[source])]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                v = visited.pop()
                if visited:
                    length -= G[visited[-1]][v]['weight']
            else:
                t = G[visited[-1]][child]['weight']
                
                if length + t <= cutoff:
                    if child == target:
                        yield visited + [target]
                    elif child not in visited:
                        visited.append(child)
                        stack.append(iter(G[child]))
                        length += t

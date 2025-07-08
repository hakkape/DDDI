from gurobipy import *
import itertools
from CheckSolution import *
from math import log10, ceil
from collections import Counter
from DrawLaTeX import *
from tools import * 
import time
import networkx as nx
import logging

check_count = 0
solve_time = 0

logging.basicConfig(format='%(message)s',level=logging.DEBUG)
logger = logging.getLogger("IntervalSolver")

# add the handlers to logger
fh = logging.FileHandler('intervalsolver.log')
logger.addHandler(fh)

def enum(**enums):
    return type('Enum', (), enums)

preprocessing_option = enum(node='node', arc='arc')
shortest_path_option = enum(commodity='commodity', shared='shared', edges='edges')

MIP_GAP = 0.01
PRECISION = 2 # decimal places

## useful check for exploring solution graph
is_node = lambda n: type(n[1]) is not frozenset

def round_tuple(t):
    return tuple(map(lambda x: isinstance(x, float) and round(x, PRECISION) or x, t)) if isinstance(t,tuple) else t

class HIntervalSolver(object):
    """Solves time discretized service network design problems using an iterative approach"""
    __slots__ = ['problem', 'model', 'network', 'shared_shortest_paths', 'S', 'T', 'x', 'z', 'commodities', 'intervals', 'arcs', 'origin_destination', 'constraint_consolidation', 'constraint_flow' ,'constraint_path_length', 'solution_paths','consolidations', 'fixed_timepoints_model', 'timepoints', 'incumbent', 'lower_bound', 'shouldEnforceCycles', 'fixed_paths','timed_network','cons_network','suppress_output','GAP', 'incumbent_solution','all_paths', 'edge_shortest_path', 'status','timepoints_per_iteration', 'archive_timed_network', 'archive_cons_network']

    def __init__(self, problem, time_points=None, full_solve=True, fixed_paths=[], suppress_output=False, gap=MIP_GAP):
        self.problem = problem
        self.commodities = [{k: round_tuple(i) for k,i in c.items()} for c in problem.commodities]

        self.S = min(c['a'][1] for c in self.commodities)  # time horizon
        self.T = max(c['b'][1] for c in self.commodities) + 1  # time horizon

        self.incumbent = None  # store the lowest upper bound
        self.incumbent_solution = None
        self.lower_bound = None
        self.solution_paths = None
        self.fixed_paths = fixed_paths
        self.suppress_output = suppress_output
        self.GAP = gap
        self.status = 0

        # build graph
        self.network = nx.DiGraph()

        for a, destinations in problem.network.items():
            for b, transit_time in destinations.items():
                self.network.add_edge(a, b, weight=transit_time, capacity=problem.capacities.get((a,b), 1.0), fixed_cost=problem.fixed_cost.get((a,b), transit_time), var_cost=problem.var_cost.get((a,b), 0))

        self.create_shortest_paths(shortest_path_option.edges)

        # create initial intervals/arcs
        self.timepoints = set(self.trivial_network())
        self.fixed_timepoints_model = time_points != None and full_solve == False
        
        ##
        ## Construct Gurobi model
        ## 
        model = self.model = Model("IMCFCNF", env=Env(""))
        model.modelSense = GRB.MINIMIZE

        if self.fixed_timepoints_model == False:
            model.setParam('OutputFlag', False)

        model.setParam(GRB.param.MIPGap, gap)
        #model.setParam(GRB.param.TimeLimit, 14400) # 4hr limit
        #model.setParam(GRB.param.MIPFocus, 2)
        #model.setParam(GRB.param.PrePasses, 3)

        # set which interval contains the origin/destination
        self.origin_destination = {k: ((c['a'][0], self.S, self.T), (c['b'][0], self.S, self.T)) for k,c in enumerate(self.commodities)}
        self.intervals = tuplelist(((n, self.S, self.T) for n in self.network.nodes()))

        self.build_network()

        # archive network
        for G1,G2 in zip(self.timed_network,self.archive_timed_network):
            for e1,e2,d in G1.edges(data=True):
                if not G2.has_edge(e1,e2):
                    G2.add_edge(e1,e2,d)

        for e1,e2,d in self.cons_network.edges(data=True):
            if not self.archive_cons_network.has_edge(e1,e2):
                self.archive_cons_network.add_edge(e1,e2,d)

        ##
        ## Constraints
        ## 

        # flow constraints
        self.constraint_flow = {}

        for k,G in enumerate(self.timed_network):
            for n in self.intervals:
                i = [d['x'] for a1,a2,d in G.in_edges(n, data=True) if 'x' in d]
                o = [d['x'] for a1,a2,d in G.out_edges(n, data=True) if 'x' in d]

                if i or o:
                    self.constraint_flow[(k,n)] = self.model.addConstr(quicksum(i) - quicksum(o) == self.r(k,n), 'flow' + str((k,n)))

        # Consolidation constraints
        self.constraint_consolidation = {(a1,a2): self.model.addConstr(quicksum(self.timed_network[k][a1][a2]['x'] * self.commodities[k]['q'] for k in d['K']) <= d['z'] * self.network[a1[0]][a2[0]]['capacity'], 'cons' + str((a1,a2))) 
                                         for a1,a2,d in self.cons_network.edges(data=True) if d['z'] != None }

        # add timepoints
        self.model.update()
        
        #if time_points != None:
        #    self.timepoints.update(time_points)
        #    self.add_network_timepoints(time_points)


    def build_network(self):
        self.timed_network = []
        self.archive_timed_network = []
        all_arcs = []

        for k in range(len(self.commodities)):
            G = nx.DiGraph()

            # Add node-intervals
            G.add_nodes_from((n, self.S, self.T) for n in self.network.nodes())

            # Create timed arcs
            origin,dest = self.commodities[k]['a'], self.commodities[k]['b']
            v = functools.partial(self.V, k)
            #v = lambda a: self.V(k,a) and (a[0][0],a[1][0]) in self.all_paths[k]

            all_arcs.append(filter(v, (((e[0], self.S, self.T), (e[1], self.S, self.T)) 
                                    for e in (self.network.edges() if not self.fixed_paths else self.fixed_paths[k]))))

            G.add_edges_from(((a[0], a[1], {'x': self.model.addVar(obj=(self.network[a[0][0]][a[1][0]]['var_cost'] * self.commodities[k]['q'] if a[0][0] != a[1][0] else 0), lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(k) + ',' + str(a)) })
                                for a in all_arcs[k]))
            self.timed_network.append(G)
            self.archive_timed_network.append(nx.DiGraph())

        # create consolidation network
        self.cons_network = nx.DiGraph()
        self.cons_network.add_nodes_from((n, self.S, self.T) for n in self.network.nodes())

        cons = itertools.groupby(sorted((a,k) for k,arcs in enumerate(all_arcs) for a in arcs), itemgetter(0))
        self.cons_network.add_edges_from(((a[0],a[1],{'z': self.model.addVar(obj=(self.network[a[0][0]][a[1][0]]['fixed_cost']), lb=0, ub=GRB.INFINITY, name='z' + str(a), vtype=GRB.INTEGER), 
                                                      'K': set(map(itemgetter(1),K))}) 
                                          for a,K in cons))

        self.archive_cons_network = nx.DiGraph()
        self.model.update()


    def r(self,k,i):
        # at origin
        if self.origin_destination[k][0] == i:
            return -1

        # at destination
        elif self.origin_destination[k][1] == i:
            return 1

        return 0

    def infeasible(self):
        if self.fixed_paths:
            return len([c for k,c in enumerate(self.commodities) if c['a'][1] + sum(self.transit(*a) for a in self.fixed_paths[k]) > c['b'][1]]) > 0
        else:
            return len([c for k,c in enumerate(self.commodities) if c['a'][1] + self.shortest_path(k,c['a'][0],c['b'][0]) > c['b'][1]]) > 0


    # walk up the tree from node t, and return all root nodes that have maximum early times
    def walks(self, G, t, top=False):
        roots = set()
        stack = [t]

        # 'recursive' walk up tree
        while stack:
            v = stack.pop()
            edges = [(0,v)]

            while edges:  # choose latest edge with largest commodity and latest time window
                v = edges[0][1]
                edges = sorted([((G.node[e[0]]['early'], self.commodities[e[0][0]]['q'] if is_node(e[0]) else 0, G.node[e[0]]['tw']), e[0]) for e in G.in_edges(v)], reverse=True)

                # for consolidations, choose all commodities that have the max early time (add to stack)
                if not top and not is_node(v):
                    k,group = next(itertools.groupby(edges, lambda x: x[0][0]), (None,[]))
                    stack.extend(map(itemgetter(1), (g for g in group if g != edges[0])))

            roots.add(v)

        return roots

    ##
    ## Iterative Solve
    ##
    def solve(self, draw_filename='', start_time=time.time(), write_filename=''):
        global solve_time
        logger = logging.getLogger("IntervalSolver")

        # feasibility check: shortest path cannot reach destination - gurobi doesn't pick it up, because no arcs exist
        if self.infeasible():
            if not self.suppress_output:
                print "INFEASIBLE"
            return 0, 0, 0

        variable_gap = True
        iterations = -1
        new_timepoints = set(self.timepoints)  # not required, but might be good for testing
        len_timepoints = 0      # used to halt processing if haven't added unique timepoints in a while (hack - this shouldn't happen if my code was good)
        it_timepoints = 0
        self.timepoints_per_iteration = [(0, n,t) for n,t in self.timepoints]

        s = CheckSolution(self)
        solve_time = 0

        # output statistics
        logger.info('{0:>3}, {1:>10}, {2:>10}, {3:>7}, {4:>6}, {5:>6}, {6}'.format(*'##,LB,UB,Gap,Time,Solver,Type [TP]'.split(',')))

        while True:
            iterations += 1

            if write_filename != '':
                self.model.update()
                self.model.write(write_filename + "_" + str(iterations) + ".lp")  # debug the model

            t0 = time.time()
            self.solve_lower_bound()  # Solve lower bound problem
            solve_time += time.time() - t0

            # return if infeasible, time out or interupted
            if self.status != GRB.status.OPTIMAL:
                if self.status in [GRB.status.TIME_LIMIT, GRB.status.INTERRUPTED]:
                    ### output statistics
                    self.solution_paths = self.get_solution_paths()
                    self.consolidations = self.get_consolidations()
                    solution, cycle = self.get_network_solution()

                    # if valid path length (for LP)
                    if not filter(lambda tw: tw[0] > tw[1], [solution.node[(k,self.commodities[k]['b'][0])]['tw'] for k in K]):
                        t0 = time.time()
                        if s.validate(self.solution_paths, self.consolidations):
                            solve_time += time.time() - t0

                            solution_cost = s.get_solution_cost()
                            self.incumbent = min(self.incumbent, solution_cost) if self.incumbent != None else solution_cost
                            logger.info("{0:>3}, {1:10.1f}, {2:10.1f}, {3:7.2%}, {4:6.2f}, {5:6.2f}, ABORTED".format(iterations, self.lower_bound, self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time))
                            logger.info('\nit: {0}, new tp: {1}, intervals: {2}, vars: {3}, time: {4:.2f} ({5:.2f})\n'.format(iterations, len(new_timepoints), len(self.intervals), len(self.model.getVars()), time.time()-start_time, solve_time))

                elif not self.suppress_output:
                    logger.error("INFEASIBLE")
                return iterations, len(new_timepoints), solve_time

            self.solution_paths = self.get_solution_paths()
            self.consolidations = self.get_consolidations()

            # return if we are only doing one iteration
            if self.fixed_timepoints_model == True:
                logger.info("{0:>3}, {1:10.1f}, ".format(iterations, self.lower_bound))
                return iterations, len(new_timepoints), solve_time

            solution, cycle = self.get_network_solution()

            # draw the timepoints if required
            portrait = True

            if draw_filename != '':
                scale = 2
                pdf = DrawLaTeX(self.commodities, self.network)
                pdf.draw_network(scale, position=self.problem.position, font='normal')
                pdf.draw_commodities(scale, position=self.problem.position)
 
                pdf.draw_solution_network(solution, cycle)

                #data = [{'k': k, 
                #         'Node': n[0],
                #         'R1': self.shortest_path(k, c['a'][0], n[0]) < sum([self.transit(n1,n2) for n1,n2 in pairwise(self.solution_paths[k][:i+1])]),  
                #         'R4': sum([self.transit(n1,n2) for n1,n2 in pairwise(self.solution_paths[k][:i+1])]) + self.shortest_path(k, n[1], c['b'][0]) <= c['b'][1]-c['a'][1],  
                #         'TW': self.time_window(k, n), 
                #         'SP': (c['a'][1] + self.shortest_path(k,c['a'][0], n[0]), c['b'][1] - self.shortest_path(k, n[0],c['b'][0])), 
                #         r'$e^k+\sum_{i=1}^{n-1} \tau^k_{\left(n_i,n_{i+1}\right)}$': c['a'][1] + sum([self.transit(n1,n2) for n1,n2 in pairwise(self.solution_paths[k][:i+1])])}
                #          for k,c in enumerate(self.commodities) 
                #            for i,n in enumerate(pairwise(self.solution_paths[k]))]
                #pdf.draw_table(r"k;Node;R1;R4;TW;SP;$e^k+\sum_{i=1}^{n-1} \tau^k_{\left(n_i,n_{i+1}\right)}$".split(';'), data)
 
                #pdf.draw_table(r"k;$o^k\rightarrow d^k$;$[e^k, l^k]$".split(';'), 
                #               [{'k': k, r'$o^k\rightarrow d^k$': (c['a'][0], c['b'][0]), '$[e^k, l^k]$': (c['a'][1], c['b'][1])} for k,c in enumerate(self.commodities)])              

                arcs = [G.edges() for G in self.timed_network]

                pdf.draw_timeline(self.intervals, arcs, 24 if not portrait else 15, self.get_solution())
                pdf.draw_latex(portrait)
                pdf.save(draw_filename + str(iterations))


            # Checks gap of a lower bound / upper bound solutions - 1% gap               
            if self.incumbent != None and (self.incumbent - self.lower_bound) < self.incumbent*self.GAP:
                # output statistics
                output = "{0:>3}, {1:10.1f}, ".format(iterations, self.lower_bound)
                output += "{0:10.1f}, {1:7.2%}, {2:6.2f}, {3:6.2f},".format(self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time)

                logger.info(output + " Optimal Solution")
                logger.info('\nit: {0}, new tp: {1}, intervals: {2}, vars: {3}, time: {4:.2f} ({5:.2f})\n'.format(iterations, len(new_timepoints), len(self.intervals), len(self.model.getVars()), time.time()-start_time, solve_time))

                self.timepoints.update(new_timepoints)
                break


            ## Add time points if needed
            path_length_timepoints = set()
            window_timepoints = set()
            cycle_timepoints = set()
            mutual_timepoints = set()

            ## Treat subgraphs separately
            for G in nx.weakly_connected_component_subgraphs(solution, False):
                K = set(n[0] for n in G if type(n[1]) is not frozenset)

                path_length_K = []  # keep track of commodities that fail path length

                ####
                ####
                ## which commodities fail to reach destination in time, and aren't connected to a path/window issue?
                #tails = [last for last in ((k,self.solution_paths[k][-1]) for k in K) 
                #            if 'valid' in solution.node[last] and not solution.node[last]['valid']]

                #for t in tails:
                #    for r in self.walks(G, t, True):
                #        if r[0] == t[0]:
                #            path_length_K.append(r[0])
                #            path_length_timepoints.update(self.get_network_timepoints(solution, G, r, t))

                #####
                ####


                ### 1. Path length is greater than time window  (choose latest node before destination and actual time it would dispatch - iterates if already in time points)
                for k in K:
                    tw = solution.node[(k,self.commodities[k]['b'][0])]['tw']

                    if tw[0] > tw[1]:
                        c = self.commodities[k]

                        tp = next(((a1[1], c['a'][1] + self.shortest_path(k, c['a'][0], a1[0]) + self.transit(*a1)) 
                                    for a1,a2 in pairwise(pairwise(self.solution_paths[k])) 
                                        if c['a'][1] + self.shortest_path(k, c['a'][0], a1[0]) + self.transit(*a1) > c['b'][1] - self.shortest_path(k, a2[1], c['b'][0]) - self.transit(*a2)), None)
                        
                        if tp != None:
                            path_length_timepoints.add(tp)
                            path_length_K.append(k)
                        else:
                            path_length_timepoints.add(self.commodities[k]['a'])  # dummy timepoint to avoid running LP

                #        else:
                #            jd = next((i+1 for i,(n1,n2) in enumerate(pairwise(self.solution_paths[k])) if solution.node[(k,n2)]['tw'][0] + self.shortest_path(k,n2,c['b'][0]) > c['b'][1]), len(self.solution_paths[k]))
                #            jo = min(jd, next((i+1 for i,(n1,n2) in enumerate(pairwise(self.solution_paths[k])) if self.shortest_path(k,c['a'][0],n2) < solution.node[(k,n2)]['tw'][0]), jd)) 

                #            path_length_timepoints.update(set((n,solution.node[(k,n)]['tw'][0]) for n in self.solution_paths[k][jo:jd+1]))


                ## 2. Simple time window
                window = []
                tmp_window = []

                # for each commodity, find first broken TW consolidation - ignoring consolidations upstream 
                for k in K:
                    n = next((n for n in nx.dfs_preorder_nodes(solution, (k,self.solution_paths[k][0])) if not is_node(n) and solution.node[n]['tw'][1] < solution.node[n]['tw'][0]), None)

                    # remove downstream consolidations, ignoring any path length violations
                    if n != None and not any(nx.has_path(solution, i, n) for i in tmp_window) and not any(nx.has_path(solution, (i,self.solution_paths[i][0]), n) for i in path_length_K):
                        tmp_window = [n] + [i for i in tmp_window if not nx.has_path(solution, n, i)]

                # set of broken TW consolidations
                for n in tmp_window:
                    # earliest time for k2 to reach n2 is greater than latest time k1 can leave n2
                    broken_cons = sorted(((k1, n[0][1], self.commodities[k2]['a'][1] + self.edge_shortest_path[self.commodities[k2]['b'][0],self.commodities[k2]['a'][0]][n[0][0]] + self.transit(*n[0])) 
                                            for k1,k2 in itertools.product(n[1],n[1]) 
                                                if k1 != k2 and self.commodities[k2]['a'][1] + self.edge_shortest_path[self.commodities[k2]['b'][0],self.commodities[k2]['a'][0]][n[0][0]] + self.transit(*n[0]) > self.commodities[k1]['b'][1] - self.edge_shortest_path[n[0]][self.commodities[k1]['b'][0]]))

                    if broken_cons:
                        # add earliest time point for each k
                        #window_timepoints.add(broken_cons[0][1:])   
                        window_timepoints.update(next(g)[1:] for k,g in itertools.groupby(broken_cons, itemgetter(0)))   
                        window.append(n)
                #    else:
                #        time_windows = sorted((solution.node[(k,n[0][0])]['tw'], k) for k in n[1])
                #        t1,k1 = time_windows[0]

                #        # get first broken pair
                #        for t in time_windows[1:]:
                #            if t1[1] > t[0][1]:
                #                t1,k1 = t

                #            if t[0][0] > t1[1]:
                #                t2,k2 = t
                #                break

                #        ## force 'before' commodities to not travel with 'after commodities
                #        c1,c2 = self.commodities[k1], self.commodities[k2]

                #        dt = abs(t2[0] - t1[1])

                #        j1 = self.solution_paths[k1].index(n[0][0])
                #        j2 = self.solution_paths[k2].index(n[0][0])

                #        #window_timepoints.update([(x,solution.node[(k1,x)]['tw'][1]+dt) for x in self.solution_paths[k1][j1:]])   # start arc till dest
                #        #window_timepoints.update([(x,solution.node[(k2,x)]['tw'][0]) for x in self.solution_paths[k2][:j2+2]]) # origin to end arc

                #        for x in reversed(list(pairwise(self.solution_paths[k2][:j2+2]))):
                #            window_timepoints.add((x[0], solution.node[(k2,x[0])]['tw'][0]))

                #            if c2['a'][1] + self.edge_shortest_path[c2['b'][0],c2['a'][0]][x[0]] >= solution.node[(k2,x[1])]['tw'][0]:
                #                break

                #        for x in pairwise(self.solution_paths[k1][j1:]):
                #            window_timepoints.add((x[0], solution.node[(k1,x[0])]['tw'][1] + dt))

                #            if solution.node[(k1,x[0])]['tw'][1] + dt + self.transit(*x) + self.edge_shortest_path[x][c1['b'][0]] > c1['b'][1]:
                #                break


                ## 3. Check for cycle
                for c in cycle:
                    # get all commodities explicitly in cycle (ignore incidental commodities)
                    cycle_K = set(n[0] for n in c if is_node(n))

                    # ignore any node that is already connected to an invalid path or TW consolidation
                    if  not (cycle_K & K) or any(nx.has_path(solution, n, c[0]) for n in window) or any(nx.has_path(solution, (k, self.solution_paths[k][0]), c[0]) for k in path_length_K):
                        continue
                    #if  not (cycle_K & K):
                    #    continue

                    # want to start at earliest point in cycle, and rewrite cycle to match
                    tw, start_node,i = sorted([(solution.node[n]['tw'], n[1], i) for i,n in enumerate(c) if is_node(n)])[0]
                    t = tw[0]

                    ####
                    ####

                    # add timepoints from origin to cycle start
                    for k in cycle_K:
                        origin = self.commodities[k]['a']
                        enter = sorted([x for x in c if nx.has_path(G, (k,origin[0]), x)], key=lambda x: nx.shortest_path_length(G, (k,origin[0]), x))[0]

                        for p in pairwise(nx.shortest_path(G, (k,origin[0]), enter)):
                            #if not is_node(p[1]):
                            #    continue

                            #e = (p[0][1] if is_node(p[0]) else p[0][0][0], p[1][1])
                            #if G.node[p[1]]['tw'][0] <= origin[1] + self.shortest_path(k, *e):
                            #    continue

                            cycle_timepoints.add((p[1][1] if is_node(p[1]) else p[1][0][0], G.node[p[1]]['tw'][0]))
                    ####
                    ####

                    for n in itertools.cycle(pairwise(map(operator.itemgetter(1), filter(is_node, c[i:] + c[:i])) + [start_node])):
                        cycle_timepoints.add((n[0], t))

                        # stop if past time horizon or if one commodity has reached it's end
                        if t > self.T or [k for k in cycle_K if t + self.shortest_path(k, n[0], self.commodities[k]['b'][0]) > self.commodities[k]['b'][1]]:
                            break

                        t += self.transit(*n)


                # 4. Mutual consolidations
                # return the objective value cost for a given subnetwork
                def network_cost(N):
                    cost = 0
                    
                    for n1,n2 in N.edges():
                        if is_node(n1) and not is_node(n2):
                            continue

                        if is_node(n1):
                            cost += self.network[n1[1]][n2[1]]['var_cost'] * self.commodities[n1[0]]['q'] + self.network[n1[1]][n2[1]]['fixed_cost'] * math.ceil(self.commodities[n1[0]]['q'] / self.network[n1[1]][n2[1]]['capacity'])
                        else:
                            q = sum(self.commodities[k]['q'] for k in n1[1])
                            cost += self.network[n1[0][0]][n1[0][1]]['var_cost'] * q + self.network[n1[0][0]][n1[0][1]]['fixed_cost'] * math.ceil(q / self.network[n1[0][0]][n1[0][1]]['capacity'])
            
                    return cost

                # return the average price per node along the path from r to t
                def path_cost(G,r,t):
                    nodes = set(itertools.chain(*nx.all_shortest_paths(G,r,t)))
                    return network_cost(nx.subgraph(G, nodes)) / len(nodes)

                ## which commodities fail to reach destination in time, and aren't connected to a path/window issue?
                tails = [last for last in ((k,self.solution_paths[k][-1]) for k in K) 
                            if 'valid' in solution.node[last] and not solution.node[last]['valid'] and 
                                not any(nx.has_path(solution, w, last) for w in window) and 
                                not any(nx.has_path(solution, (kp, self.solution_paths[kp][0]), last) for kp in path_length_K)]

                #tails = [last for last in ((k,self.solution_paths[k][-1]) for k in K) 
                #            if 'valid' in solution.node[last] and not solution.node[last]['valid']]

                for t in tails:
                    for r in self.walks(G, t, True):
                        mutual_timepoints.update(self.get_network_timepoints(solution, G, r, t))

                ## for better performance add some extra timepoints for issues at the root nodes (choosing the tail with most expensive path per node)
                #roots = [r for r in ((k,self.solution_paths[k][0]) for k in K) 
                #                if 'valid' in solution.node[r] and not solution.node[r]['valid']]

                #for r in roots:
                #    for _,tmp in sorted([(-path_cost(G,r,t),t) for t in tails if nx.has_path(G, r, t)]):
                #        mutual_timepoints.update(self.get_network_timepoints(solution, G, r, tmp))
                #        break # only add first one



            tp = path_length_timepoints | window_timepoints | cycle_timepoints | mutual_timepoints

            # output statistics
            output = "{0:>3}, {1:10.1f}, ".format(iterations, self.lower_bound)

            # if invalid path, skip the LP (since it's infeasible)
            if path_length_timepoints:
                output += "{0:>10}, {0:>7}, {1:6.2f}, {2:6.2f}, P[{3}]".format('-', time.time()-start_time, solve_time, len(path_length_timepoints))
            else:
                # Solve UB LP, check solution costs
                t0 = time.time()
                if s.validate(self.solution_paths, self.consolidations):
                    solve_time += time.time() - t0

                    solution_cost = s.get_solution_cost()

                    if self.incumbent == None or solution_cost < self.incumbent:
                        self.incumbent = solution_cost
                        self.incumbent_solution = (self.solution_paths, s.get_consolidations())


                    output += "{0:10.1f}, {1:7.2%}, {2:6.2f}, {3:6.2f},".format(self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time)

                    # Checks gap of a lower bound / upper bound solutions - 1% gap               
                    if (self.incumbent - self.lower_bound) < self.incumbent*self.GAP:
                        logger.info(output + " Optimal Solution")
                        logger.info('\nit: {0}, new tp: {1}, intervals: {2}, vars: {3}, time: {4:.2f} ({5:.2f})\n'.format(iterations, len(new_timepoints), len(self.intervals), len(self.model.getVars()), time.time()-start_time, solve_time))

                        self.timepoints.update(new_timepoints)
                        break

                    # variable gap for faster solve (turn off if no new time points found)
                    #if not tp or variable_gap and (self.incumbent - self.lower_bound) < self.incumbent * self.GAP * 3:
                    #    self.model.setParam(GRB.param.MIPGap, self.GAP*0.95)
                    #    #self.model.setParam(GRB.param.MIPFocus, 2) # focus on proving optimality
                    #    variable_gap = False
                    #elif variable_gap:
                    #    self.model.setParam(GRB.param.MIPGap, ((self.incumbent - self.lower_bound)/self.incumbent * 0.35))

            # Stop endless loop - hack for my bad code
            if len(new_timepoints) == len_timepoints:
                it_timepoints += 1
            else:
                len_timepoints = len(new_timepoints)
                it_timepoints = 0

            if it_timepoints > 5: # arbitrary number
                for i,G in enumerate(nx.weakly_connected_component_subgraphs(solution, False)):
                    if any(True for n in G if not G.node[n]['valid']):
                        pdf = DrawLaTeX(self.commodities, self.network)
                        pdf.draw_solution_network(G, cycle)
                        #pdf.draw_timeline(self.intervals, self.arcs, 24, self.get_solution(), [0,2,3])
                        pdf.draw_latex(False)
                        pdf.save("endless_" + str(i))

                output += " Endless Loop\n"
                logger.error(output)
                sys.exit()
                break


            if window_timepoints:
                output += " W[{0}]".format(len(window_timepoints-new_timepoints))

            if cycle_timepoints:
                output += " C[{0}]".format(len(cycle_timepoints-new_timepoints))

            if mutual_timepoints:
                output += " M[{0}]".format(len(mutual_timepoints-new_timepoints))

            if not self.suppress_output:
                logger.info(output)

            self.add_network_timepoints(tp)
            new_timepoints.update(tp)
            self.timepoints_per_iteration.extend((iterations+1, n,t) for n,t in tp)

        return iterations, len(new_timepoints), solve_time

    def write_timepoints(self, filename):
        with open(filename, "w") as file:
            file.write("data = [");
            for i,n,t in self.timepoints_per_iteration:
                file.write("{{i:{0},n:{1},t:{2}}},".format(i,n,int(t)))
            file.write("];")

    # solve the model
    def solve_lower_bound(self):
        global check_count
        check_count = time.time()

        def callback(model, where):
            global check_count
            global solve_time

            if where == GRB.callback.MIP:
                objbst = model.cbGet(GRB.callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.callback.MIP_OBJBND)
                self.lower_bound = max(objbnd, self.lower_bound) if self.lower_bound != None else objbnd

                if self.incumbent != None and (self.incumbent - self.lower_bound) < self.incumbent * self.GAP:
                    self.model.terminate()

                if self.incumbent != None:
                    sys.stdout.write('{0:7.2%}, {1:7.2%} {2:7.1f}s\r'.format((objbst-objbnd)/objbst, (self.incumbent - self.lower_bound) / self.incumbent, solve_time + time.time() - check_count))
                else:
                    sys.stdout.write('{0:7.2%}\r'.format((objbst-objbnd)/objbst))
                sys.stdout.flush()

            #elif where == GRB.callback.MIPSOL:
            #    objbst = model.cbGet(GRB.callback.MIPSOL_OBJBST)
            #    objbnd = model.cbGet(GRB.callback.MIPSOL_OBJBND)
            #    self.lower_bound = max(objbnd, self.lower_bound) if self.lower_bound != None else objbnd

            #    # gap < GAP
            #    if self.incumbent != None and objbst - objbnd < objbst * self.GAP * 2: 
            #        try:
            #            s = CheckSolution(self)
            #            check_count += 1

            #            if s.validate(*self.get_inprogress()):
            #                solution_cost = s.get_solution_cost()

            #                if self.incumbent == None or solution_cost < self.incumbent:
            #                    self.incumbent = solution_cost
            #                    self.incumbent_solution = (s.solution_paths, s.get_consolidations())

            #                    sys.stdout.write('{0:7.2%}, {1:7.2%} {2}\r'.format((objbst-objbnd)/objbst, (self.incumbent - self.lower_bound) / self.incumbent, check_count))
            #                    sys.stdout.flush()
            #        except (KeyboardInterrupt, SystemExit):
            #            self.model.terminate()

            #        if (self.incumbent - self.lower_bound) < self.incumbent * self.GAP:
            #            self.model.terminate()

        self.model.update()
        #self.model.write('test.lp')
        if self.suppress_output:
            self.model.optimize()
        else:
            self.model.optimize(callback)

        self.lower_bound = max(self.model.objBound, self.lower_bound) if self.lower_bound != None else self.model.objBound
        self.status = GRB.status.OPTIMAL if self.model.status == GRB.status.INTERRUPTED and (self.incumbent - self.lower_bound) < self.incumbent * self.GAP else self.model.status


    def get_network_timepoints(self, solution, G, r, t):
        cr = self.commodities[r[0]]
        ct = self.commodities[t[0]]
        x = cr['a'][0]
        has_k = False
        next_tk = None
        tp = set()

        path = list(nx.shortest_path(G, r, t))

        first = next(v2 for v1,v2 in pairwise(path) if t[0] in (is_node(v1) and [v1[0]] or v1[1]))
        last = next(v1 for v1,v2 in pairwise(reversed(path)) if r[0] in (is_node(v2) and [v2[0]] or v2[1]))

        # skip nodes that can be reached by shortest paths
        def canDrop(v):
            has_k,x = (r[0] == v[0], v[1]) if is_node(v) else (r[0] in v[1], v[0][0])
            #return has_k and cr['a'][1] + self.edge_shortest_path[cr['b'][0], cr['a'][0]][x] >= solution.node[v]['early']
            return v != last and v != first and cr['a'][1] + self.edge_shortest_path[cr['b'][0], cr['a'][0]][x] >= solution.node[v]['early']
                    
        for n,n2 in pairwise(itertools.dropwhile(canDrop, path)):
            if not has_k and not is_node(n) and t[0] in n[1]:      # only add another point if aending at first consolidation
                next_tk = n

            if solution.node[n]['early'] > self.T:
                break

            has_k,x = (n[0] == t[0], n[1]) if is_node(n) else (t[0] in n[1], n[0][0])
            tp.add((x, solution.node[n]['early']))
        
            # finish if can reach end of commodity using shortest paths
            x2 = n2[1] if is_node(n2) else n2[0][1]

            if has_k and solution.node[t[0],x2]['early'] + self.edge_shortest_path[x,x2][ct['b'][0]] > ct['b'][1]:
                if next_tk == n and solution.node[n2]['early'] < self.T:
                    tp.add((x2, solution.node[n2]['early']))

                return tp

        if solution.node[t]['early'] < self.T:
            tp.add((t[1], solution.node[t]['early']))

        return tp

    # print solution
    def writeSolution(self, file):
        if self.incumbent_solution != None:
            self.problem.save(file, (self.incumbent, self.incumbent_solution[0], self.incumbent_solution[1]))

    def drawSolution(self, file, incumbent=True):
        pdf = DrawLaTeX(self.commodities, self.network)
        #pdf.draw_network(scale, position=self.problem.position, font='normal')
        #pdf.draw_commodities(scale, position=self.problem.position)

        solution, cycle = self.get_network_solution(*(self.incumbent_solution if incumbent else (None,None))) 
        pdf.draw_solution_network(solution, cycle)

        #arcs = [G.edges() for G in self.timed_network]
        #pdf.draw_timeline(self.intervals, arcs, 24 if not portrait else 15, self.get_solution())
        pdf.draw_latex()
        pdf.save(file)

    def printSolution(self):
        if self.incumbent_solution != None:
            print "cost={0}".format(self.incumbent)

            print "PATHS,{0}".format(len(self.incumbent_solution[0]))

            for k,p in enumerate(self.incumbent_solution[0]):
                print "{0},[{1}]".format(k,",".join(map(lambda x: str(x+1),p)))

            print "CONSOLIDATIONS,{0}".format(len(self.incumbent_solution[1]))

            for (n1,n2),K in self.incumbent_solution[1]:
                print "{0},{1},[{2}]".format(n1+1,n2+1,",".join(map(str,K)))

        else:
            print('No solution')

    # get statistics
    def get_statistics(self):
        stats = {}
        stats['cost'] = self.model.objVal if self.status == GRB.status.OPTIMAL and self.incumbent == None else self.incumbent
        stats['nodes'] = len(self.network.nodes())
        stats['intervals'] = len(self.intervals)
        stats['arcs'] = sum(len(G.edges()) for G in self.timed_network)
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
        return tuplelist((k,(a1,a2)) for k,G in enumerate(self.timed_network) for a1,a2,d in G.edges(data=True) if 'x' in d and d['x'].x > 0) if self.status in [GRB.status.OPTIMAL, GRB.status.INTERRUPTED, GRB.status.TIME_LIMIT] else None

    ##
    ## From the solution, return the path chosen for each commodity
    ##
    def get_solution_paths(self):
        result = []
        
        for k,G in enumerate(self.archive_timed_network):
            n = self.commodities[k]['a'][0]
            path = [n]

            while n != self.commodities[k]['b'][0]:
                for _,i2,d in G.out_edges((n,self.S,self.T), data=True):
                    if 'x' in d and d['x'].x > 0.5:
                        n = i2[0]
                        break

                path.append(n)
            result.append(path)

        return result

    ##
    ## From the solution, return the commodities that are consolidated together
    ##
    def get_consolidations(self, path=None):
        # ignore cycles that are not apart of actual path - can occur when x has 0 cost
        if path == None:
            path = self.solution_paths

        return {(a1,a2): [k for k in z['K'] if self.timed_network[k][a1][a2]['x'].x > 0 and a1[0] in path[k]] 
                    for a1,a2,z in self.cons_network.edges(data=True) if z['z'] != None and z['z'].x > 0} if self.status in [GRB.status.OPTIMAL, GRB.status.INTERRUPTED, GRB.status.TIME_LIMIT] else None

    def get_inprogress(self):
        x = [(k,(a1,a2),d['x']) for k,G in enumerate(self.timed_network) for a1,a2,d in G.edges(data=True) if 'x' in d]
        z = [(a1,a2,z['z'],z['K']) for a1,a2,z in self.cons_network.edges(data=True) if z['z'] != None]

        vars = self.model.cbGetSolution(map(itemgetter(2), x + z))

        paths = []
        arcs = dict(((k,a[0]),a[1]) for i,(k,a,_) in enumerate(x) if round(vars[i]) > 0)

        for k, (origin,dest) in self.origin_destination.items():
            i = origin
            paths.append([i[0]])
                
            while i[0] != dest[0]:
                i = arcs[(k,i)]
                
                if i[0] != paths[k][-1]:
                    paths[k].append(i[0])

        cons = {(a1,a2): [k for k in K if (k,a1) in arcs and a2 in arcs[k,a1]] 
                    for i,(a1,a2,_,K) in enumerate(z) if round(vars[len(x) + i]) > 0}

        return paths,cons



    ##
    ## Builds a graph from the solution (paths & consolidations)
    ##
    def get_network_solution(self, paths=None, cons=None):
        solution = nx.DiGraph()

        if paths == None:
            paths = self.solution_paths

        if cons == None:
            cons = [((c[0][0],c[1][0]), frozenset(k)) for c,k in self.consolidations.items() if len(k) > 1]

        CD = dict((k,map(itemgetter(1), g)) for k,g in itertools.groupby(sorted(cons), itemgetter(0)))

        solution.add_nodes_from(c for c in cons)
        solution.add_nodes_from((k,n) for k,P in enumerate(paths) for n in P)

        solution.add_weighted_edges_from(((k,a[0]), (k,a[1]), self.transit(*a)) 
                                            for k,P in enumerate(paths) for a in pairwise(P) if not (a in CD and any(k in K for K in CD[a])))

        solution.add_edges_from(((k,c[0][0]), c) for c in cons for k in c[1])
        solution.add_weighted_edges_from((c, (k,c[0][1]), self.transit(*c[0])) for c in cons for k in c[1])

        # Time windows
        tw = map(lambda k: self.time_windows(k,paths[k]), range(len(self.commodities)))

        for n in solution:
            if not is_node(n):
                solution.node[n]['tw'] = (max(tw[k][n[0][0]][0] for k in n[1]), min(tw[k][n[0][0]][1] for k in n[1]))
            else:
                solution.node[n]['tw'] = tw[n[0]][n[1]]

        cycle = []
        
        # update early/late
        for G in nx.weakly_connected_component_subgraphs(solution, False):
            has_cycle = False

            for n in nx.dfs_postorder_nodes(G):
                coll = solution.out_edges(n)

                if len(coll) > 1:
                    # check for cycles
                    p = next((p[1] for p in coll if 'late' not in solution.node[p[1]]), None)
                    if p:
                        cycle.append(nx.shortest_path(solution, p, n))
                        has_cycle = True
                        break 

                    solution.node[n]['late'] = round(min(solution.node[p[1]]['late'] for p in coll) - self.transit(*n[0]), PRECISION)
                elif len(coll) == 1:
                    p = coll[0][1]

                    # check for cycles
                    if 'late' not in solution.node[p]:
                        cycle.append(nx.shortest_path(solution, p, n))
                        has_cycle = True
                        break 

                    solution.node[n]['late'] = round(solution.node[p]['late'] - (self.transit(n[1], p[1]) if is_node(p) else 0), PRECISION)
                else: # at end
                    solution.node[n]['late'] = self.commodities[n[0]]['b'][1]

            if not has_cycle:
                ## copies graph to do reverse... can be done better?
                for n in nx.dfs_postorder_nodes(G.reverse()):
                    coll = solution.in_edges(n)

                    if len(coll) > 1:
                        solution.node[n]['early'] = round(max(solution.node[p[0]]['early'] for p in coll), PRECISION)
                        solution.node[n]['diff'] = round(max(solution.node[p[0]]['early'] for p in coll) - min(solution.node[p[0]]['early'] for p in coll), PRECISION)
                    elif len(coll) == 1:
                        p = coll[0][0]
                        solution.node[n]['early'] = round(solution.node[p]['early'] + self.transit(p[1] if is_node(p) else p[0][0], n[1]), PRECISION)
                    else: # at end
                        solution.node[n]['early'] = self.commodities[n[0]]['a'][1]

                    solution.node[n]['valid'] = solution.node[n]['early'] <= solution.node[n]['late']

        return solution, cycle


    # returns the min/max time window for a commodity that travels along arc a
    def time_windows(self, k, path=None):
        t1 = self.commodities[k]['a'][1]
        t2 = self.commodities[k]['b'][1]

        paths = list(pairwise(self.solution_paths[k] if path == None else path))

        early = [0] + list(accumulate((self.transit(*n) for n in paths)))
        late  = [0] + list(accumulate((self.transit(*n) for n in reversed(paths))))

        return {p: round_tuple((float(t1 + e), float(t2 - l))) for (p,e,l) in zip(self.solution_paths[k], early, reversed(late))}


    ## get the transit time between 2 nodes (looks nicer than direct access)
    def transit(self,n1,n2):
        return self.network[n1][n2]['weight']

    def shortest_path(self,k,n1,n2):
        return self.shared_shortest_paths[n1].get(n2, None)

    def shortest_paths(self,k,n1):
        return self.shared_shortest_paths[n1]

    def create_shortest_paths(self, opt=shortest_path_option.shared):
        self.shared_shortest_paths = nx.shortest_path_length(self.network, weight='weight')
        self.edge_shortest_path = {}

        # create shortest paths excluding n1 node on arc
        if opt == shortest_path_option.edges:
            for n in self.network:
                network_copy = self.network.copy()
                network_copy.remove_node(n)

                #for e in self.network.out_edges(n):
                for n2 in self.network:
                    if n != n2:
                        self.edge_shortest_path[(n,n2)] = nx.shortest_path_length(network_copy, n2, weight='weight')


    ## creates arcs/nodes for time horizon
    def trivial_network(self):
        return tuplelist((n,t) for n in self.network.nodes() for t in [self.S,self.T])

    ##
    ## Arc validation
    ##
    
    def is_valid_storage_arc(self, arc, origin, dest, origin_to_arc, arc_to_dest):
        return (arc[0] != None and arc[1] != None and origin_to_arc != None and arc_to_dest != None and     # 1. is valid node and path
               arc[0][2] == arc[1][1] and                                                                   # 2. arc is consectutive
               origin[1] + origin_to_arc < min(arc[0][2], arc[1][2]) and                                    # 3. can reach this arc using shortest paths
               max(origin[1] + origin_to_arc, arc[0][1], arc[1][1]) + arc_to_dest <= dest[1])               # 4. can reach destination in time

    def is_arc_valid(self, arc, origin, dest, origin_to_arc, arc_to_arc, arc_to_dest):
        return (arc[0] != None and arc[1] != None and origin_to_arc != None and arc_to_dest != None and                     # 1. is valid node and path
               arc[1][0] != origin[0] and arc[0][0] != dest[0] and                                                          # 2. no inflow into origin, nor outflow from destination
               origin[1] + origin_to_arc < min(arc[0][2], arc[1][2] - arc_to_arc) and                                       # 3. can reach this arc using shortest paths
               max(origin[1] + origin_to_arc + arc_to_arc, arc[0][1] + arc_to_arc, arc[1][1]) + arc_to_dest <= dest[1] and  # 4. can reach destination in time
               arc[1][1] - arc[0][2] < arc_to_arc < arc[1][2] - arc[0][1])                                                  # 5. transit time within interval is valid?
    
    ##
    ## Validate any arc
    def V(self, k, a):
        origin,dest = self.commodities[k]['a'], self.commodities[k]['b']

        if a[0][0] != a[1][0]:
            origin_to_arc = self.edge_shortest_path[(dest[0],origin[0])].get(a[0][0], None)
            arc_to_dest = self.edge_shortest_path[(a[0][0],a[1][0])].get(dest[0], None)

            return self.is_arc_valid(a, origin, dest, origin_to_arc, self.network[a[0][0]][a[1][0]]['weight'], arc_to_dest)
        else:
            return self.is_valid_storage_arc(a, origin, dest, self.shortest_path(k, origin[0], a[0][0]), self.shortest_path(k, a[0][0], dest[0]))

    ##
    ## Add new timepoints to the system
    def add_network_timepoints(self, new_timepoints):
        if not new_timepoints:
            return

        #self.initial_timepoints.update(new_timepoints)
        new_arcs = [dict() for k in range(len(self.commodities))]

        updated_intervals = {}

        # archive network
        for G1,G2 in zip(self.timed_network,self.archive_timed_network):
            for e1,e2,d in G1.edges(data=True):
                if not G2.has_edge(e1,e2):
                    G2.add_edge(e1,e2,d)

        for e1,e2,d in self.cons_network.edges(data=True):
            if not self.archive_cons_network.has_edge(e1,e2):
                self.archive_cons_network.add_edge(e1,e2,d)

        ## Update Graph
        ##
        for n,t in new_timepoints:
            # find current interval that gets split by new timepoint - should always succeed if time >= 0 and time <= T
            i0 = next((i for i in self.intervals.select(n, '*', '*') if i[1] < t < i[2]), None)

            # ignore timepoints that are already in the system
            if i0 == None:
                continue

            # split interval
            i1,i2 = (n, i0[1], t), (n, t, i0[2])
            self.intervals.remove(i0)
            self.intervals.extend([i1,i2])

            has_i0 = updated_intervals.has_key(i0)
            updated_intervals[i1] = updated_intervals[i0] if has_i0 else i0
            updated_intervals[i2] = updated_intervals[i0] if has_i0 else i0

            # update consolidation nodes (i0 -> i1)
            self.cons_network.add_node(i1)
            self.cons_network.add_node(i2)
            self.cons_network.remove_node(i0)

            # update all commodity networks
            for k,G in enumerate(self.timed_network):
                self.add_network_timepoint(k, n, i0, i1, i2)

        ## Update Model
        ##
        # Improve performance of model by only adding the first dispatch arc for the same set of commodities
        for n in self.cons_network:
            for key,g in itertools.groupby(sorted(self.cons_network.edges(n, data=True)), lambda (a1,a2,d): a2[0]):
                for a,b in pairwise(g):
                    if a[2]['K'] >= b[2]['K']:
                        if 'z' not in b[2]:
                            b[2]['z'] = None
                    elif 'z' in b[2] and b[2]['z'] == None:
                        del b[2]['z']

        for a1,a2,d in self.cons_network.edges(data=True):
            if 'z' not in d:
                d['z'] = self.model.addVar(obj=(self.network[a1[0]][a2[0]]['fixed_cost']), lb=0, ub=GRB.INFINITY, name='z' + str((a1,a2)), vtype=GRB.INTEGER)

        for k,G in enumerate(self.timed_network):
            for a1,a2,d in G.edges(data=True):
                if 'x' not in d and (a1[0] == a2[0] or self.cons_network[a1][a2]['z'] != None):
                    d['x'] = self.model.addVar(obj=(self.network[a1[0]][a2[0]]['var_cost'] * self.commodities[k]['q'] if a1[0] != a2[0] else 0), lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(k) + ',' + str((a1,a2)))
                    new_arcs[k][(a1,a2)] = d['x']

        self.model.update()  # add variables to model


        for i in updated_intervals.items():
            # create flow constraints for new intervals
            for k,G in enumerate(self.timed_network):
                in_flow = [d['x'] for a1,a2,d in G.in_edges(i[0], data=True) if 'x' in d]
                out_flow = [d['x'] for a1,a2,d in G.out_edges(i[0], data=True) if 'x' in d]

                if in_flow or out_flow:
                    self.constraint_flow[(k,n)] = self.model.addConstr(quicksum(in_flow) - quicksum(out_flow) == self.r(k,i[0]), 'flow' + str((k,i[0])))

            # Consolidation constraints
            for a1,a2,d in self.cons_network.out_edges(i[0], data=True):
                if d['z'] != None:
                    self.constraint_consolidation[(a1,a2)] = self.model.addConstr(quicksum(self.timed_network[k][a1][a2]['x'] * self.commodities[k]['q'] for k in d['K']) <= d['z'] * self.network[a1[0]][a2[0]]['capacity'], 'cons' + str((a1,a2))) 

            for a1,a2,d in self.cons_network.in_edges(i[0], data=True):
                if d['z'] != None:
                    self.constraint_consolidation[(a1,a2)] = self.model.addConstr(quicksum(self.timed_network[k][a1][a2]['x'] * self.commodities[k]['q'] for k in d['K']) <= d['z'] * self.network[a1[0]][a2[0]]['capacity'], 'cons' + str((a1,a2)))


        # Subproblem constraints - let the new arcs equal the old ones
        group = defaultdict(list)

        for i in updated_intervals.items():
            group[i[1]].append(i[0])

        for k,G2 in enumerate(self.timed_network):
            G1 = self.archive_timed_network[k]

            for i in group.items():
                for _,i2 in G1.out_edges(i[0]):
                    if 'x' in G1[i[0]][i2]:
                        tmp = [G2[v][u]['x'] for v in i[1] for _,u in G2.out_edges(v) if u[0] == i2[0] and u[0] != i[0][0] and i2[1] <= u[1] and u[2] <= i2[2] and 'x' in G2[v][u]]
    
                        if tmp:                    
                            self.model.addConstr(quicksum(tmp) == G1[i[0]][i2]['x'], 'sub' + str((k,i[0],i2)))

                        # Since we are replacing this variable we set it's objective to 0
                        G1[i[0]][i2]['x'].Obj = 0

                        if i[0][0] != i2[0] and 'z' in self.archive_cons_network[i[0]][i2]:
                            self.archive_cons_network[i[0]][i2]['z'].Obj = 0

                for i1,_ in G1.in_edges(i[0]):
                    if 'x' in G1[i1][i[0]]:
                        tmp = [G2[u][v]['x'] for v in i[1] for u,_ in G2.in_edges(v) if u[0] == i1[0] and u[0] != i[0][0] and i1[1] <= u[1] and u[2] <= i1[2] and 'x' in G2[u][v]]

                        if tmp:                     
                            self.model.addConstr(quicksum(tmp) == G1[i1][i[0]]['x'], 'sub' + str((k,i1,i[0])))

                        # Since we are replacing this variable we set it's objective to 0
                        G1[i1][i[0]]['x'].Obj = 0

                        if i[0][0] != i1[0] and 'z' in self.archive_cons_network[i1][i[0]]:
                            self.archive_cons_network[i1][i[0]]['z'].Obj = 0



    # modify the graph to insert a time point
    def add_network_timepoint(self, k, n, i0, i1, i2):
        origin,dest = self.commodities[k]['a'], self.commodities[k]['b']
        v = functools.partial(self.V, k)

        # if splitting the origin/destination
        if i0 == self.origin_destination[k][0]:
            self.origin_destination[k] = (i1[1] <= origin[1] < i1[2] and i1 or i2, self.origin_destination[k][1])
        elif i0 == self.origin_destination[k][1]:
            self.origin_destination[k] = (self.origin_destination[k][0], i1[1] <= dest[1] < i1[2] and i1 or i2)

        # Add new node
        G = self.timed_network[k]
        G.add_node(i2)
        new_edges = v((i1,i2)) and [(i1,i2)] or []

        # Relabel nodes (i0 -> i1)
        G.add_node(i1)
        i1_out_edges = [(i1, target) for (_,target) in G.edges(i0)]
        i1_in_edges = [(source, i1) for (source,_) in G.in_edges(i0)]
        G.remove_node(i0)

        # Copy appropriate edges from i1 (out / in)
        i2_edges = filter(v, ((i2,e2) for e1,e2 in i1_out_edges)) + filter(v, ((e1,i2) for e1,e2 in i1_in_edges))
        new_edges.extend(i2_edges)

        # Remove invalid edges from i1, Keep good edges
        i1_edges = filter(lambda (e1,e2): v((e1,e2)), i1_out_edges + i1_in_edges)

        new_edges.extend(i1_edges)
        G.add_edges_from(new_edges)

        ## Update consolidation network
        for a1,a2 in new_edges:
            if a1[0] != a2[0]:
                if self.cons_network.has_edge(a1,a2):
                    self.cons_network[a1][a2]['K'].add(k)
                else:
                    self.cons_network.add_edge(a1,a2,{'K': set([k])})

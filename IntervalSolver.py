from collections import defaultdict
import itertools
import time
import networkx as nx
import logging
import random
import sys
from gurobipy import Env, GRB, tuplelist, Constr, Var
from Solver import Solver
from operator import itemgetter
from functools import partial
from math import ceil, floor
from tools import partition, TypedDiGraph
from itertools import pairwise
from enum import Enum, IntEnum
from CheckSolution import CheckSolution, SolutionGraphCommodity, SolutionGraphConsolidation, SolutionGraphNode
from DrawLaTeX import DrawLaTeX
from ProblemData import Commodity, NodeInterval, NodeTime, ProblemData, TimedArc

check_count = 0
solve_time = 0

logging.basicConfig(format='%(message)s',level=logging.DEBUG)
logger = logging.getLogger("IntervalSolver")

# add the handlers to logger
fh = logging.FileHandler('intervalsolver.log')
logger.addHandler(fh)

class preprocessing_option(str, Enum):
    node = "node"
    arc = "arc"

class shortest_path_option(str, Enum):
    commodity = "commodity"
    shared = "shared"
    edges = "edges"

class algorithm_option(IntEnum):
    default = 0
    multiplex = 1
    eclectic = 2
    reduced = 3
    adaptive = 4

ALGORITHM = algorithm_option.adaptive
MIP_GAP = 0.00001
TIMEOUT = 3600 #float("inf") # seconds

IN_TREE = False
ORIGIN_CUTS = False
USER_CUTS = True # True
PATH_CUTS = False
ALLOW_SPLIT = False
SPLIT_PRECISION = 0.001
HOLDING_COSTS = False
MAX_WALK_PATHS = 100
PRECISION = 2 # decimal places
INCUMBENT_CHECK_INTERVAL = float("inf") #50
USE_HEURISTIC_START = False

## useful check for exploring solution graph
def is_node(n: SolutionGraphNode):
    return isinstance(n, SolutionGraphCommodity)

# fix gurobi/cplex issue
def quicksum(col):
    return sum(col)

def round_tuple(t):
    return tuple(map(lambda x: isinstance(x, float) and round(x, PRECISION) or x, t)) if isinstance(t,tuple) else t

class IntervalSolver(object):
    """Solves time discretized service network design problems using an iterative approach"""
    __slots__ = ['problem', 'model', 'network', 'commodity_shortest_paths', 'shared_shortest_paths', 'S', 'T', 'x', 'z', 'commodities', 'intervals', 'arcs', 'origin_destination', 
                 'constraint_consolidation', 'constraint_flow', 'constraint_cycle' ,'constraint_path_length', 'solution_paths','consolidations', 'fixed_timepoints_model', 'timepoints', 
                 'incumbent', 'lower_bound', 'shouldEnforceCycles', 'fixed_paths','timed_network','cons_network','suppress_output','GAP', 'incumbent_solution','all_paths', 'edge_shortest_path', 
                 'status','timepoints_per_iteration', 'ALGORITHM', 'constraints_user', 'constraints_origin', 'constraints_dest', 'constraints_intree_path', 'constraints_intree', 'var_intree', 
                 'constraints_holding_offset', 'constraints_holding_enforce', 'constraints_holding_enforce2', 'environment']

    def __init__(self, problem: ProblemData, time_points:set[NodeTime]|None=None, full_solve=True, fixed_paths=[], suppress_output=False, gap=MIP_GAP, algorithm=None, full_discretization=False, full_results_log=None, environment=None):
        self.problem = problem
        self.commodities = [Commodity(NodeTime(c.a[0], round(c.a[1], PRECISION)), NodeTime(c.b[0], round(c.b[1], PRECISION)), round(c.q, PRECISION)) for c in problem.commodities]

        self.S = min(c.a[1] for c in self.commodities)  # time horizon
        self.T = max(c.b[1] for c in self.commodities) + 1  # time horizon

        self.incumbent = None  # store the lowest upper bound
        self.incumbent_solution = None
        self.lower_bound = 0.0 # can assume optimal is >= 0
        self.solution_paths = []
        self.shouldEnforceCycles = True
        self.fixed_paths = fixed_paths
        self.suppress_output = suppress_output
        self.GAP = gap
        self.ALGORITHM = algorithm if algorithm is not None else ALGORITHM
        self.status = False

        # build graph
        self.network = TypedDiGraph[int]()

        for a, destinations in problem.network.items():
            # support holding costs
            self.network.add_edge(a, a, weight=0, capacity=1, fixed_cost=0, var_cost=1 if HOLDING_COSTS else 0)

            # dispatch arcs
            for b, transit_time in destinations.items():
                self.shouldEnforceCycles = self.shouldEnforceCycles or problem.var_cost[0].get((a,b),0) == 0  # if all arcs have positive costs then we don't need to add extra constraints
                self.network.add_edge(a, b, weight=transit_time, capacity=problem.capacities.get((a,b), 1.0), fixed_cost=problem.fixed_cost.get((a,b), transit_time))#, var_cost=problem.var_cost[0].get((a,b), 0))

        self.create_shortest_paths(shortest_path_option.edges)

        ## testing - gets all valid paths for usable arcs in build network
        #self.all_paths = [set(itertools.chain(*map(pairwise, limit_shortest_paths(self.network, c['a'][0], c['b'][0], 'weight', c['b'][1] - c['a'][1])))) for c in self.problem.commodities]

        # create initial intervals/arcs
        self.timepoints = set(self.trivial_network())
        self.fixed_timepoints_model = time_points is not None and not full_solve
        
        ##
        ## Construct Gurobi model
        ## 
        #model = self.model = Model("IMCFCNF", env=Env(""))
        #model = self.model = Solver(CPLEX, quiet=False)
        if environment is None:
            self.environment = Env("", empty=True)
            self.environment.setParam('OutputFlag', False)
            self.environment.start()
        else:
            self.environment = environment

        model = self.model = Solver(quiet=(not self.fixed_timepoints_model or suppress_output), env=self.environment)

        self.model.set_gap(0.04 if self.ALGORITHM >= algorithm_option.adaptive else gap)
        self.model.set_timelimit(TIMEOUT)
        ##model.setParam(GRB.param.TimeLimit, 14400) # 4hr limit
        #self.model.set_threads(1)

        self.timed_network: list[TypedDiGraph[NodeInterval]] = []

        # support full discretization mode (specialized model generation for full discretization because it's really slow o/w)
        if not full_discretization:
            self.build_network()

            # set which interval contains the origin/destination
            self.origin_destination = {k: TimedArc(NodeInterval(c.a[0], self.S, self.T), NodeInterval(c.b[0], self.S, self.T)) for k,c in enumerate(self.commodities)}
            self.intervals = tuplelist(((n, self.S, self.T) for n in self.network.nodes()))
        else:
            self.build_full_network()

            # set which interval contains the origin/destination
            self.origin_destination = {k: TimedArc(NodeInterval(c.a[0], ceil(c.a[1]), ceil(c.a[1])+1), NodeInterval(c.b[0], floor(c.b[1]), floor(c.b[1])+1)) for k,c in enumerate(self.commodities)}
            self.intervals = tuplelist(map(lambda n: (n[0], n[1], n[2]), self.cons_network.nodes()))


        ## in-tree constraint
        Kd = {}  # used later if IN_TREE
        self.var_intree = {}

        if IN_TREE:
            Kd = {g: list(map(itemgetter(0), coll)) for g,coll in itertools.groupby(sorted(enumerate(self.commodities), key=lambda t: t[1].b[0]), key=lambda t: t[1].b[0])}
            self.var_intree = {(ult_dest,orig,dest): self.model.addVar(0, 0, 1, self.model.binary(), 'intree') for ult_dest in self.network.nodes() for orig in self.network.nodes() for dest in self.network.nodes() if orig != dest and orig != ult_dest }


        self.model.update()

        ##
        ## Constraints
        ## 
        build_time = time.time()

        # flow constraints
        self.constraint_flow: dict[tuple[int, NodeInterval], Constr] = {}

        for k,G in enumerate(self.timed_network):
            for n in self.intervals:
                n = NodeInterval(*n)
                i = [d['x'] for a1,a2,d in G.in_edges_data(n) if 'x' in d]
                o = [d['x'] for a1,a2,d in G.out_edges_data(n) if 'x' in d]

                if i or o:
                    self.constraint_flow[(k,n)] = self.model.addConstr(quicksum(i) - quicksum(o) == self.r(k,n), 'flow' + str((k,n)))

            sys.stdout.write('{0:5.1f}%, {1:4.0f}s\r'.format(100*k/float(len(self.commodities)), time.time() - build_time))
            sys.stdout.flush()

        # Consolidation constraints
        self.constraint_consolidation = {(a1,a2): self.model.addConstr(quicksum(self.timed_network[k].edge_data(a1, a2)['x'] * self.commodities[k].q for k in d['K']) <= d['z'] * self.network.edge_data(a1[0], a2[0])['capacity'], 'cons' + str((a1,a2))) 
                                         for a1,a2,d in self.cons_network.edges_data() if d['z'] is not None }

        # Ensure no flat-cycles
        self.constraint_cycle = None

        if self.shouldEnforceCycles:
            self.constraint_cycle = {}
    
            for k,G in enumerate(self.timed_network):
                for n in self.network.nodes():
                    outflow = [d['x'] for a1,a2,d in G.edges_data() if a1[0] == n and a2[0] != n and 'x' in d]

                    if outflow:
                        self.constraint_cycle[(k,n)] = self.model.addConstr(quicksum(outflow) <= 1, 'cycle')


        # Ensure path length
        if PATH_CUTS:
            self.constraint_path_length = {}

            for k,G in enumerate(self.timed_network):
                path_length = 0

                for n in self.network.nodes():
                    outflow = [d['x']*self.transit(a1[0],a2[0]) for a1,a2,d in G.edges_data() if a1[0] == n and a2[0] != n and 'x' in d]

                    if outflow:
                        path_length += quicksum(outflow)

                self.constraint_path_length[k] = model.addConstr(self.commodities[k].a[1] + path_length <= self.commodities[k].b[1], 'path')


        ## in-tree constraint
        if IN_TREE:
            self.constraints_intree_path = {(ult_dest, orig): model.addConstr(quicksum([self.var_intree[ult_dest, orig, dest] for dest in self.network.nodes() if orig != dest and orig != ult_dest]) ==  1, 'intree') 
                                    for ult_dest in self.network.nodes() 
                                        for orig in self.network.nodes() if orig != ult_dest}

                
            self.constraints_intree = {}

            for ult_dest,coll in Kd.items():
                for k in coll:
                    for orig in self.network.nodes():
                        for dest in self.network.nodes():
                            if orig != dest and orig != ult_dest:
                                outflow = [d['x'] for a1,a2,d in self.timed_network[k].edges(data=True) if a1[0] == orig and a2[0] == dest and 'x' in d]
       
                                if outflow:
                                    self.constraints_intree[k,orig,dest] = model.addConstr(quicksum(outflow) <=  self.var_intree[ult_dest, orig, dest], 'intree') 



        # holding cost support
        self.constraints_holding_offset = {}
        self.constraints_holding_enforce = {}
        self.constraints_holding_enforce2 = {}

        # if HOLDING_COSTS:
        #     # todo remove if holding cost is 0
        #     for k,G in enumerate(self.timed_network):
        #         for n in self.intervals:
        #             i = [d['x'] for a1,a2,d in G.in_edges(n, data=True) if 'x' in d and a1[0] != a2[0]]
        #             o = [(d['x'], d['y']) for a1,a2,d in G.out_edges(n, data=True) if 'x' in d  and 'y' in d and a1[0] == a2[0]] # get holding arc out

        #             if i and o:
        #                 self.constraints_holding_offset[k,n] = self.model.addConstr(1 + o[0][1] >= quicksum(i) + o[0][0], 'holding-offset')
        #                 self.constraints_holding_enforce[k,n] = self.model.addConstr(o[0][1] <= o[0][0], 'holding-enforce')
        #                 self.constraints_holding_enforce2[k,a[0]] = self.model.addConstr(o[0][1] <= quicksum(i), 'holding-enforce')

        ## User cuts
        self.constraints_origin = []
        self.constraints_dest = []
        self.constraints_user = []
        self.user_cuts()

        # add timepoints
        self.model.update()
        
        if time_points is not None:
            self.timepoints.update(time_points)
            self.add_network_timepoints(time_points)

    ##
    ## testing user cuts
    ##
    # Consider a location and time in the network (l,t).  Furthermore, consider all the commodities that become available at that location and time.  
    # Suppose the sum of those commodity sizes is q, then I know that the flow out of (l,t) has to be at least the roundup of q (assuming q is specified in terms of trailer size).  
    # I can add that as a cut.  Furthermore, if there is only a single outbound arc at (l,t), then the same argument can be made at the location and time at the end of the arc.
    def user_cuts(self):
        # remove cuts
        for c in self.constraints_user:
            self.model.removeCons(c)

        for c in self.constraints_origin:
            self.model.removeCons(c)

        for c in self.constraints_dest:
            self.model.removeCons(c)

        self.model.update()
        self.constraints_user = []
        self.constraints_origin = {}
        self.constraints_dest = {}

        ## add cuts

        # todo when # outbound edges = 1
        if USER_CUTS:
            for a1,a2,d in self.cons_network.edges_data():
                if 'z' in d and d['z'] is not None:
                    for c in d['K']:
                        self.constraints_user.append(self.model.addConstr(d['z'] >= ceil(self.commodities[c].q/self.network.edge_data(a1[0], a2[0])['capacity']) * self.timed_network[c].edge_data(a1, a2)['x'], 'user'))

        ## after the origin timed-node, we must dispatch at some point.  Sum all dispatch arcs >= ceil(q/max capacity) 

        # each commodity
        if ORIGIN_CUTS:
            # get commodities that share origin
            for orig, K in itertools.groupby(sorted(map(lambda t: (t[1][0][0],t[0]), self.origin_destination.items())), itemgetter(0)):
                K = list(map(itemgetter(1),K))
                q = sum(self.commodities[k].q for k in K)

                dispatch_arcs = set()
                max_capacity = 0

                # get intervals at origin node
                for i in self.intervals.select(orig):
                    for k in K:
                        # get all dispatch arcs after and out of origin node (that are valid for commodity via shortest paths)
                        for a1,a2 in self.timed_network[k].out_edges(i, data=False):
                            d = self.cons_network.edge_data(a1, a2)
                            if 'z' in d and d['z'] is not None:
                                if not dispatch_arcs:
                                    max_capacity = self.network.edge_data(a1[0], a2[0])['capacity']
                                else:
                                    if max_capacity < self.network.edge_data(a1[0], a2[0])['capacity']:
                                        max_capacity = self.network.edge_data(a1[0], a2[0])['capacity']

                                dispatch_arcs.add((a1,a2))

                cons = self.model.addConstr(quicksum(self.cons_network.edge_data(a1, a2)['z'] for a1,a2 in dispatch_arcs) >= ceil(q/max_capacity), 'user')

                for k in K:
                    self.constraints_origin[k] = cons

            # get commodities that share dest
            for dest, K in itertools.groupby(sorted(map(lambda t: (t[1][1][0],t[0]), self.origin_destination.items())), itemgetter(0)):
                K = list(map(itemgetter(1),K))
                q = sum(self.commodities[k].q for k in K)

                dispatch_arcs = set()
                max_capacity = 0

                # get intervals at dest node
                for i in self.intervals.select(dest):
                    for k in K:
                        # get all dispatch arcs before and into dest node (that are valid for commodity via shortest paths)
                        for a1,a2 in self.timed_network[k].in_edges(i, data=False):
                            d = self.cons_network.edge_data(a1, a2)
                            if 'z' in d and d['z'] is not None:
                                if not dispatch_arcs:
                                    max_capacity = self.network.edge_data(a1[0], a2[0])['capacity']
                                else:
                                    if max_capacity < self.network.edge_data(a1[0], a2[0])['capacity']:
                                        max_capacity = self.network.edge_data(a1[0], a2[0])['capacity']

                                dispatch_arcs.add((a1,a2))

                cons = self.model.addConstr(quicksum(self.cons_network.edge_data(a1, a2)['z'] for a1,a2 in dispatch_arcs) >= ceil(q/max_capacity), 'user')

                for k in K:
                    self.constraints_dest[k] = cons



    def build_network(self):
        all_arcs: list[list[TimedArc]] = []

        for k in range(len(self.commodities)):
            G = TypedDiGraph[NodeInterval]()
            v = partial(self.V, k)
            #v = lambda a: self.V(k,a) and (a[0][0],a[1][0]) in self.all_paths[k]

            # Add node-intervals
            G.add_nodes_from(NodeInterval(n, self.S, self.T) for n in self.network.nodes())

            # Create timed arcs
            all_arcs.append(list(filter(v, (TimedArc(NodeInterval(e[0], self.S, self.T), NodeInterval(e[1], self.S, self.T)) 
                        for e in (self.network.edges() if not self.fixed_paths else self.fixed_paths[k])))))

            G.add_edges_from(((a[0], a[1], {
                'x': self.model.addVar(obj=(self.problem.var_cost[k].get((a[0][0],a[1][0]),0) * self.commodities[k].q if a[0][0] != a[1][0] else self.problem.var_cost[k].get((a[0][0],a[1][0]),0) * self.commodities[k].q * (a[0][2] - a[0][1])), lb=0, ub=1, type=self.model.binary() if not ALLOW_SPLIT else self.model.continuous(), name='x' + str(k) + ',' + str(a)),
                'y': self.model.addVar(obj=(0 if a[0][0] != a[1][0] else -self.problem.var_cost[k].get((a[0][0],a[1][0]),0) * self.commodities[k].q * (a[0][2] - a[0][1])), lb=0, ub=1, type=self.model.binary() if not ALLOW_SPLIT else self.model.continuous(), name='y' + str(k) + ',' + str(a)),
               })  for a in all_arcs[k]))
            self.timed_network.append(G)

        # create consolidation network
        self.cons_network = TypedDiGraph[NodeInterval]()
        self.cons_network.add_nodes_from(NodeInterval(n, self.S, self.T) for n in self.network.nodes())

#        cons = itertools.groupby(sorted(((a.source.T, a.target.T),k) for k,arcs in enumerate(all_arcs) for a in arcs), itemgetter(0))
        cons = itertools.groupby(sorted((a,k) for k,arcs in enumerate(all_arcs) for a in arcs), itemgetter(0))

        # need to store K in order to use it twice
        def create_data(a,coll):
            K = set(map(itemgetter(1),coll))

            return {'z': self.model.addVar(obj=(self.network.edge_data(a[0][0], a[1][0])['fixed_cost']), lb=0,
                                                                             ub=self.model.inf(), 
                                                                             name='z' + str(a), type=self.model.integer()), 
                    'K': K}

        self.cons_network.add_edges_from(((a[0], a[1], create_data(a,K)) for a,K in cons))


    ##
    ## Code for generating a full network model - yes it's horrible, but its much faster to run than reusing code
    ##
    def build_full_network(self):
        build_time = time.time()

        # group timepoints by node
        time_points = [NodeTime(n,t) for n in self.network.nodes() for t in range(int(self.S), int(self.T+1), 1)]
        interval_cache = {g: [NodeInterval(g,t1,t2) for t1,t2 in pairwise(sorted(set(map(itemgetter(1), coll)) | set([self.S, self.T])))] for g,coll in itertools.groupby(time_points, itemgetter(0)) }

        redirected_arcs = set()
        arcs: list[list[TimedArc]] = []

        # Create arcs ((n1, t1, t2), (n2, t3, t4)) pairs
        for k,c in enumerate(self.commodities):
            origin,dest = c.a, c.b

            # setup storage arcs
            origin_to_arc = self.shortest_paths(k, origin[0])
            arcs.append([TimedArc(i1,i2) for n in self.network.nodes() if n in origin_to_arc and self.shortest_path(k, n, dest[0]) is not None
                                    for (i1,i2) in pairwise(interval_cache[n]) if self.is_valid_storage_arc(TimedArc(i1,i2), origin, dest, origin_to_arc.get(n, None), self.shortest_path(k, n, dest[0]))])

            for e in self.network.edges():
                origin_to_arc = self.shortest_path(k, origin[0], e[0])
                arc_to_dest = self.shortest_path(k, e[1], dest[0])

                if origin_to_arc is not None and arc_to_dest is not None:
                    arcs[k].extend(self.create_arcs_between_nodes(k, e[0], e[1], origin, dest, interval_cache[e[0]], interval_cache[e[1]], redirected_arcs))

            sys.stdout.write('{0:5.1f}%, {1:4.0f}s\r'.format(100*k/float(len(self.commodities)), time.time() - build_time))
            sys.stdout.flush()
                

        # add 'redirected' arcs to all commodities so we don't miss consolidation opportunities
        for k,c in enumerate(self.commodities):
            origin,dest = c.a, c.b
            origin_to_arc = self.shortest_paths(k, origin[0])
 
            missing_arcs = set(a for a in redirected_arcs 
                                 if a not in arcs[k] and self.is_arc_valid(a, origin, dest, origin_to_arc.get(a[0][0], None), self.transit(a[0][0], a[1][0]), self.shortest_path(k, a[1][0], dest[0])))

            # add missing arcs if valid for k
            arcs[k].extend(missing_arcs)

            G = TypedDiGraph[NodeInterval]()
            G.add_nodes_from(itertools.chain(*interval_cache.values()))

            # added holding cost support
            G.add_edges_from(((a[0], a[1], {'x': self.model.addVar(obj=(self.problem.var_cost[k].get((a[0][0],a[1][0]),0) * self.commodities[k].q * ((a[0][2] - a[0][1]) if a[0][0] == a[1][0] else 1)), lb=0, ub=1, type=self.model.binary() if not ALLOW_SPLIT else self.model.continuous(), name='x' + str(k) + ',' + str(a)) })
                                for a in arcs[k]))
            self.timed_network.append(G)

        # create consolidation network
        self.cons_network = TypedDiGraph[NodeInterval]()
        self.cons_network.add_nodes_from(itertools.chain(*interval_cache.values()))

        cons = itertools.groupby(sorted(((a.source.T, a.target.T), k) for k,k_arcs in enumerate(arcs) for a in k_arcs if a[0][0] != a[1][0]), itemgetter(0))
        self.cons_network.add_edges_from(((a[0],a[1],{'z': self.model.addVar(obj=(self.network.edge_data(a[0][0], a[1][0])['fixed_cost']), lb=0, ub=self.model.inf(), name='z' + str(a), type=self.model.integer()), 
                                                        'K': set(map(itemgetter(1),K))}) 
                                            for a,K in cons))
        return arcs
    
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
            if time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and ((i1[0] != dest[0] or i2[0] == dest[0]) and (origin[1] + origin_to_arc < i1[2]) and (i2[1] + arc_to_dest <= dest[1]) and (((dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1]))))):
                new_arcs.append((i1, i2))
            else:
                # skip to correct interval
                while i2 is not None and time >= i2[2]:
                    i2 = next(it, None)

                # keep skipping if invalid, up until latest time
                while i2 is not None and i1[2] + transit_time >= i2[2]:
                    if (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and ((i1[0] != dest[0] or i2[0] == dest[0]) and (origin[1] + origin_to_arc < i1[2]) and (i2[1] + arc_to_dest <= dest[1]) and (((dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1]))))):
                        new_arcs.append((i1,i2))

                        # redirected
                        if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                            redirected_arcs.add((i1,i2))

                        break

                    i2 = next(it, None)

                # we are done for this arc
                if i2 is None:
                    break

                # possibly the last transit time (from above loop) is valid
                if i1[2] + transit_time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and ((i1[0] != dest[0] or i2[0] == dest[0]) and (origin[1] + origin_to_arc < i1[2]) and (i2[1] + arc_to_dest <= dest[1]) and (((dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1]))))):
                    new_arcs.append((i1,i2))

                    # redirected
                    if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                        redirected_arcs.add((i1,i2))

        return new_arcs



    def r(self, k: int, i: tuple[int, float, float]) -> int:
        # at origin
        if self.origin_destination[k].source == i:
            return -1

        # at destination
        elif self.origin_destination[k].target == i:
            return 1

        return 0

    def infeasible(self):
        if self.fixed_paths:
            return len([c for k,c in enumerate(self.commodities) if c.a[1] + sum(self.transit(*a) for a in self.fixed_paths[k]) > c.b[1]]) > 0
        else:
            return len([c for k,c in enumerate(self.commodities) if c.a[1] + self.shortest_path(k,c.a[0],c.b[0]) > c.b[1]]) > 0


    # walk up the tree from node t, and return all root nodes that have maximum early times
    def walks(self, G: TypedDiGraph[SolutionGraphNode], t: SolutionGraphCommodity, top=False):
        roots = []
        stack: list[tuple[list[SolutionGraphNode], list[SolutionGraphNode]]] = [([t],[])] # tail nodes, current path

        # 'recursive' walk up tree
        while stack:
            tails, path = stack.pop()

            for v in tails:
                edges = [(0.0,v)]
                k = v[0] if isinstance(v, SolutionGraphCommodity) else 0

                while edges:  # choose latest edge with largest commodity and latest time window
                    v = edges[0][1]
                    path.insert(0, v)

                    # choose 'latest' arc, preferring to stay with current commodity if tie
                    if isinstance(v, SolutionGraphCommodity):
                        k = v[0]

                    edges = sorted([((float(G.node_data(e1)['early']), len(self.commodities) - abs(k-e1[0]) if isinstance(e1, SolutionGraphCommodity) else 0), e1) for e1,_ in G.in_edges(v)], reverse=True)

                    # for consolidations, choose all commodities that have the max early time (add to stack)
                    if not top and not is_node(v) and len(roots) < MAX_WALK_PATHS:
                        k_,group = next(itertools.groupby(edges, lambda x: x[0][0]), (None,[]))
                        group = list(group)
                        stack.append((list(map(itemgetter(1), [g for g in group if g != edges[0]])), path[:]))

                roots.append((v, path))

        return roots
     
    ##
    ## Iterative Solve
    ##
    def solve(self, draw_filename='', start_time=time.time(), write_filename=''):
        global solve_time
        global USE_HEURISTIC_START
        logger = logging.getLogger("IntervalSolver")

        USE_HEURISTIC_START = False

        info = []
        info_time = time.time()
        start_time= time.time()

        # feasibility check: shortest path cannot reach destination - gurobi doesn't pick it up, because no arcs exist
        if self.infeasible():
            if not self.suppress_output:
                print("INFEASIBLE")
            self.status = False
            return info

        variable_gap = True
        iterations = -1
        new_timepoints = set((NodeTime(*tp) for tp in self.timepoints))  # not required, but might be good for testing
        len_timepoints = 0      # used to halt processing if haven't added unique timepoints in a while (hack - this shouldn't happen if my code was good)
        it_timepoints = 0
        self.timepoints_per_iteration = [(0, n,t) for n,t in self.timepoints]

        s = CheckSolution(self, self.environment)
        solve_time = 0

        # output statistics
        logger.info('{0:>3}, {1:>10}, {2:>10}, {3:>7}, {4:>6}, {5:>6}, {6}'.format(*'{0}#,LB,UB,Gap,Time,Solver,Type [TP]'.format('G').split(',')))

        while True:
            iterations += 1

            if write_filename != '':
                self.model.update()
                self.model.write(write_filename + "_" + str(iterations) + ".lp")  # debug the model

            t0 = time.time()

            if USE_HEURISTIC_START:
                self.model.update()
                relaxed = self.model.model.relax()

                t0 = time.time()
                relaxed.optimize()

                self.lower_bound = float(max(relaxed.objBound, self.lower_bound) if self.lower_bound is not None else relaxed.objBound)
                self.status = True if relaxed.status in [GRB.status.TIME_LIMIT, GRB.status.INTERRUPTED] and self.incumbent and (self.incumbent - self.lower_bound) < self.incumbent * self.GAP else (relaxed.status == GRB.status.OPTIMAL)
            else:
                self.solve_lower_bound(s)

            #self.solve_lower_bound()  # Solve lower bound problem
            solve_time += time.time() - t0

            self.model.set_timelimit(max(0, TIMEOUT - solve_time))

            # return if infeasible, time out or interupted
            if not self.status or solve_time > TIMEOUT:
                if self.model.is_abort() or solve_time > TIMEOUT:
                    ### output statistics
                    self.solution_paths, self.consolidations = self.get_inprogress()
                    solution, cycle = self.get_network_solution()

                    # if valid path length (for LP)
                    if not list(filter(lambda tw: tw[0] > tw[1], [solution.node_data(SolutionGraphCommodity(k,c.b[0]))['tw'] for k,c in enumerate(self.commodities)])):
                        t0 = time.time()
                        if s.validate(self.solution_paths, self.consolidations):
                            solve_time += time.time() - t0

                            solution_cost = s.get_solution_cost()
                            self.incumbent = min(self.incumbent, solution_cost) if self.incumbent is not None else solution_cost
                            logger.info("{0:>3}, {1:10.1f}, {2:10.1f}, {3:7.2%}, {4:6.2f}, {5:6.2f}, ABORTED".format(iterations, self.lower_bound, self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time))
                            logger.info('\nit: {0}, new tp: {1}, intervals: {2}, vars: {3}, time: {4:.2f} ({5:.2f})\n'.format(iterations, len(new_timepoints), len(self.intervals), len(self.model.getVars()), time.time()-start_time, solve_time))

                elif not self.suppress_output:
                    logger.error("INFEASIBLE")

                self.status = False
                info.append((self.lower_bound, self.incumbent, time.time()-info_time, solve_time, 0, self.model.NumVars, self.model.NumConstrs, self.model.PresolveNumVars, self.model.PresolveNumConstrs, iterations, ((self.incumbent - self.lower_bound)/self.incumbent) if self.incumbent is not None and self.incumbent > 0 else None)) # track information about the iteration
                return info

            using_heuristic = USE_HEURISTIC_START

            if using_heuristic:
                heuristic_objective, heuristic_solution_paths, heuristic_consolidations = self.solve_heuristic_lower_bound(False)

                best_obj = heuristic_objective
                best_sol = (heuristic_solution_paths, heuristic_consolidations)

                for run in range(0):
                    heuristic_objective, heuristic_solution_paths, heuristic_consolidations = self.solve_heuristic_lower_bound(True)

                    if best_obj > heuristic_objective:
                        best_obj = heuristic_objective
                        best_sol = (heuristic_solution_paths, heuristic_consolidations)

                self.solution_paths, self.consolidations = best_sol

                if self.incumbent is not None and best_obj > self.incumbent:
                    USE_HEURISTIC_START = False

            else:
                self.solution_paths, self.consolidations = self.get_inprogress()

            # return if we are only doing one iteration
            if self.fixed_timepoints_model:
                logger.info("{0:>3}, {1:10.1f}, ".format(iterations, self.lower_bound))
                self.status = True
                return iterations, len(new_timepoints), solve_time

            solution, cycle = self.get_network_solution()

            # draw the timepoints if required
            portrait = True

            if draw_filename != '':
                pdf = DrawLaTeX(self.commodities, self.network)
                pdf.draw_solution_network(solution, cycle)
                pdf.draw_latex(portrait)
                pdf.save(draw_filename + str(iterations))


            # Checks gap of a lower bound / upper bound solutions - 1% gap               
            if self.incumbent is not None and (self.incumbent - self.lower_bound) < self.incumbent*self.GAP:
                # output statistics
                output = "{0:>3}{2}, {1:10.1f}, ".format(iterations, self.lower_bound, "*" if using_heuristic else "")
                output += "{0:10.1f}, {1:7.2%}, {2:6.2f}, {3:6.2f},".format(self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time)

                logger.info(output + " Optimal Solution")
                logger.info('\nit: {0}, new tp: {1}, intervals: {2}, vars: {3}, time: {4:.2f} ({5:.2f})\n'.format(iterations, len(new_timepoints), len(self.intervals), len(self.model.getVars()), time.time()-start_time, solve_time))

                self.timepoints.update(new_timepoints)
                break


            ##
            ## Add time points if needed
            ##

            # switch to default algorithm if we're stuck (hack for bad code)
            if self.ALGORITHM >= algorithm_option.eclectic and it_timepoints < 2:
                path_failure, path_length_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints = self.find_timepoints_all(solution, cycle)
            elif self.ALGORITHM >= algorithm_option.multiplex and it_timepoints < 2:
                path_failure, path_length_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints = self.find_timepoints_multiplex(solution, cycle)
            else:
                path_failure, path_length_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints = self.find_timepoints_default(solution, cycle)

            #path_failure, path_length_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints = self.find_timepoints_advanced(solution, cycle)
            #path_failure, path_length_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints = self.find_timepoints_simple(solution, cycle)

            tp = path_length_timepoints | window_timepoints | cycle_timepoints | mutual_timepoints

            # output statistics
            output = "{0:>3}{2}, {1:10.1f}, ".format(iterations, self.lower_bound, "*" if using_heuristic else "")

            # if invalid path, skip the LP (since it's infeasible)
            if path_failure and self.incumbent is None:
                output += "{0:>10}, {0:>7}, {1:6.2f}, {2:6.2f}, P[{3}]".format('-', time.time()-start_time, solve_time, len(path_length_timepoints))
            else:
                # Solve UB LP, check solution costs
                t0 = time.time()
                if path_failure or s.validate(self.solution_paths, self.consolidations):
                    solve_time += time.time() - t0

                    if not path_failure:
                        solution_cost = s.get_solution_cost()

                        if self.incumbent is None or solution_cost < self.incumbent:
                            self.incumbent = solution_cost
                            self.incumbent_solution = (self.solution_paths, s.get_consolidations(), self.get_in_tree_paths())

                    assert self.incumbent is not None, "Incumbent should not be None at this point"
                    output += "{0:10.1f}, {1:7.2%}, {2:6.2f}, {3:6.2f},".format(self.incumbent, ((self.incumbent - self.lower_bound)/self.incumbent), time.time()-start_time, solve_time)

                    # Checks gap of a lower bound / upper bound solutions - 1% gap               
                    if (self.incumbent - self.lower_bound) < self.incumbent*self.GAP:
                        logger.info(output + " Optimal Solution")
                        logger.info('\nit: {0}, new tp: {1}, intervals: {2}, vars: {3}, time: {4:.2f} ({5:.2f})\n'.format(iterations, len(new_timepoints), len(self.intervals), len(self.model.getVars()), time.time()-start_time, solve_time))

                        self.timepoints.update(new_timepoints)
                        self.status = True
                        break

                    ##
                    ## holding cost - should only hit here if cts time feasible, but lowerbound is not tight (due to holding costs)
                    # if HOLDING_COSTS and not tp:
                    #     # how to choose timepoints for holding arcs?  Want to find the arc with the most cost in the solution that isn't accounted for in the lowerbound
                    #     test = [{ha: (self.network[ha[1]][ha[1]]['var_cost'] if self.problem.commodities[k]['b'][0] == ha[1] else self.network[ha[0]][ha[0]]['var_cost']) * self.problem.commodities[k]['q'] * h.X for ha,h in H.items()} for k,H in enumerate(s.h)]
                    #     pass


                    # variable gap for faster solve (turn off if no new time points found)
                    if self.ALGORITHM >= algorithm_option.adaptive:
                        if not tp or variable_gap and (self.incumbent - self.lower_bound)*0.25 < self.incumbent * self.GAP:
                            self.model.set_gap(self.GAP*0.98)
                            self.model.set_aggressive_cuts() # focus on proving optimality
                            variable_gap = False
                        elif variable_gap:
                            self.model.set_gap(((self.incumbent - self.lower_bound)/self.incumbent * 0.25))

            # Stop endless loop - hack for my bad code
            if len(new_timepoints) == len_timepoints:
                it_timepoints += 1

                USE_HEURISTIC_START = False
            else:
                len_timepoints = len(new_timepoints)
                it_timepoints = 0

            if it_timepoints > 5: # arbitrary number
                self.write_raw_timepoints("endless.txt")
                self.write_raw_solution('endless.sol')

                def all_nodes(G, t):
                    nodes = set([t])
                    roots = set()
                    stack = [t]

                    # 'recursive' walk up tree
                    while stack:
                        v = stack.pop()
                        prev_nodes = set(e[0] for e in G.in_edges(v))

                        if not prev_nodes:
                            roots.add(v)

                        nodes.update(prev_nodes)
                        stack.extend(prev_nodes)

                    # 'recursive' walk down tree
                    stack = list(roots)

                    while stack:
                        v = stack.pop()
                        next_nodes = set(e[1] for e in G.out_edges(v))
                       
                        nodes.update(next_nodes)
                        stack.extend(next_nodes)
                                
                    return nodes

                for i,G_ in enumerate(nx.weakly_connected_components(solution.G)):
                    G = solution.subgraph(G_)
                    ## test code - only return partial network based on invalid root/tails
                    #K = set(n[0] for n in G if type(n[1]) is not frozenset)
                    #tails = [last for last in ((k,self.commodities[k]['b'][0]) for k in K) if 'valid' in solution.nodes[last] and not solution.nodes[last]['valid']]

                    #nodes = set()

                    #for t in tails[:1]:
                    #    nodes.update(all_nodes(G, t))

                    #G = nx.subgraph(G, nodes)

                    if True: #any(True for n in G if not G.nodes[n]['valid']):
                        pdf = DrawLaTeX(self.commodities, self.network)
                        pdf.draw_solution_network(G, cycle)
                        #pdf.draw_timeline(self.intervals, self.arcs, 24, self.get_solution(), [0,2,3])
                        pdf.draw_latex(False)
                        pdf.save("endless_" + str(i))

                output += " Endless Loop\n"
                logger.error(output)
                self.status = False
                sys.exit()
                break


            if window_timepoints:
                output += " W[{0}]".format(len(window_timepoints-new_timepoints))

            if cycle_timepoints:
                output += " C[{0}]".format(len(cycle_timepoints-new_timepoints))

            if mutual_timepoints:
                output += " M[{0}]".format(len(mutual_timepoints-new_timepoints))

            if len(path_length_timepoints-new_timepoints) > 0:
                output += " P[{0}]".format(len(path_length_timepoints-new_timepoints))

            # track information about the iteration
            info.append((self.lower_bound, self.incumbent, time.time()-info_time, solve_time, len(tp - new_timepoints),  self.model.NumVars, self.model.NumConstrs, None, None, iterations, ((self.incumbent - self.lower_bound)/self.incumbent) if self.incumbent is not None and self.incumbent > 0 else None))

            if not self.suppress_output:
                logger.info(output)

            self.add_network_timepoints(tp)
            new_timepoints.update(tp)
            self.timepoints_per_iteration.extend((iterations+1, n,t) for n,t in tp)

        info.append((self.lower_bound, self.incumbent, time.time()-info_time, solve_time, 0, self.model.NumVars, self.model.NumConstrs, self.model.PresolveNumVars, self.model.PresolveNumConstrs, iterations,((self.incumbent - self.lower_bound)/self.incumbent) if self.incumbent is not None and self.incumbent > 0 else None)) # track information about the iteration
        return info

    def write_timepoints(self, filename):
        if not self.infeasible():
            with open(filename, "w") as file:
                file.write("data = [")
                for i,n,t in self.timepoints_per_iteration:
                    file.write("{{i:{0},n:{1},t:{2}}},".format(i,n,int(t)))
                file.write("];")

    def write_raw_timepoints(self, filename):
        with open(filename, "w") as file:
            file.write("[")
            for i,n,t in self.timepoints_per_iteration:
                file.write("({0},{1}),".format(n,int(t)))
            file.write("]")

    # solve the model
    def solve_lower_bound(self, solution_check: CheckSolution):
        global check_count
        global last_print
        check_count = time.time()
        last_print = time.time()
        incumbent_check = [time.time()]

        def callback(model, where):
            global check_count
            global last_print
            global solve_time

            if where == GRB.callback.MIP:
                objbst = model.cbGet(GRB.callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.callback.MIP_OBJBND)

                self.lower_bound = max(objbnd, self.lower_bound) if self.lower_bound is not None else objbnd

                if self.incumbent is not None and (self.incumbent - self.lower_bound) < self.incumbent * self.GAP:
                    model.terminate()
                    return # terminate

                if time.time() > last_print + 1:
                    last_print = time.time()
                    if self.incumbent is not None:
                        sys.stdout.write('{0:7.2%}, {1:7.2%} {2:7.1f}s\r'.format((objbst-objbnd)/objbst, (self.incumbent - self.lower_bound) / self.incumbent, solve_time + time.time() - check_count))
                    else:
                        sys.stdout.write('{0:7.2%} {1:7.1f}s\r'.format((objbst-objbnd)/objbst, solve_time + time.time() - check_count))
                    sys.stdout.flush()

            elif where == GRB.callback.MIPSOL:
                    if time.time() > incumbent_check[0] + INCUMBENT_CHECK_INTERVAL and (self.incumbent is None or model.cbGet(GRB.callback.MIPSOL_OBJBST) < self.incumbent):
                        incumbent_check[0] = time.time()
                        # try to get solution
                        solution_paths, consolidations = self.get_inprogress(True)
                        cons = [SolutionGraphConsolidation(c, frozenset(k)) for c,K in consolidations.items() for k in K if len(k) > 1]
                        solution, _ = self.get_network_solution(solution_paths, cons)

                        if not list(filter(lambda tw: tw[0] > tw[1], [solution.node_data(SolutionGraphCommodity(k,c.b[0]))['tw'] for k,c in enumerate(self.commodities)])):
                            if solution_check.validate(solution_paths, consolidations):
                                solution_cost = solution_check.get_solution_cost()

                                if self.incumbent is None or solution_cost < self.incumbent:
                                    self.incumbent = solution_cost
                                    self.solution_paths = solution_paths
                                    self.consolidations = consolidations


        self.model.update()
        #self.model.write('test.lp')
        
        if self.suppress_output:
            self.model.optimize()
        else:
            self.model.optimize(callback)

        self.lower_bound = max(self.model.objBound(), self.lower_bound) if self.lower_bound is not None else self.model.objBound()
        self.status = True if self.model.is_abort() and self.incumbent and (self.incumbent - self.lower_bound) < self.incumbent * self.GAP else self.model.is_optimal()


    def solve_heuristic_lower_bound(self, randomize):
        # process commodities in random order
        K = list(range(0,len(self.commodities)))

        if randomize:
            random.shuffle(K)
        else:
            K.sort(key=lambda k: self.commodities[k].q, reverse=True)

        consolidation_network = self.cons_network.copy()

        # create a path-graph for each commodity - this will simply be a path if freight does not allow splitting
        path_graphs = [nx.DiGraph() for k in self.timed_network]

        for k in K:
            commodity_network = self.timed_network[k].copy()

            for a1,a2,d in commodity_network.edges_data():
                # holding arcs
                if a1[0] == a2[0]:
                    d['weight'] = 0
                    continue

                # dispatch arcs
                if 'SOLUTION_K' not in consolidation_network.edge_data(a1, a2):
                    consolidation_network.edge_data(a1, a2)['SOLUTION_K'] = set()

                remaining_quantity = sum([self.commodities[sk].q for sk in consolidation_network.edge_data(a1, a2)['SOLUTION_K']]) % self.network.edge_data(a1[0], a2[0])['capacity']
                consolidation_quantity = (self.commodities[k].q + remaining_quantity) / self.network.edge_data(a1[0], a2[0])['capacity']
                consolidation_cost = self.network.edge_data(a1[0], a2[0])['fixed_cost'] * (ceil(consolidation_quantity) - (1 if remaining_quantity > 0 else 0))

                d['weight'] = self.problem.var_cost[k].get((a1[0],a2[0]), 0.0) * self.commodities[k].q + consolidation_cost + 0.001 # small constant to prevent cycles


            ## update consolidation network cost
            #for a1,a2,d in consolidation_network.edges(data=True):
            #    if 'SOLUTION_K' not in d:
            #        d['SOLUTION_K'] = set()

            #    remaining_quantity = sum([self.commodities[sk]['q'] for sk in d['SOLUTION_K']]) % self.network[a1[0]][a2[0]]['capacity']
            #    consolidation_quantity = (self.commodities[k]['q'] + remaining_quantity) / self.network[a1[0]][a2[0]]['capacity']
            #    consolidation_cost = self.network[a1[0]][a2[0]]['fixed_cost'] * (ceil(consolidation_quantity) - (1 if remaining_quantity > 0 else 0))

            #    d['weight'] = self.network[a1[0]][a2[0]]['var_cost'] * self.commodities[k]['q'] + consolidation_cost

            # find cheapest path
            path = nx.shortest_path(commodity_network.G, source=self.origin_destination[k].source, target=self.origin_destination[k].target, weight='weight')

            # update consolidation network solution
            for arc in pairwise(path):
                if arc[0][0] != arc[1][0]:
                    consolidation_network.edge_data(arc[0], arc[1])['SOLUTION_K'].add(k)
                    path_graphs[k].add_edge(arc[0][0], arc[1][0])


        # return objective
        total = 0.0

        arcs = dict()
        cons = dict()

        for a1,a2,d in consolidation_network.edges_data():
            quantity = sum([self.commodities[sk].q for sk in d['SOLUTION_K']])

            consolidation_cost = float(self.network.edge_data(a1[0], a2[0])['fixed_cost']) * ceil(quantity / float(self.network.edge_data(a1[0], a2[0])['capacity']))
            variable_cost = sum([self.commodities[sk].q * self.problem.var_cost[sk].get((a1[0],a2[0]),0) for sk in d['SOLUTION_K']])

            total += variable_cost + consolidation_cost

            cons[(a1,a2)] = list(d['SOLUTION_K'])

            for sk in d['SOLUTION_K']:
                arcs[(sk, a1, a2)] = 1

        # remove any subtours that don't involve origin/destination
        for k,PG in enumerate(path_graphs):
            for subtour in list(nx.weakly_connected_components(PG)):
                if self.commodities[k].a[0] not in subtour:
                    PG.remove_nodes_from(subtour)

        cons = {arc:frozenset(v) for arc,v in cons.items() if len(v) > 0}

        # associate consolidations to path_graph
        for arc,K in cons.items():
            for k in K:
                if 'K' in path_graphs[k].get_edge_data(arc[0][0],arc[1][0]):
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['K'].append(K)
                else:
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['K']= [K]

                if 'q' in path_graphs[k].get_edge_data(arc[0][0],arc[1][0]):
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['q'].append(arcs[k,arc[0],arc[1]])
                else:
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['q']= [arcs[k,arc[0],arc[1]]]


        for k,PG in enumerate(path_graphs):
            if [d for n1,n2,d in PG.edges(data=True) if 'K' not in d]:
                print("Missing")

        #return paths, {k:v for k,v in cons.items() if len(v) > 0}
        consolidations: defaultdict[tuple[int, int], list[frozenset[int]]] = defaultdict(list)
        for a,c in cons.items():
            consolidations[a[0][0],a[1][0]].append(c)  # may have multiple consolidations for same arc (at different times)

        return total, path_graphs, consolidations






    # check for single point path (theorem)
    def find_path_timepoint(self, k):
        (ok,ek),(dk,lk) = self.commodities[k].a, self.commodities[k].b

        return next((NodeTime(n2, ek + self.restricted_shortest_path((dk,n2), (ok, n1)) + self.transit(n1,n2)) 
                    for (n1,n2),(_,n3) in [(a,b) for a in self.solution_paths[k].edges() for b in self.solution_paths[k].out_edges(a[1])] 
                        if ek + self.restricted_shortest_path((dk,n2), (ok, n1)) + self.transit(n1,n2) + self.transit(n2,n3) + self.restricted_shortest_path((ok,n2), (n3, dk)) > lk), None)

    # check for single point disjoint (theorem)
    def find_disjoint_timepoint(self, k1, k2, n1, n2):
        (ok1,ek1),(dk1,_) = self.commodities[k1].a,self.commodities[k1].b
        (ok2,_),(dk2,lk2) = self.commodities[k2].a,self.commodities[k2].b

        # earliest time k1 to reach n2 is greater than latest time for k2
        tmp = ek1 + self.restricted_shortest_path((dk1, n2), (ok1,n1)) + self.transit(n1,n2)
        return NodeTime(n2, tmp) if tmp + self.restricted_shortest_path((ok2, n1), (n2,dk2)) > lk2 else None


    ##
    ## adds timepoints that are proven (in paper) to find optimal solution in finite time (only using major theorems)
    ##
    def find_timepoints_default(self, solution: TypedDiGraph[SolutionGraphNode], cycle):
        path_timepoints: set[NodeTime] = set()
        window_timepoints: set[NodeTime] = set()
        cycle_timepoints: set[NodeTime] = set()
        mutual_timepoints: set[NodeTime] = set()
        path_failure = None

        # check if path is too long
        for k,c in enumerate(self.commodities):
            tw = solution.node_data(SolutionGraphCommodity(k,c.b[0]))['tw']
            if tw[0] > tw[1]:
                path_failure = k
                break

        G = solution
        K = set(n[0] for n in G.nodes() if isinstance(n, SolutionGraphCommodity))
                
        ## 3. Check for cycle
        cycle_window = []

        for c in cycle:
            # get all commodities explicitly in cycle (ignore incidental commodities)
            cycle_K = set(n[0] for n in c if is_node(n))

            # want to start at earliest point in cycle, and rewrite cycle to match
            tw, start_node,i = sorted([(solution.node_data(n)['tw'], n, i) for i,n in enumerate(c) if is_node(n)])[0]
            t = tw[0]

            # paper uses this as a stopping criteria for cycle timepoints
            cycle_length = sum([self.transit(*a) for a in pairwise([n[1] for n in c if is_node(n)])])
            M = ceil((self.commodities[start_node[0]].b[1] - t) / cycle_length) * cycle_length + t

            cycle_window.append(c[0])

            # add timepoints from origin to cycle start
            for k in cycle_K:
                origin = self.commodities[k].a
                enter = sorted([x for x in c if G.has_path(SolutionGraphCommodity(k,origin[0]), x)], key=lambda x: nx.shortest_path_length(G.G, (k,origin[0]), x))[0]

                for p in pairwise(self.get_path(G, SolutionGraphCommodity(k,origin[0]), enter)):
                    cycle_timepoints.add(NodeTime(p[1][1] if isinstance(p[1], SolutionGraphCommodity) else p[1][0][0], G.node_data(p[1])['tw'][0]))

            for n in itertools.cycle(pairwise(list(map(itemgetter(1), filter(is_node, c[i:] + c[:i]))) + [start_node[1]])):
                cycle_timepoints.add(NodeTime(n[0], t))

                # stop if past time horizon or if one commodity has reached it's end
                if t > self.T or t > M:
                    break

                t += self.transit(*n)

        ## 4. Generic case
        # Note: if cycle then 'valid' is not set and is ignored
        # get all commodities with invalid destination time (ignoring already processed items)
        # prioritize path failed commodities
        tails = [last for last in (SolutionGraphCommodity(k, self.commodities[k].b[0]) for k in (K if path_failure is None else [path_failure])) 
                    if not cycle_timepoints and 'valid' in solution.node_data(last) and not solution.node_data(last)['valid']]

        for t in tails[:1]:
            for r,path in self.walks(G, t, True):
                if t[0] == r[0]:
                    path_timepoints.update(self.get_network_timepoints(solution, G, path, True))
                else:
                    mutual_timepoints.update(self.get_network_timepoints(solution, G, path, True))

        return path_failure is not None, path_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints

    ##
    ## uses theorems from paper, preferring single timepoints.
    ##
    def find_timepoints_multiplex(self, solution: TypedDiGraph[SolutionGraphNode], cycle):
        path_timepoints: set[NodeTime] = set()
        window_timepoints: set[NodeTime] = set()
        cycle_timepoints: set[NodeTime] = set()
        mutual_timepoints: set[NodeTime] = set()
        path_failure = False

        # check if path is too long
        for k,c in enumerate(self.commodities):
            tw = solution.node_data(SolutionGraphCommodity(k,c.b[0]))['tw']
            if tw[0] > tw[1]:
                path_failure = True
                break

        ## Treat subgraphs separately
        for G_ in nx.weakly_connected_components(solution.G):
            G = solution.subgraph(G_)
            K = set(n[0] for n in G.nodes() if isinstance(n, SolutionGraphCommodity))
                
            ## 1. Check for cycle
            cycle_window = []

            for c in cycle:
                # get all commodities explicitly in cycle (ignore incidental commodities)
                cycle_K = set(n[0] for n in c if is_node(n))

                if  not (cycle_K & K):
                    continue

                # want to start at earliest point in cycle, and rewrite cycle to match
                tw, start_node,i = sorted([(solution.node_data(n)['tw'], n, i) for i,n in enumerate(c) if is_node(n)])[0]
                t = tw[0]

                # paper uses this as a stopping criteria for cycle timepoints
                cycle_length = sum([self.transit(*a) for a in pairwise([n[1] for n in c if is_node(n)])])
                M = ceil((self.commodities[start_node[0]].b[1] - t) / cycle_length) * cycle_length + t

                cycle_window.append(c[0])

                # add timepoints from origin to cycle start
                for k in cycle_K:
                    origin = self.commodities[k].a
                    enter = sorted([x for x in c if G.has_path(SolutionGraphCommodity(k,origin[0]), x)], key=lambda x: nx.shortest_path_length(G.G, (k,origin[0]), x))[0]

                    for p in pairwise(self.get_path(G, SolutionGraphCommodity(k,origin[0]), enter)):
                        cycle_timepoints.add(NodeTime(p[1][1] if isinstance(p[1], SolutionGraphCommodity) else p[1][0][0], G.node_data(p[1])['tw'][0]))

                for n in itertools.cycle(pairwise(list(map(itemgetter(1), filter(is_node, c[i:] + c[:i]))) + [start_node[1]])):
                    cycle_timepoints.add(NodeTime(n[0], t))

                    # stop if past time horizon or if one commodity has reached it's end
                    if t > self.T or t > M:
                        break

                    t += self.transit(*n)

            ## 4. Generic case
            # Note: if cycle then 'valid' is not set and is ignored
            # get all commodities with invalid destination time (ignoring already processed items)
            tails = [last for last in (SolutionGraphCommodity(k,self.commodities[k].b[0]) for k in K) 
                        if 'valid' in solution.node_data(last) and not solution.node_data(last)['valid']]

            for t in tails:
                for r,path in self.walks(G, t, False):
                    if t[0] == r[0]:
                        path_timepoints.update(self.get_network_timepoints(solution, G, path, True))
                    else:
                        mutual_timepoints.update(self.get_network_timepoints(solution, G, path, True))

        return path_failure, path_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints

    ##
    ## uses all tricks to get 'best' timepoints to add
    ##
    def find_timepoints_all(self, solution: TypedDiGraph[SolutionGraphNode], cycle) -> tuple[bool, set[NodeTime], set[NodeTime], set[NodeTime], set[NodeTime]]:
        path_timepoints: set[NodeTime] = set()
        window_timepoints: set[NodeTime] = set()
        cycle_timepoints: set[NodeTime] = set()
        mutual_timepoints: set[NodeTime] = set()
        path_failure = False

        ## Treat subgraphs separately
        for G_ in nx.weakly_connected_components(solution.G):
            G = solution.subgraph(G_)
            K = set(n[0] for n in G.nodes() if isinstance(n, SolutionGraphCommodity))
        
            ## 1. Path length violation, find if there is a single timepoint fix
            path_fail_K: set[int] = set()  # keep track of commodities that fail path length

            for k in K:
                tw = solution.node_data(SolutionGraphCommodity(k, self.commodities[k].b[0]))['tw']

                # check if path is too long
                if tw[0] > tw[1]:
                    path_failure = True

                    # check for single point (theorem)
                    tp = self.find_path_timepoint(k)
                    
                    if tp is not None:
                        path_timepoints.add(tp)
                        path_fail_K.add(k)

            ## 2. Disjoint time window for two commodities, find if there is a single timepoint fix
            window = []
            first_failed_consolidation = []

            # find first broken TW consolidation for each commodity
            commodity_failed_consolidation = {k: next((n for n in self.get_path(G, SolutionGraphCommodity(k, self.commodities[k].a[0]), SolutionGraphCommodity(k,self.commodities[k].b[0]))
                                                         if isinstance(n, SolutionGraphConsolidation) and solution.node_data(n)['tw'][1] < solution.node_data(n)['tw'][0]), None)
                                                for k in (K-path_fail_K)}

            # remove downstream consolidations, ignoring any path length violations
            for k,cons in commodity_failed_consolidation.items():
                if cons is not None and not any(G.has_path(n, cons) for n in first_failed_consolidation) and not any(G.has_path(SolutionGraphCommodity(i,self.commodities[i].a[0]), cons) for i in path_fail_K):
                    first_failed_consolidation = [cons] + [n for n in first_failed_consolidation if not G.has_path(cons, n)]

            # find if any consolidations breaks with a single timepoint (theorem)
            for cons in first_failed_consolidation:
                n1,n2 = cons[0]

                # sort by smallest gap, this way it attempts to split the consolidation commodities more evenly (and have greater effect)
                for k1,k2 in sorted(itertools.permutations(cons.commodities, 2), key=lambda x: abs(G.node_data(SolutionGraphCommodity(x[0],n1))['tw'][0]-G.node_data(SolutionGraphCommodity(x[1],n1))['tw'][1]), reverse=False):
                    tp = self.find_disjoint_timepoint(k1, k2, n1, n2)

                    if tp is not None:
                        window_timepoints.add(tp)
                        window.append(cons)
                        break

            ## Add more - if some of the 'upstream' consolidations couldn't be broken with single point, try a little further downstream!
            for k in sorted(K-path_fail_K, key=lambda k: self.commodities[k].q, reverse=True):
                cons = commodity_failed_consolidation[k]

                if cons is None or any(G.has_path(n, cons) for n in window):
                    continue

                n1,n2 = cons[0]

                for k2 in sorted(cons[1]-set([k]), key=lambda k2: abs(G.node_data(SolutionGraphCommodity(k,n1))['tw'][0]-G.node_data(SolutionGraphCommodity(k2,n1))['tw'][1]), reverse=True):
                    tp = self.find_disjoint_timepoint(k, k2, n1, n2) or self.find_disjoint_timepoint(k2, k, n1, n2)

                    if tp is not None:
                        window_timepoints.add(tp)
                        window.append(cons)
                        break
        
        
            ## 3. Check for cycle
            cycle_window = []

            for c in cycle:
                # get all commodities explicitly in cycle (ignore incidental commodities)
                cycle_K = set(n[0] for n in c if is_node(n))

                # ignore any node that is already connected to an invalid path or TW consolidation
                if  not (cycle_K & K) or any(solution.has_path(n, c[0]) for n in window) or any(solution.has_path(SolutionGraphCommodity(k, self.commodities[k].a[0]), c[0]) for k in path_fail_K):
                    continue

                ### check to see if we can break cycle by simply TW
                for cons in c:
                    if is_node(cons):
                        continue

                    n1,n2 = cons[0]

                    # find if this commodity breaks this consolidation with a single timepoint (theorem)
                    for k1,k2 in sorted(itertools.permutations(cons[1] & cycle_K, 2), key=lambda x: abs(G.node_data(SolutionGraphCommodity(x[0],n1))['tw'][0]-G.node_data(SolutionGraphCommodity(x[1],n1))['tw'][1]), reverse=False):
                        tp = self.find_disjoint_timepoint(k1, k2, n1, n2) or self.find_disjoint_timepoint(k1, k2, n1, n2)

                        if tp is not None:
                            cycle_timepoints.add(tp)
                            cycle_window.append(cons)
                            break

                if cycle_timepoints:
                    break

                ### couldn't break with TW timepoint, so use generic method instead
                # want to start at earliest point in cycle, and rewrite cycle to match
                tw, start_node,i = sorted([(solution.node_data(n)['tw'], n[1], i) for i,n in enumerate(c) if is_node(n)])[0]
                t = tw[0]

                cycle_window.append(c[0])

                # add timepoints from origin to cycle start
                for k in cycle_K:
                    origin = self.commodities[k].a
                    enter = sorted([x for x in c if G.has_path(SolutionGraphCommodity(k,origin[0]), x)], key=lambda x: nx.shortest_path_length(G.G, (k,origin[0]), x))[0]

                    for p in pairwise(self.get_path(G, SolutionGraphCommodity(k,origin[0]), enter)):
                        cycle_timepoints.add(NodeTime(p[1][1] if isinstance(p[1], SolutionGraphCommodity) else p[1][0][0], G.node_data(p[1])['tw'][0]))

                for n in itertools.cycle(pairwise(list(map(itemgetter(1), filter(is_node, c[i:] + c[:i]))) + [start_node])):
                    cycle_timepoints.add(NodeTime(n[0], t))

                    # stop if past time horizon or if one commodity has reached it's end
                    if t > self.T or [k for k in cycle_K if t + self.shortest_path(k, n[0], self.commodities[k].b[0]) > self.commodities[k].b[1]]:
                        break

                    t += self.transit(*n)

            ## 4. Generic case
            # Note: if cycle then 'valid' is not set and is ignored
            # get all commodities with invalid destination time (ignoring already processed items)
            tails = [last for last in (SolutionGraphCommodity(k,self.commodities[k].b[0]) for k in K) 
                        if 'valid' in solution.node_data(last) and not solution.node_data(last)['valid'] and 
                            not any(solution.has_path(w, last) for w in window) and 
                            not any(solution.has_path(SolutionGraphCommodity(kp, self.commodities[kp].a[0]), last) for kp in path_fail_K)]

            for t in tails:
                for r, path in self.walks(G, t, False):
                    if t[0] == r[0]:
                        path_timepoints.update(self.get_network_timepoints(solution, G, path))
                    else:
                        mutual_timepoints.update(self.get_network_timepoints(solution, G, path))

                ###
                ### Testing: add all shortest path timepoints for failed commodities
                ###
                #c = self.commodities[t[0]]
                #tmp = c['a'][1]

                #for n1, n2 in pairwise(nx.shortest_path(self.network, c['a'][0], c['b'][0], weight='weight')):
                #    path_timepoints.add((n1, tmp))
                #    tmp += self.transit(n1,n2)
                #path_timepoints.add((c['b'][0], tmp))

        return path_failure, path_timepoints, window_timepoints, cycle_timepoints, mutual_timepoints

    ##
    ## Gets the shortest path between 2 nodes, while prioritizing the visited commodities
    ##
    def get_path(self, G: TypedDiGraph[SolutionGraphNode], r: SolutionGraphCommodity, t: SolutionGraphNode):
        yield r

        def prioritize_commodity(x):
            if is_node(x):
                if x.commodity in t.commodities:
                    return -1
                elif x.commodity == r.commodity:
                    return 0
                else:
                    return G.node_data(n)['tw'][1] # earliest 'late time window'
            return 0
                    
        n = r
        # follow first and last commodity as much as possible
        while n != t:
            arcs = G.out_edges(n)
            n = sorted([a[1] for a in arcs if G.has_path(a[1], t)], key=prioritize_commodity)[0]
            yield n
    
    ##
    ## Gets timepoints to break a generic acyclic solution
    ##
    def get_network_timepoints(self, solution: TypedDiGraph[SolutionGraphNode], G: TypedDiGraph[SolutionGraphNode], path: list[SolutionGraphNode], all_timepoints=False):
        r,t = path[0], path[-1]
        assert type(r) is SolutionGraphCommodity and type(t) is SolutionGraphCommodity, "First and last nodes of path should not be a consolidation"

        cr = self.commodities[r[0]]
        ct = self.commodities[t[0]]
        x = cr.a.node
        has_k = False
        next_tk = None
        tp: set[NodeTime] = set()

        first = next(v2 for v1,v2 in pairwise(path) if t.commodity in v1.commodities)
        last = next(v1 for v1,v2 in pairwise(reversed(path)) if r.commodity in v2.commodities)

        # skip nodes that can be reached by shortest paths
        def canDrop(v: SolutionGraphNode):
            return not all_timepoints and v != last and v != first and self.shortest_path(r[0], cr.a[0], v.node) is not None and cr.a[1] + self.shortest_path(r[0], cr.a[0], v.node) >= solution.node_data(v)['early']

        for n,n2 in pairwise(itertools.dropwhile(canDrop, path)):
            #has_k = r.commodity in n.commodities

            if not has_k and not is_node(n) and t[0] in n.commodities:      # only add another point if aending at first consolidation
                next_tk = n

            if solution.node_data(n)['early'] > self.T:
                break

            has_k,x = (t.commodity in n.commodities, n.node)
            tp.add(NodeTime(x, solution.node_data(n)['early']))

            # finish if can reach end of commodity using shortest paths
            x2 = n2.node if isinstance(n2, SolutionGraphCommodity) else n2.arc[1]

            x2_to_dest = self.restricted_shortest_path((ct.a[0], x), (x2, ct.b[0]))
            x2_early = solution.node_data(SolutionGraphCommodity(t[0],x2))['early'] if solution.has_node(SolutionGraphCommodity(t[0],x2)) else None

            if not all_timepoints and has_k and x2_to_dest is not None and x2_early is not None and x2_early + x2_to_dest > ct.b[1]:
                if next_tk == n and solution.node_data(n2)['early'] < self.T:
                    tp.add(NodeTime(x2, solution.node_data(n2)['early']))

                return tp

        if solution.node_data(t)['early'] < self.T:
            tp.add(NodeTime(t[1], solution.node_data(t)['early']))

        return tp

    # print solution
    def writeSolution(self, file):
        if self.incumbent_solution is not None:
            self.problem.save(file, (self.incumbent, self.incumbent_solution[0], self.incumbent_solution[1]))

    def write_raw_solution(self, file):
        self.problem.save(file, (self.model.objVal(), self.solution_paths, sorted([((c[0],c[1]), frozenset(k)) for c,k in self.consolidations.items() if len(k) > 1])))


    def drawSolution(self, file, incumbent=True):
        pdf = DrawLaTeX(self.commodities, self.network)
        #pdf.draw_network(scale, position=self.problem.position, font='normal')
        #pdf.draw_commodities(scale, position=self.problem.position)

        assert self.incumbent_solution is not None, "No incumbent solution to draw"
        solution, cycle = self.get_network_solution(*(self.incumbent_solution[:2] if incumbent else (None,None))) 
        pdf.draw_solution_network(solution, cycle)

        #arcs = [G.edges() for G in self.timed_network]
        #pdf.draw_timeline(self.intervals, arcs, 24 if not portrait else 15, self.get_solution())
        pdf.draw_latex()
        pdf.save(file)

    def printSolution(self, incumbent=True):
        sol = self.incumbent_solution if incumbent else (self.solution_paths, sorted([SolutionGraphConsolidation(c, k) for c,K in self.consolidations.items() for k in K if len(k) > 1]), self.get_in_tree_paths())

        if sol is not None:
            print("cost={0}".format(self.incumbent))

            print("PATHS,{0}".format(len(sol[0])))

            for k,p in enumerate(sol[0]):
                print("{0},[{1}]".format(k,",".join(map(lambda x: str(x),p))))

            print("CONSOLIDATIONS,{0}".format(len(sol[1])))

            for (n1,n2),K in sol[1]:
                print("{0},{1},[{2}]".format(n1,n2,",".join(map(str,K))))

            # in-tree
            for ult_dest, orig, dest in sol[2]:
                print(ult_dest, orig, dest)

        else:
            print('No solution')

    # get statistics
    def get_statistics(self):
        stats = {}
        stats['cost'] = self.model.objVal() if self.status and self.incumbent is None else self.incumbent
        stats['nodes'] = len(self.network.nodes())
        stats['intervals'] = len(self.intervals)
        stats['arcs'] = sum(len(G.edges()) for G in self.timed_network)
        stats['variables'] = len(self.model.getVars())
       
        if self.status:
            paths, cons = self.get_inprogress()
            stats['paths'] = paths
            stats['consolidation'] = cons

        stats['commodities'] = len(self.commodities)
        stats['lower_bound'] = self.lower_bound

        try:
            m = self.model.presolve()
            stats['presolve_vars'] = len(m.getVars())
            stats['presolve_cons'] = len(m.getConstrs())
        except Exception:
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


    # get in-tree paths
    def get_in_tree_paths(self):
        if not IN_TREE:
            return []

        return sorted(((ult_dest, orig, dest) for (ult_dest, orig, dest), y in self.var_intree.items() if self.model.val(y) > 0.5))

    # get all arcs used for transport
    def get_solution(self):
        assert self.status, "Model is not solved"
        return tuplelist((k,(a1.T,a2.T)) for k,G in enumerate(self.timed_network) for a1,a2,d in G.edges_data() if 'x' in d and self.model.val(d['x']) > 0)

    ##
    ## From the solution, return all arcs used for a given commodity
    ##
    def get_solution_arcs(self):
        result: dict[int, list[NodeInterval]] = {}
        arcs = dict(((k,a[0]),NodeInterval(*a[1])) for (k,a) in self.get_solution())

        for k,(origin,dest) in self.origin_destination.items():
            i = origin
            result[k] = [i]

            while i != dest:
                i = arcs[(k,i)]
                result[k].append(i)
        
        return result


    def get_inprogress(self, cb=False):
        x = [(k,(a1,a2),d['x']) for k,G in enumerate(self.timed_network) for a1,a2,d in G.edges_data() if 'x' in d]
        z = [(a1,a2,z['z'],z['K']) for a1,a2,z in self.cons_network.edges_data() if z['z'] is not None]

        if cb:
            vars = self.model.model.cbGetSolution(list(map(itemgetter(2), x + z)))
        else:
            vars = self.model.vals(list(map(itemgetter(2), x + z)))

        # create a path-graph for each commodity - this will simply be a path if freight does not allow splitting
        path_graphs = [nx.DiGraph() for k in self.timed_network]

        for i,(k,a,_) in enumerate(x):
            if vars[i] > SPLIT_PRECISION and a[0][0] != a[1][0]:
                path_graphs[k].add_edge(a[0][0], a[1][0])

        # remove any subtours that don't involve origin/destination
        for k,PG in enumerate(path_graphs):
            for subtour in list(nx.weakly_connected_components(PG)):
                if self.commodities[k].a[0] not in subtour:
                    PG.remove_nodes_from(subtour)

        arcs = dict(((k,a[0],a[1]),vars[i]) for i,(k,a,_) in enumerate(x) if vars[i] > SPLIT_PRECISION)
        offset = len(x)
        pw = list(map(frozenset,[PG.edges() for PG in path_graphs]))

        ## add paths - for non-splitting case
        #paths = []
        #arcs = dict(((k,a[0]),a[1]) for i,(k,a,_) in enumerate(x) if vars[i] > 0)

        #for k, (origin,dest) in self.origin_destination.items():
        #    i = origin
        #    paths.append([i[0]])
                
        #    while i[0] != dest[0]:
        #        i = arcs[(k,i)]
                
        #        if i[0] != paths[k][-1]:
        #            paths[k].append(i[0])

        #offset = len(x)
        #pw = map(frozenset,map(pairwise,paths))

        cons = {(a1,a2): [k for k in K if (a1[0],a2[0]) in pw[k] and (k,a1,a2) in arcs] 
                    for i,(a1,a2,_,K) in enumerate(z) if round(vars[offset + i]) > 0}

        cons = {arc:frozenset(v) for arc,v in cons.items() if len(v) > 0}

        # associate consolidations to path_graph
        for arc,K in cons.items():
            for k in K:
                if 'K' in path_graphs[k].get_edge_data(arc[0][0],arc[1][0]):
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['K'].append(K)
                else:
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['K']= [K]

                if 'q' in path_graphs[k].get_edge_data(arc[0][0],arc[1][0]):
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['q'].append(arcs[k,arc[0],arc[1]])
                else:
                    path_graphs[k].get_edge_data(arc[0][0],arc[1][0])['q']= [arcs[k,arc[0],arc[1]]]


        for k,PG in enumerate(path_graphs):
            if [d for n1,n2,d in PG.edges(data=True) if 'K' not in d]:
                print("Missing")

        #return paths, {k:v for k,v in cons.items() if len(v) > 0}
        consolidations: defaultdict[tuple[int, int], list[frozenset[int]]] = defaultdict(list)
        for a,c in cons.items():
            consolidations[a[0][0],a[1][0]].append(c)  # may have multiple consolidations for same arc (at different times)

        return path_graphs, consolidations




    ##
    ## Builds a graph from the solution (paths & consolidations)
    ##
    def get_network_solution(self, paths=None, cons: list[SolutionGraphConsolidation] | None = None):
        solution = TypedDiGraph[SolutionGraphNode]()

        if paths is None:
            paths = self.solution_paths
            assert paths is not None

        if cons is None:
            cons = [SolutionGraphConsolidation(c, frozenset(k)) for c,K in self.consolidations.items() for k in K if len(k) > 1]

        CD = dict((k,list(map(itemgetter(1), g))) for k,g in itertools.groupby(sorted(cons), itemgetter(0)))

        solution.add_nodes_from(c for c in cons)
        solution.add_nodes_from(SolutionGraphCommodity(k,n) for k,P in enumerate(paths) for n in P.nodes())

        solution.add_weighted_edges_from((SolutionGraphCommodity(k,a[0]), SolutionGraphCommodity(k,a[1]), self.transit(*a)) 
                                            for k,P in enumerate(paths) for a in P.edges() if not (a in CD and any([k in K for K in CD[a]])))

        solution.add_edges_from((SolutionGraphCommodity(k,c[0][0]), c, {}) for c in cons for k in c[1])
        solution.add_weighted_edges_from((c, SolutionGraphCommodity(k,c[0][1]), self.transit(*c[0])) for c in cons for k in c[1])

        # Time windows
        tw = list(map(lambda k: self.time_windows(k,paths[k]), range(len(self.commodities))))

        for n in solution.nodes():
            if not isinstance(n, SolutionGraphCommodity):
                solution.node_data(n)['tw'] = (max(tw[k].nodes[n[0][0]]['early'] for k in n[1]), min(tw[k].nodes[n[0][0]]['late'] for k in n[1]))
            else:
                solution.node_data(n)['tw'] = (tw[n[0]].nodes[n[1]]['early'], tw[n[0]].nodes[n[1]]['late'])# tw[n[0]][n[1]]

        cycle = []
        
        # update early/late
        for G_ in nx.weakly_connected_components(solution.G):
            has_cycle = False
            G = solution.subgraph(G_)

            for n in nx.dfs_postorder_nodes(G.G):
                coll = list(solution.out_edges(n))

                if len(coll) > 1:
                    # check for cycles
                    p = next((p[1] for p in coll if 'late' not in solution.node_data(p[1])), None)
                    if p:
                        cycle.append(nx.shortest_path(solution.G, p, n))
                        has_cycle = True
                        break 

                    # added support for split freight - a node can have multiple out arcs to consolidations or movements
                    solution.node_data(n)['late'] = round(min(solution.node_data(p[1])['late'] - (self.transit(n[1], p[1][1]) if is_node(n) and is_node(p[1]) else 0) for p in coll) - (self.transit(*n[0]) if not is_node(n) else 0), PRECISION)
                elif len(coll) == 1:
                    p = coll[0][1]

                    # check for cycles
                    if 'late' not in solution.node_data(p):
                        cycle.append(nx.shortest_path(solution.G, p, n))
                        has_cycle = True
                        break 

                    solution.node_data(n)['late'] = round(solution.node_data(p)['late'] - (self.transit(n[1], p[1]) if is_node(p) else 0), PRECISION)
                else: # at end
                    solution.node_data(n)['late'] = self.commodities[n[0]].b[1]

            if not has_cycle:
                ## copies graph to do reverse... can be done better?
                for n in nx.dfs_postorder_nodes(G.G.reverse()):
                    coll = list(solution.in_edges(n))

                    if len(coll) > 1:
                        # support splitting freight (multiple arcs could be split)
                        early = max(solution.node_data(p[0])['early'] + (self.transit((p[0][1] if isinstance(p[0], SolutionGraphCommodity) else p[0][0][0]), n[1]) if isinstance(n, SolutionGraphCommodity) else 0) for p in coll)
                        min_early = min(solution.node_data(p[0])['early'] + (self.transit((p[0][1] if isinstance(p[0], SolutionGraphCommodity) else p[0][0][0]), n[1]) if isinstance(n, SolutionGraphCommodity) else 0) for p in coll)

                        solution.node_data(n)['early'] = round(early, PRECISION)
                        solution.node_data(n)['diff'] = round(early - min_early, PRECISION)
                    elif len(coll) == 1:
                        p = coll[0][0]
                        solution.node_data(n)['early'] = round(solution.node_data(p)['early'] + self.transit(p[1] if isinstance(p, SolutionGraphCommodity) else p[0][0], n[1]), PRECISION)
                    else: # at end
                        solution.node_data(n)['early'] = self.commodities[n[0]].a[1]

                    solution.node_data(n)['valid'] = solution.node_data(n)['early'] <= solution.node_data(n)['late']

        #self.solve_dual_solution(None, cons)

        return solution, cycle


    # returns the min/max time window for a commodity that travels along arc a
    def time_windows(self, k, path=None):
        t1 = self.commodities[k].a[1]
        t2 = self.commodities[k].b[1]

        if path is None:
            path = self.solution_paths[k]
        #paths = list(pairwise(path))

        path.nodes[self.commodities[k].a[0]]['early'] = t1
        path.nodes[self.commodities[k].b[0]]['late'] = t2
        #try:
        for n in nx.dfs_postorder_nodes(path):
            for a in path.out_edges(n):
                late = round(path.nodes[a[1]]['late'] - self.transit(*a), PRECISION)

                if 'late' not in path.nodes[a[0]] or path.nodes[a[0]]['late'] < late:
                    path.nodes[a[0]]['late'] = late

        for n in nx.dfs_postorder_nodes(path.reverse()):
            for a in path.out_edges(n):
                early = round(path.nodes[n]['early'] + self.transit(*a), PRECISION)

                if 'early' not in path.nodes[a[1]] or path.nodes[a[1]]['early'] > early:
                    path.nodes[a[1]]['early'] = early
        #except:
        #    print("ERROR")
            
        #    for n in nx.dfs_postorder_nodes(path):
        #        for a in path.out_edges(n):
        #            late = round(path.nodes[a[1]]['late'] - self.transit(*a), PRECISION)

        #            if 'late' not in path.nodes[a[0]] or path.nodes[a[0]]['late'] < late:
        #                path.nodes[a[0]]['late'] = late

        #    for n in nx.dfs_postorder_nodes(path.reverse()):
        #        for a in path.out_edges(n):
        #            early = round(path.nodes[n]['early'] + self.transit(*a), PRECISION)

        #            if 'early' not in path.nodes[a[1]] or path.nodes[a[1]]['early'] > early:
        #                path.nodes[a[1]]['early'] = early


        return path
        #early = [0] + list(accumulate((self.transit(*n) for n in paths)))
        #late  = [0] + list(accumulate((self.transit(*n) for n in reversed(paths))))

        #return {p: round_tuple((float(t1 + e), float(t2 - l))) for (p,e,l) in zip(path, early, reversed(late))}


    ## get the transit time between 2 nodes (looks nicer than direct access)
    def transit(self,n1,n2):
        return self.network.edge_data(n1, n2)['weight']

    def shortest_path(self,k,n1,n2):
        return self.commodity_shortest_paths[k][n1].get(n2, None) if self.commodity_shortest_paths else self.shared_shortest_paths[n1].get(n2, None)

    def shortest_paths(self,k,n1):
        return self.commodity_shortest_paths[k][n1] if self.commodity_shortest_paths else self.shared_shortest_paths[n1]

    def create_shortest_paths(self, opt=shortest_path_option.shared):
        self.shared_shortest_paths = {a[0]: a[1] for a in nx.shortest_path_length(self.network.G, weight='weight')} # nx.shortest_path_length(self.network, weight='weight')
        self.commodity_shortest_paths = []
        self.edge_shortest_path = {}

        if opt == shortest_path_option.commodity:
            # create shortest paths excluding destination node from calculations
            for k,c in enumerate(self.commodities):
                network_copy = self.network.copy()
                network_copy.remove_node(c.b[0]) # remove destination
                network_copy.remove_node(c.a[0]) # remove origin

                path = nx.shortest_path_length(network_copy.G, weight='weight')

                # merge destination path with all other paths
                # cluster server uses networkx version 1.7 which does not like a target without a source
                #dest = nx.shortest_path_length(self.network, source=None, target=c['b'][0], weight='weight')
                #for n,t in dest.items():
                #    if n in path:
                #        path[n][c['b'][0]] = t

                for n in self.network.nodes():
                    if n in path:
                        path[n][c.b[0]] = nx.shortest_path_length(self.network.G, source=n, target=c.b[0], weight='weight')

                network_copy = self.network.copy()
                network_copy.remove_node(c.a[0]) # remove origin

                path[c.b[0]] = nx.shortest_path_length(network_copy.G, source=c.b[0], weight='weight')

                network_copy = self.network.copy()
                network_copy.remove_node(c.b[0]) # remove destination

                path[c.a[0]] = nx.shortest_path_length(network_copy.G, source=c.a[0], weight='weight')
                path[c.a[0]][c.b[0]] = self.shared_shortest_paths[c.a[0]][c.b[0]]

                self.commodity_shortest_paths.append(path)

        # create shortest paths excluding n1 node on arc
        if opt == shortest_path_option.edges:
            for i in self.network.nodes():
                for j in self.network.nodes():
                    if i <= j:
                        network_copy = self.network.copy()
                        network_copy.remove_node(i)

                        if i != j:
                            network_copy.remove_node(j)

                        #for e in self.network.out_edges(n):
                        for n2 in self.network.nodes():
                            if i != n2 and j != n2:
                                self.edge_shortest_path[((i,j),n2)] = nx.shortest_path_length(network_copy.G, n2, weight='weight')

    # gets the tightened shortest path
    def restricted_shortest_path(self, reject_nodes: tuple[int, int], arc: tuple[int, int]):
        if reject_nodes[0] > reject_nodes[1]:
             reject_nodes = (reject_nodes[1],reject_nodes[0])

        return self.edge_shortest_path[reject_nodes, arc[0]].get(arc[1], None) if arc[0] not in reject_nodes and arc[1] not in reject_nodes else None

    ## creates arcs/nodes for time horizon
    def trivial_network(self):
        return tuplelist((n,t) for n in self.network.nodes() for t in [self.S,self.T])

    ##
    ## Arc validation
    ##
    
    def is_valid_storage_arc(self, arc: TimedArc, origin: NodeTime, dest: NodeTime, origin_to_arc: float | None, arc_to_dest: float | None):
        return (arc[0] is not None and arc[1] is not None and origin_to_arc is not None and arc_to_dest is not None and     # 1. is valid node and path
               arc[0][2] == arc[1][1] and                                                                                   # 2. arc is consectutive
               origin[1] + origin_to_arc < min(arc[0][2], arc[1][2]) and                                                    # 3. can reach this arc using shortest paths
               max(origin[1] + origin_to_arc, arc[0][1], arc[1][1]) + arc_to_dest <= dest[1])                               # 4. can reach destination in time

    def is_arc_valid(self, arc: TimedArc, origin: NodeTime, dest: NodeTime, origin_to_arc: float | None, arc_to_arc: float, arc_to_dest: float | None):
        return (arc[0] is not None and arc[1] is not None and origin_to_arc is not None and arc_to_dest is not None and     # 1. is valid node and path
               arc[1][0] != origin[0] and arc[0][0] != dest[0] and                                                          # 2. no inflow into origin, nor outflow from destination
               origin[1] + origin_to_arc < min(arc[0][2], arc[1][2] - arc_to_arc) and                                       # 3. can reach this arc using shortest paths
               max(origin[1] + origin_to_arc + arc_to_arc, arc[0][1] + arc_to_arc, arc[1][1]) + arc_to_dest <= dest[1] and  # 4. can reach destination in time
               arc[1][1] - arc[0][2] < arc_to_arc < arc[1][2] - arc[0][1])                                                  # 5. transit time within interval is valid?
    
    ##
    ## Validate any arc
    def V(self, k: int, a: TimedArc):
        origin, dest = self.commodities[k].a, self.commodities[k].b

        if a[0][0] != a[1][0]:
            origin_to_arc = self.restricted_shortest_path((dest.node, a.target.node), (origin.node, a.source.node)) 
            arc_to_dest = self.restricted_shortest_path((origin.node, a.source.node), (a.target.node, dest.node)) 

            return self.is_arc_valid(a, origin, dest, origin_to_arc, self.network.edge_data(a[0][0], a[1][0])['weight'], arc_to_dest)
        else:
            return self.is_valid_storage_arc(a, origin, dest, self.shortest_path(k, origin[0], a[0][0]), self.shortest_path(k, a[0][0], dest[0]))

    ##
    ## Add new timepoints to the system
    def add_network_timepoints(self, new_timepoints: set[NodeTime]):
        if not new_timepoints:
            return

        #self.initial_timepoints.update(new_timepoints)
        new_arcs: list[dict[TimedArc, Var]] = [dict() for k in range(len(self.commodities))]

        ## Update Graph
        ##
        for n,t in new_timepoints:
            # find current interval that gets split by new timepoint - should always succeed if time >= 0 and time <= T
            i0 = next((i for i in self.intervals.select(n, '*', '*') if i[1] < t < i[2]), None)

            # ignore timepoints that are already in the system
            if i0 is None:
                continue

            # split interval
            i1,i2 = NodeInterval(n, i0[1], t), NodeInterval(n, t, i0[2])
            self.intervals.remove(i0)
            self.intervals.extend([i1.T, i2.T])

            # update consolidation nodes (i0 -> i1)
            i0 = NodeInterval(*i0)
            self.cons_network.add_node(i1, *self.cons_network.node_data(i0))
            self.cons_network.add_node(i2)

            i1_out_edges = [(i1, target, data) for (_,target,data) in self.cons_network.out_edges_data(i0)]
            i1_in_edges = [(source, i1, data) for (source,_,data) in self.cons_network.in_edges_data(i0)]

            # rename variables/constraints
            for a,a0,d in itertools.chain((((a1,a2),(i0,a2),d) for a1,a2,d in i1_out_edges), (((a1,a2),(a1,i0),d) for a1,a2,d in i1_in_edges)):
                if 'z' in d and d['z'] is not None:
                    d['z'].VarName='z' + str(a)

                    self.constraint_consolidation[a] = self.constraint_consolidation.pop(a0)
                    self.constraint_consolidation[a].ConstrName = 'cons' + str(a)

            self.cons_network.remove_node(i0)
            self.cons_network.add_edges_from(i1_out_edges + i1_in_edges)

            # update all commodity networks
            for k,G in enumerate(self.timed_network):
                self.add_network_timepoint(k, n, i0, i1, i2)

        ## Update Model
        ##
        # Improve performance of model by only adding the first dispatch arc for the same set of commodities
        if self.ALGORITHM >= algorithm_option.reduced:
            for n in self.cons_network.nodes():
                for _, group in itertools.groupby(sorted(self.cons_network.G.edges(n, data=True)), lambda t: t[1][0]):
                    for a,b in pairwise(group):
                        if a[2]['K'] >= b[2]['K']:
                            if 'z' not in b[2]:
                                b[2]['z'] = None
                        elif 'z' in b[2] and b[2]['z'] is None:
                            del b[2]['z']

        for a1,a2,d in self.cons_network.edges_data():
            if 'z' not in d:
                d['z'] = self.model.addVar(obj=(self.network.edge_data(a1[0],a2[0])['fixed_cost']), lb=0, 
                                           ub=self.model.inf(), 
                                           name='z' + str((a1,a2)), type=self.model.integer())

        for k,G in enumerate(self.timed_network):
            for a1,a2,d in G.edges_data():
                if 'x' not in d and (a1[0] == a2[0] or self.cons_network.edge_data(a1,a2)['z'] is not None):
                    d['x'] = self.model.addVar(obj=(self.problem.var_cost[k].get((a1[0],a2[0]),0) * self.commodities[k].q if a1[0] != a2[0] else self.problem.var_cost[k].get((a1[0],a2[0]),0) * self.commodities[k].q * (a1[2] - a1[1])), lb=0, ub=1, type=self.model.binary() if not ALLOW_SPLIT else self.model.continuous(), name='x' + str(k) + ',' + str((a1,a2)))
                    new_arcs[k][TimedArc(a1,a2)] = d['x']

                    # holding arc - here we assume precision of system!  i.e. full discretization of 1 is considered continuous-time optimal
                    if a1[0] == a2[0] and a1[2] > a1[1] + 1:
                        d['y'] = self.model.addVar(obj=(-self.problem.var_cost[k].get((a1[0],a2[0]),0) * self.commodities[k].q * (a1[2] - a1[1])), lb=0, ub=1, type=self.model.binary() if not ALLOW_SPLIT else self.model.continuous(), name='y' + str(k) + ',' + str((a1,a2)))



        self.model.update()  # add variables to model
        self.update_constraints(new_arcs)
      #  self.user_cuts()

        #self.model.update()
        #self.model.write('test.lp')


    # modify the graph to insert a time point
    def add_network_timepoint(self, k: int, n, i0: NodeInterval, i1: NodeInterval, i2: NodeInterval):
        origin,dest = self.commodities[k].a, self.commodities[k].b
        v = partial(self.V, k)

        # if splitting the origin/destination
        if i0 == self.origin_destination[k][0]:
            self.origin_destination[k] = TimedArc(i1[1] <= origin[1] < i1[2] and i1 or i2, self.origin_destination[k][1])
        elif i0 == self.origin_destination[k][1]:
            self.origin_destination[k] = TimedArc(self.origin_destination[k][0], i1[1] <= dest[1] < i1[2] and i1 or i2)

        # Add new node
        G = self.timed_network[k]
        G.add_node(i2)
        new_edges = v(TimedArc(i1,i2)) and [(i1,i2,{})] or []

        # Relabel nodes (i0 -> i1)
        G.add_node(i1, *G.node_data(i0))
        i1_out_edges = [(i1, target, data) for (_,target,data) in G.out_edges_data(i0)]
        i1_in_edges = [(source, i1, data) for (source,_,data) in G.in_edges_data(i0)]
        G.remove_node(i0)

        # Copy appropriate edges from i1 (out / in)
        i2_edges = [(i2,e2,{}) for e1,e2,d in i1_out_edges if v(TimedArc(i2,e2))] + [(e1,i2,{}) for e1,e2,d in i1_in_edges if v(TimedArc(e1,i2))]
        new_edges.extend(i2_edges)

        # Remove invalid edges from i1, Keep good edges
        del_edges, keep_edges = partition(lambda t: v(TimedArc(t[0],t[1])), i1_out_edges + i1_in_edges)

        new_edges.extend(keep_edges)
        G.add_edges_from(new_edges)

        ## Update consolidation network
        for a1,a2,d in i2_edges:
            if a1[0] != a2[0]:
                if self.cons_network.has_edge(a1,a2):
                    self.cons_network.edge_data(a1,a2)['K'].add(k)
                else:
                    self.cons_network.add_edge(a1,a2, K=set([k]))

        ## Update Model
        ##

        ## Rename flow interval
        if (k,i0) in self.constraint_flow:
            self.constraint_flow[(k,i1)] = self.constraint_flow.pop((k,i0))
            self.constraint_flow[(k,i1)].ConstrName = 'flow' + str((k,i1))

        # Check if i2 interval is origin/destination interval (if i1 is, we are renaming so no change)
        if i2 in self.origin_destination[k] and (k,i1) in self.constraint_flow:
            self.model.set_rhs(self.constraint_flow[(k,i1)], 0) 

        # rename arcs
        for a,a0,d in itertools.chain((((a1,a2),(i0,a2),d) for a1,a2,d in G.out_edges_data(i1)), (((a1,a2),(a1,i0),d) for a1,a2,d in G.in_edges_data(i1))):
            if 'x' in d:
                d['x'].VarName = 'x' + str(k) + ',' + str(a)

            if 'y' in d:
                d['y'].VarName = 'y' + str(k) + ',' + str(a)


        ## Delete old arcs
        for a1,a2,d in del_edges:
            if 'x' in d:
                x = d['x']
                #self.model.setAttr("UB", [x], [0])  # remove the arc by setting UB to 0

                self.model.chgCoeff(self.constraint_flow[(k,a1)], x, 0)
                self.model.chgCoeff(self.constraint_flow[(k,a2)], x, 0)

                if PATH_CUTS:
                    self.model.chgCoeff(self.constraint_path_length[k], x, 0)

                # in-tree
                if IN_TREE and a1[0] != a2[0]:
                    self.model.chgCoeff(self.constraints_intree[k,a1[0],a2[0]], x, 0)

                self.model.removeVar(x)

            # holding costs
            if 'y' in d:
                if (k,a1) in self.constraints_holding_offset:
                    self.model.chgCoeff(self.constraints_holding_offset[(k,a1)], d['y'], 0)
                self.model.removeVar(d['y'])


            # dispatch arc
            if a1[0] != a2[0] and self.cons_network.has_edge(a1,a2):
                attr = self.cons_network.edge_data(a1,a2)
                attr['K'].remove(k)

                # removed all consolidation - remove variable
                if not attr['K']:
                    self.cons_network.remove_edge(a1,a2)

                    if 'z' in attr and attr['z'] is not None:
                        #self.model.setAttr("UB", [attr['z']], [0])
                        self.model.removeVar(attr['z'])
                        self.model.removeCons(self.constraint_consolidation.pop((a1,a2)))
                #elif 'z' in attr:
                #    self.model.chgCoeff(attr['z'], d['x'], 0)


    # adds a new variable, inserts itself into appropriate constraints
    def update_constraints(self, new_arcs: list[dict[TimedArc, Var]]):
        chg_coeff = []

        # update constraints
        for k,arcs in enumerate(new_arcs):
            for a,x in arcs.items():
                # arc is a dispatch
                if a.source.node != a.target.node:
                    # consolidation
                    if a not in self.constraint_consolidation:
                        self.constraint_consolidation[a] = self.model.addConstr(x * self.commodities[k].q <= self.cons_network.edge_data(a.source,a.target)['z'] * self.network.edge_data(a.source.node,a.target.node)['capacity'], 'cons' + str(a))
                    else:
                        chg_coeff.append((self.constraint_consolidation[a], x, self.commodities[k].q))

                    # cycle
                    if self.constraint_cycle is not None:
                        if (k,a[0][0]) not in self.constraint_cycle:
                            self.constraint_cycle[(k,a[0][0])] = self.model.addConstr(x <= 1, 'cycle')
                        else:
                            chg_coeff.append((self.constraint_cycle[(k,a[0][0])], x, 1))

                    # path length
                    if PATH_CUTS:
                        chg_coeff.append((self.constraint_path_length[k], x, self.transit(a[0][0],a[1][0])))

                    # in-tree
                    if IN_TREE:
                        chg_coeff.append((self.constraints_intree[k,a[0][0],a[1][0]], x, 1))

                    # user_cuts
                    if 'z' in self.cons_network.edge_data(a[0],a[1]) and self.cons_network.edge_data(a[0],a[1])['z'] is not None:
                        if USER_CUTS:
                            self.constraints_user.append(self.model.addConstr(self.cons_network.edge_data(a[0],a[1])['z'] >= ceil(self.commodities[k].q/self.network.edge_data(a[0][0],a[1][0])['capacity']) * x, 'user'))

                        if ORIGIN_CUTS:
                            if a[0][0] == self.commodities[k].a[0]:
                                chg_coeff.append((self.constraints_origin[k], self.cons_network.edge_data(a[0],a[1])['z'], 1))

                            if a[1][0] == self.commodities[k].b[0]:
                                chg_coeff.append((self.constraints_dest[k], self.cons_network.edge_data(a[0],a[1])['z'], 1))

                # inflow
                if (k,a[1]) not in self.constraint_flow:
                    self.constraint_flow[(k,a[1])] = self.model.addConstr(x == self.r(k,a[1]), 'flow' + str((k,a[1])))
                else:
                    chg_coeff.append((self.constraint_flow[(k,a[1])], x, 1))

                # outflow
                if (k,a[0]) not in self.constraint_flow:
                    self.constraint_flow[(k,a[0])] = self.model.addConstr(-1.0 * x == self.r(k,a[0]), 'flow' + str((k,a[0])))
                else:
                    chg_coeff.append((self.constraint_flow[(k,a[0])], x, -1))


                # holding costs!
                if (k,a[0]) not in self.constraints_holding_offset:
                    i = [d['x'] for a1,a2,d in self.timed_network[k].in_edges_data(a[0]) if 'x' in d and a1[0] != a2[0]]
                    o = [(d['x'], d['y']) for a1,a2,d in self.timed_network[k].out_edges_data(a[0]) if 'x' in d  and 'y' in d and a1[0] == a2[0]] # get holding arc out

                    if i and o:
                        self.constraints_holding_offset[k,a[0]] = self.model.addConstr(1 + o[0][1] >= quicksum(i) + o[0][0], 'holding-offset')
                        self.constraints_holding_enforce[k,a[0]] = self.model.addConstr(o[0][1] <= o[0][0], 'holding-enforce')
                        self.constraints_holding_enforce2[k,a[0]] = self.model.addConstr(o[0][1] <= quicksum(i), 'holding-enforce')
                else:
                    pass # not sure what to do here?

        # minimize the number of updates
        self.model.update()

        for c,x,q in chg_coeff:
            self.model.chgCoeff(c, x, q)

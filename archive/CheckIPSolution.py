from gurobipy import *
import itertools

import time

try:
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

class CheckIPSolution(object):
    """description of class"""
    __slots__ = ['problem', 'model', 'network', 'S', 'T', 'x', 'z', 'commodities', 'intervals', 'arcs', 'origin_destination', 'constraint_consolidation', 'constraint_flow', 'constraint_cycle', 'constraint_path_length', 'solution_paths','consolidations', 'fixed_timepoints_model', 'initial_timepoints', 'incumbent', 'lower_bound']

    def __init__(self, problem, solution_paths):
        self.problem = problem
        self.commodities = problem.commodities
        self.solution_paths = solution_paths

        self.S = min(c['a'][1] for c in self.commodities)  # time horizon
        self.T = max(c['b'][1] for c in self.commodities) + 1  # time horizon

        self.incumbent = None  # store the lowest upper bound
        self.lower_bound = None
        shouldEnforceCycles = False

        # build graph
        self.network = nx.DiGraph()

        for a, destinations in problem.network.items():
            for b, transit_time in destinations.items():
                shouldEnforceCycles = shouldEnforceCycles or problem.var_cost.get((a,b),0) == 0  # if all arcs have positive costs then we don't need to add extra constraints
                self.network.add_edge(a, b, weight=transit_time, capacity=problem.capacities.get((a,b), 1.0), fixed_cost=problem.fixed_cost.get((a,b), transit_time), var_cost=problem.var_cost.get((a,b), 0))

        if self.infeasible():
            return

        # create initial intervals/arcs
        t0 = time.time()
        self.arcs = self.create_arcs()
        t1 = time.time()
        print "Create arcs", t1 - t0
        
        # find which interval contains the origin/destination
        self.origin_destination = {k: ((c['a'][0], c['a'][1],c['a'][1]+1), (c['b'][0], c['b'][1],c['b'][1]+1)) for k,c in enumerate(self.commodities)}

        ##
        ## Construct Gurobi model
        ## 
        model = self.model = Model("IMCFCNF", env=Env(""))
        model.modelSense = GRB.MINIMIZE

        model.setParam('OutputFlag', False)

        model.setParam(GRB.param.MIPGap, 0.01)  # 1% gap
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

        t0 = time.time()
        print "Model update", t0 - t1

        ##
        ## Constraints
        ## 

        # flow constraints
        self.constraint_flow = {}

        for k in range(K):
            # get distinct list of intervals from arcs
            for n in set(itertools.chain(*itertools.izip(*itertools.chain(self.arcs[k])))):
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

        t1 = time.time()
        print "Constraints", t1 - t0

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
        return len([c for k,c in enumerate(self.commodities) if c['a'][1] + sum(self.transit(*a) for a in pairwise(self.solution_paths[k])) > c['b'][1]]) > 0

    ##
    ## Iterative Solve
    ##
    def solve(self, draw_filename='', start_time=time.time()):
        # feasibility check: shortest path cannot reach destination - gurobi doesn't pick it up, because no arcs exist
        if self.infeasible():
            print "INFEASIBLE"
            return 0, 0, 0

        solve_time = 0
        t0 = time.time()
        self.solve_lower_bound()  # Solve lower bound problem
        solve_time += time.time() - t0

        # return if not optimal
        if self.model.status != GRB.status.OPTIMAL:
            print "INFEASIBLE"
            return solve_time, None

        self.consolidations = self.get_consolidations()

        return solve_time, self.incumbent

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
            print '\nCost:', model.objVal

            print "Path:", self.solution_paths

            print "Consolidation: "
            for c,v in self.get_consolidations().items():
                print(v)
        else:
            print('No solution')

    # get statistics
    def get_statistics(self):
        stats = {}
        stats['cost'] = self.model.objVal if self.model.status == GRB.status.OPTIMAL and self.incumbent == None else self.incumbent
        stats['nodes'] = len(self.network.nodes())
        stats['arcs'] = len(self.x)
        stats['variables'] = len(self.model.getVars())
        stats['paths'] = self.solution_paths
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

        return stats

    ##
    ## From the solution, return the commodities that are consolidated together
    ##
    def get_consolidations(self):
        # ignore cycles that are not apart of actual path - can occur when x has 0 cost
        return {a: [k for k in range(len(self.commodities)) if (k,a) in self.x and self.x[k,a].x > 0 and a[0][0] in self.solution_paths[k]] 
                    for a, za in self.z.items() if za.x > 0} if self.model.status == GRB.status.OPTIMAL else None

    ## get the transit time between 2 nodes (looks nicer than direct access)
    def transit(self,n1,n2):
        return self.network[n1][n2]['weight']

    ##
    ## Creates valid arcs (using lower bound) for each commodity and then shares 'redirected' arcs across all commodities
    ##
    def create_arcs(self):
        arcs = {}

        # Create arcs ((n1, t1, t2), (n2, t3, t4)) pairs
        for k,c in enumerate(self.commodities):
            origin,dest = c['a'], c['b']

            # setup storage arcs
            window = self.time_window(k)

            arcs[k] = tuplelist(((n,i1[0],i1[1]),(n,i1[1],i1[1]+1)) for n in self.solution_paths[k]
                                    for i1 in pairwise(range(window[n][0], window[n][1] + 1)))

            # for each physical arc
            arcs[k].extend(((a[0],i1[0],i1[1]),(a[1],i2[0],i2[1])) for a in pairwise(self.solution_paths[k])
                        for i1,i2 in itertools.izip(pairwise(range(window[a[0]][0], window[a[0]][1] + 2)), pairwise(range(window[a[1]][0], window[a[1]][1] + 2))))

        return arcs
    
    # returns the min/max time window for a commodity that travels along arc a
    def time_window(self, k):
        t1 = self.commodities[k]['a'][1]
        t2 = self.commodities[k]['b'][1]

        early = [0] + list(accumulate((self.transit(*n) for n in pairwise(self.solution_paths[k]))))
        late  = [0] + list(accumulate((self.transit(*n) for n in reversed(list(pairwise(self.solution_paths[k]))))))

        return {self.solution_paths[k][i]: (int(t1 + e), int(t2 - l)) for i,(e,l) in enumerate(zip(early, reversed(late)))}




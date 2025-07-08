from gurobipy import *
import networkx as nx
import functools
import itertools
from tools import pairwise
from operator import itemgetter
import math

class VerifySolution(object):
    """Verifies if a solution is implementable"""
    __slots__ = ['solution_paths', 'consolidations', 'model', 't', 's', 'problem']

    def __init__(self, problem):
        self.problem = problem

    def validate(self):
        if self.problem.solution == None:
            return False

        lp = self.model = Model("implementable")
        lp.setParam('OutputFlag', False)
        lp.modelSense = GRB.MINIMIZE

        solution_paths = self.problem.solution[1]
        solution_cons = self.problem.solution[2]

        # t - dispatch time at each node in path {k,n}
        self.t = {(k,n): lp.addVar(obj=0, lb=0, ub=GRB.INFINITY, name='t' + str(k) + ',' + str(n)) 
                          for k, path in enumerate(solution_paths)
                            for n in path}

        # s - slack variables for consolidation s[a,{K}]
        self.s = {c: lp.addVar(obj=0, lb=0, ub=GRB.INFINITY, name='s' + str(c))
                        for c in solution_cons}

        lp.update()

        # Constraints
        for k, path in enumerate(solution_paths):
            # origin dispatch time >= origin time
            lp.addConstr(self.t[k, path[0]] >= self.problem.commodities[k]['a'][1], 'L' + str(k) + ',' + str(path[0]))

            # destination dispatch time <= destination time
            lp.addConstr(self.t[k, path[-1]] <= self.problem.commodities[k]['b'][1], 'U' + str(k) + ',' + str(path[-1]))
            
            # dispatch time >= last dispatch + transit time along path
            for n1,n2 in zip(path,path[1:]):
                lp.addConstr(self.t[k,n2] >= self.t[k,n1] + self.problem.network[n1][n2], 'L' + str(k) + ',' + str(n1))
    
        # consolidating dispatch time is equal + slack variable
        for a,K in solution_cons:
            for k in K:
                lp.addConstr(self.s[a,K] == self.t[k,a[0]], 'a' + str(a) + ',' + str(K))

        lp.update()
        lp.optimize()

        return lp.status == GRB.status.OPTIMAL and self.problem.solution[0] == self.get_solution_cost()

    def get_solution_times(self):
        if not self.t:
            if not self.validate():
                return None

        # node lookup dispatch time
        times_map = [None] * len(self.problem.commodities)
        for (k,n),x in self.t.items():
            if times_map[k] == None:
                times_map[k] = {}

            times_map[k][n] = x.x

        # dispatch time in order
        return [sorted(M.values()) for M in times_map]


    # get new consolidations and calculate new solution cost
    def get_solution_cost(self):
        consolidation = {}
        var_cost = 0

        for k, path in enumerate(self.problem.solution[1]):
            for n1,n2 in zip(path, path[1:]):
                a = (n1,n2,self.t[k,n1].x)
                consolidation[a] = consolidation.get(a, 0) + self.problem.commodities[k]['q']
                var_cost += self.problem.var_cost[n1,n2] * self.problem.commodities[k]['q']

        return var_cost + sum(self.problem.fixed_cost[a,b] * math.ceil(q/self.problem.capacities[a,b]) for (a,b,t), q in consolidation.items())


    def check_gap(self):
        if self.problem.solution == None or not self.validate():
            return False

        S = int(min(c['a'][1] for c in self.problem.commodities))  # time horizon
        T = int(max(c['b'][1] for c in self.problem.commodities)) + 1  # time horizon

        shouldEnforceCycles = False

        # build graph
        network = nx.DiGraph()

        for a, destinations in self.problem.network.items():
            for b, transit_time in destinations.items():
                shouldEnforceCycles = shouldEnforceCycles or self.problem.var_cost.get((a,b),0) == 0  # if all arcs have positive costs then we don't need to add extra constraints
                network.add_edge(a, b, weight=transit_time, capacity=self.problem.capacities.get((a,b), 1.0), fixed_cost=self.problem.fixed_cost.get((a,b), transit_time), var_cost=self.problem.var_cost.get((a,b), 0))

        #if self.infeasible():
        #    return

        shortest_path = nx.shortest_path_length(network, weight='weight')
        edge_shortest_path = {}

        # create shortest paths excluding n1 node on arc
        for n in network:
            network_copy = network.copy()
            network_copy.remove_node(n)

            for n2 in network:
                if n != n2:
                    edge_shortest_path[(n,n2)] = nx.shortest_path_length(network_copy, n2, weight='weight')

        ##
        ## Validate any arc
        def v(c, arc):
            origin,dest = c['a'], c['b']

            if arc[0][0] != arc[1][0]:
                origin_to_arc = edge_shortest_path[(dest[0],origin[0])].get(arc[0][0], None)
                arc_to_dest = edge_shortest_path[(arc[0][0],arc[1][0])].get(dest[0], None)
                arc_to_arc = network[arc[0][0]][arc[1][0]]['weight']

                return (arc[0] != None and arc[1] != None and origin_to_arc != None and arc_to_dest != None and                     # 1. is valid node and path
                       arc[1][0] != origin[0] and arc[0][0] != dest[0] and                                                          # 2. no inflow into origin, nor outflow from destination
                       origin[1] + origin_to_arc < min(arc[0][2], arc[1][2] - arc_to_arc) and                                       # 3. can reach this arc using shortest paths
                       max(origin[1] + origin_to_arc + arc_to_arc, arc[0][1] + arc_to_arc, arc[1][1]) + arc_to_dest <= dest[1] and  # 4. can reach destination in time
                       arc[1][1] - arc[0][2] < arc_to_arc < arc[1][2] - arc[0][1])                                                  # 5. transit time within interval is valid?
            else:
                origin_to_arc = shortest_path[origin[0]].get(arc[0][0])
                arc_to_dest = shortest_path[arc[0][0]].get(dest[0])

                return (arc[0] != None and arc[1] != None and origin_to_arc != None and arc_to_dest != None and     # 1. is valid node and path
                       arc[0][2] == arc[1][1] and                                                                   # 2. arc is consectutive
                       origin[1] + origin_to_arc < min(arc[0][2], arc[1][2]) and                                    # 3. can reach this arc using shortest paths
                       max(origin[1] + origin_to_arc, arc[0][1], arc[1][1]) + arc_to_dest <= dest[1])               # 4. can reach destination in time


        ##
        ## Construct Gurobi model
        ## 
        model = Model("IMCFCNF", env=Env(""))
        model.modelSense = GRB.MINIMIZE

        model.setParam(GRB.param.MIPGap, 0.01)  # 1% gap
        model.setParam(GRB.param.TimeLimit, 14400) # 4hr limit
        #model.setParam(GRB.param.MIPFocus, 2) # focus on proving optimality

        # create intervals/arcs
        timed_network = []
        all_arcs = []

        arcs = [((e[0],i, i+1), (e[1], i + int(network[e[0]][e[1]]['weight']), i + int(network[e[0]][e[1]]['weight']) + 1)) for e in network.edges() for i in range(S,T)]

        for k,c in enumerate(self.problem.commodities):
            G = nx.DiGraph()

            # holding arcs
            for n in network:
                for i in range(S,T):
                    a = ((n, i, i+1), (n, i+1, i+2))

                    if v(c,a):
                        G.add_edge(a[0], a[1], {'x': model.addVar(obj=(network[a[0][0]][a[1][0]]['var_cost'] * c['q'] if a[0][0] != a[1][0] else 0), lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(k) + ',' + str(a)) })

            # dispatch arcs
            all_arcs.append(filter(functools.partial(v,c), arcs))

            G.add_edges_from(((a[0], a[1], {'x': model.addVar(obj=(network[a[0][0]][a[1][0]]['var_cost'] * c['q'] if a[0][0] != a[1][0] else 0), lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(k) + ',' + str(a)) })
                                for a in all_arcs[k]))

            timed_network.append(G)

        cons_network = nx.DiGraph()

        cons = itertools.groupby(sorted((a,k) for k,arcs in enumerate(all_arcs) for a in arcs), itemgetter(0))
        cons_network.add_edges_from(((a[0],a[1],{'z': model.addVar(obj=(network[a[0][0]][a[1][0]]['fixed_cost']), lb=0, ub=GRB.INFINITY, name='z' + str(a), vtype=GRB.INTEGER), 
                                                      'K': set(map(itemgetter(1),K))}) 
                                          for a,K in cons))
        
        # find which interval contains the origin/destination
        origin_destination = {k: ((c['a'][0], c['a'][1],c['a'][1]+1), (c['b'][0], c['b'][1],c['b'][1]+1)) for k,c in enumerate(self.problem.commodities)}
                
        model.update()

        ##
        ## Constraints
        ## 
        # flow constraints
        constraint_flow = {}

        for k,G in enumerate(timed_network):
            for n in G:
                i = [d['x'] for a1,a2,d in G.in_edges(n, data=True) if 'x' in d]
                o = [d['x'] for a1,a2,d in G.out_edges(n, data=True) if 'x' in d]
                od = origin_destination[k]

                if i or o:
                    constraint_flow[(k,n)] = model.addConstr(quicksum(i) - quicksum(o) == (-1 if od[0] == n else (1 if od[1] == n else 0)), 'flow' + str((k,n)))

        # Consolidation constraints
        constraint_consolidation = {(a1,a2): model.addConstr(quicksum(timed_network[k][a1][a2]['x'] * self.problem.commodities[k]['q'] for k in d['K']) <= d['z'] * network[a1[0]][a2[0]]['capacity'], 'cons' + str((a1,a2))) 
                                         for a1,a2,d in cons_network.edges(data=True) if d['z'] != None }

        # Ensure no flat-cycles
        constraint_cycle = None

        if shouldEnforceCycles:
            constraint_cycle = {}
    
            for k,G in enumerate(timed_network):
                for n in network:
                    outflow = [d['x'] for a1,a2,d in G.edges(data=True) if a1[0] == n and a2[0] != n and 'x' in d]

                    if outflow:
                        constraint_cycle[(k,n)] = model.addConstr(quicksum(outflow) <= 1, 'cycle')

        model.update()


        ## clear solution
        for G in timed_network:
            for i1,i2,d in G.edges(data=True):
                d['x'].start = 0

        for i1,i2,d in cons_network.edges(data=True):
            d['z'].start = 0

        ## load solution
        CD = dict((a,map(itemgetter(1), g)) for a,g in itertools.groupby(sorted(self.problem.solution[2]), itemgetter(0)))

        for k,p in enumerate(self.problem.solution[1]):
            G = timed_network[k]

            # origin to first dispatch
            for t in range(int(origin_destination[k][0][1]), int(self.t[k, p[0]].x)):
                G[(p[0], t, t+1)][(p[0], t+1, t+2)]['x'].start = 1

            for e in pairwise(p):
                t1 = int(self.t[k, e[0]].x)
                t2 = int(self.t[k, e[0]].x + network[e[0]][e[1]]['weight'])

                # dispatch to dispatch
                G[(e[0], t1, t1+1)][(e[1], t2, t2 + 1)]['x'].start = 1

                # all single flows
                if e not in CD or not [K for K in CD[e] if k in K]:
                    cons_network[(e[0], t1, t1+1)][(e[1], t2, t2+1)]['z'].start = math.ceil(self.problem.commodities[k]['q'])

                # storage arcs
                for t in range(t2, int(self.t[k, e[1]].x)):
                    G[(e[1], t, t+1)][(e[1], t+1, t+2)]['x'].start = 1

            # storage to destination
            for t in range(t2, int(origin_destination[k][1][1])):
                G[(p[-1], t, t+1)][(p[-1], t+1, t+2)]['x'].start = 1

        # all consolidations
        for c in self.problem.solution[2]:
            t1 = int(self.s[c].x)
            t2 = int(self.s[c].x + network[c[0][0]][c[0][1]]['weight'])

            cons_network[(c[0][0], t1, t1+1)][(c[0][1], t2, t2+1)]['z'].start = math.ceil(sum(self.problem.commodities[k]['q'] for k in c[1]))

        model.optimize()

        return model.status == GRB.status.OPTIMAL


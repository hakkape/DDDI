from Solver import *
import itertools
from tools import *
import math
from collections import defaultdict

# Time precision
PRECISION = 2 # decimal places
FAST_CONSOLIDATIONS = True  # don't consider all pairwise consolidations in LP (i.e., |K|^2), instead only compare against one commodity (i.e., |K|)

class CheckSolution(object):
    """Takes the solution from a simplified network flow model and validates/corrects for original problem"""
    __slots__ = ['solution_paths', 'consolidations', 'model', 't', 'x', 'L', 'problem', 'h', 'environment']

    def __init__(self, problem, env=None):
        self.problem = problem
        self.environment = env

    def infeasible(self):
        return len([c for k,c in enumerate(self.problem.commodities) if c['a'][1] + self.problem.shortest_path(k,c['a'][0],c['b'][0]) > c['b'][1]]) > 0

    def validate(self, solution_paths, consolidations):
        self.solution_paths = solution_paths
        self.consolidations = consolidations

        if self.consolidations == None:
            return False

        lp = self.model = Solver(self.problem.model.solver, use_callback=False, env=self.environment)

        # dispatch time at each node in path-graph, multiple nodes for multiple dispatches
        t = self.t = [{(a[0],a[1],K): lp.addVar(obj=0, lb=0, ub=lp.inf(), name='t' + str((k,a,K))) 
                         for a,K_coll in self.consolidations.items()
                            for K in K_coll if k in K}
                                for k, path_graph in enumerate(self.solution_paths)]

        # slack variables for each consolidation
        if FAST_CONSOLIDATIONS:
                x = self.x = {(a,min(group),k2,group): lp.addVar(obj=self.problem.network[a[0]][a[1]]['fixed_cost']
                                           #+self.problem.network[a[0][0]][a[1][0]]['var_cost']*((self.problem.commodities[k1]['q'] + self.problem.commodities[k2]['q']) % 1)
                                           , lb=0, ub=lp.inf(), name='x' + str(a) + ',' + str((min(group),k2)))
                        for a, K_coll in self.consolidations.items()
                            for group in K_coll
                                for k2 in group if k2 > min(group)}
        else:
                x = self.x = {(a,k1,k2,group): lp.addVar(obj=self.problem.network[a[0]][a[1]]['fixed_cost'] 
                                                   #+self.problem.network[a[0][0]][a[1][0]]['var_cost']*((self.problem.commodities[k1]['q'] + self.problem.commodities[k2]['q']) % 1)
                                           , lb=0, ub=lp.inf(), name='x' + str(a) + ',' + str((k1,k2)))
                        for a, K_coll in self.consolidations.items()
                            for group in K_coll
                                for k1 in group for k2 in group if k1 < k2}


        ## slack variables for dispatch times (holding cost support)
        ## TODO: support for splitting
        #h = self.h = [{(a[0],a[1],K): lp.addVar(obj=0, lb=self.problem.network[a[0]][a[0]]['var_cost'] * self.problem.commodities[k]['q'], ub=lp.inf(), name='q' + str((k,a,K))) 
        #                 for a,K_coll in self.consolidations.items()
        #                    for K in K_coll if k in K}
        #                        for k, path_graph in enumerate(self.solution_paths)]
        ## holding at last node
        #for k, hk in enumerate(h):
        #    hk.update({(a[1],a[1],K): lp.addVar(obj=0, lb=self.problem.network[a[1]][a[1]]['var_cost'] * self.problem.commodities[k]['q'], ub=lp.inf(), name='q' + str((k,a,K))) 
        #                 for a,K_coll in self.consolidations.items()
        #                    for K in K_coll if k in K and self.problem.commodities[k]['b'][0] == a[1]})

        ## y - dispatch time at each node in path [k]{n}
        #t = self.t = [{path[i]: lp.addVar(obj=0, lb=0, ub=lp.inf(), name='t' + str(k) + ',' + str(path[i])) 
        #                 for i in range(len(path))}
        #              for k, path in enumerate(self.solution_paths)]

        ## x - slack variables for consolidation x[a,k1,k2]
        #x = self.x = {(a,k1,k2): lp.addVar(obj=self.problem.network[a[0][0]][a[1][0]]['fixed_cost'] 
        #                                   #+self.problem.network[a[0][0]][a[1][0]]['var_cost']*((self.problem.commodities[k1]['q'] + self.problem.commodities[k2]['q']) % 1)
        #                                   , lb=0, ub=lp.inf(), name='x' + str(a) + ',' + str((k1,k2)))
        #                for a, group in self.consolidations.items() for k1 in group for k2 in group if k1 < k2}

        lp.update()

        # Constraints
        for k, path_graph in enumerate(self.solution_paths):
            c = self.problem.commodities[k]

            for n1,n2,d in path_graph.edges(data=True):
                for K in d['K']:
                    # origin dispatch time >= origin time
                    if n1 == c['a'][0]:
                        #lp.addConstr(t[k][n1,n2,K] == h[k][n1,n2,K] + c['a'][1], 'L' + str((k,n1,n2,K)))
                        lp.addConstr(t[k][n1,n2,K] >= c['a'][1], 'L' + str((k,n1,n2,K)))

                    # dispatch time >= last dispatch + transit time along path
                    for _,n3,d2 in path_graph.out_edges(n2, data=True):
                        for K2 in d2['K']:
                            #lp.addConstr(t[k][n2,n3,K2] == h[k][n2,n3,K2] + t[k][n1,n2,K] + self.problem.transit(n1,n2), 'L' + str((k,n1,n2,n3,K,K2)))
                            lp.addConstr(t[k][n2,n3,K2] >= t[k][n1,n2,K] + self.problem.transit(n1,n2), 'L' + str((k,n1,n2,n3,K,K2)))

                    # destination dispatch time <= destination time
                    if n2 == c['b'][0]:
                        #lp.addConstr(t[k][n1, n2, K] + h[k][n2,n2,K] == c['b'][1] - self.problem.transit(n1,n2), 'U' + str((k,n1,n2,K)))
                        lp.addConstr(t[k][n1, n2, K] <= c['b'][1] - self.problem.transit(n1,n2), 'U' + str((k,n1,n2,K)))

                # consolidating dispatch time is equal + slack variable
                if FAST_CONSOLIDATIONS:
                    k1 = min(K)
                    lp.addConstrs((x[((n1,n2),k1,k2,K)] >= t[k1][n1, n2, K] - t[k2][n1, n2, K] for k2 in K if k1 < k2))
                    lp.addConstrs((x[((n1,n2),k1,k2,K)] >= t[k2][n1, n2, K] - t[k1][n1, n2, K] for k2 in K if k1 < k2))
                else:
                    for k1 in K:
                        for k2 in K:
                            if k1 < k2:
                                lp.addConstr(x[((n1,n2),k1,k2,K)] >= t[k1][n1, n2, K] - t[k2][n1, n2, K], 'a' + str((n1, n2, K)))
                                lp.addConstr(x[((n1,n2),k1,k2,K)] >= t[k2][n1, n2, K] - t[k1][n1, n2, K], 'a' + str((n1, n2, K)))

                # note above is equivalent to following. Both are slow for large consolidation groups!  
                # Perhaps need to use matrix API?
                #lp.addConstrs((x[((n1,n2),k1,k2,K)] >= t[k1][n1, n2, K] - t[k2][n1, n2, K] for k2 in K for k1 in K if k1 < k2))
                #lp.addConstrs((x[((n1,n2),k1,k2,K)] >= t[k2][n1, n2, K] - t[k1][n1, n2, K] for k2 in K for k1 in K if k1 < k2))

                    

           
        #for k, path in enumerate(self.solution_paths):
        #    # origin dispatch time >= origin time, destination dispatch time <= destination time
        #    lp.addConstr(t[k][path[0]] >= self.problem.commodities[k]['a'][1], 'L' + str(k) + ',' + str(path[0]))
        #    lp.addConstr(t[k][path[-1]] <= self.problem.commodities[k]['b'][1], 'U' + str(k) + ',' + str(path[-1]))
            
        #    # dispatch time >= last dispatch + transit time along path
        #    for n1,n2 in zip(path,path[1:]):
        #        lp.addConstr(t[k][n2] >= t[k][n1] + self.problem.transit(n1,n2), 'L' + str(k) + ',' + str(n1))
    
        ## consolidating dispatch time is equal + slack variable
        #for a, group in self.consolidations.items():
        #    for k1 in group:
        #        for k2 in group:
        #            if k1 < k2:
        #                lp.addConstr(x[(a,k1,k2)] >= t[k1][a[0][0]] - t[k2][a[0][0]], 'a' + str(a) + ',' + str(k1) + ',' + str(k2))
        #                lp.addConstr(x[(a,k1,k2)] >= t[k2][a[0][0]] - t[k1][a[0][0]], 'a' + str(a) + ',' + str(k1) + ',' + str(k2))

        lp.update()
        lp.optimize()

        return lp.is_optimal()

    ## force broken consolidation to consolidate and see what else breaks
    #def test_fix_and_resolve(self, arc):
    #    if arc == None:
    #        return self.model.status

    #    tmp = [x for k, x in self.x.items() if k[0] == arc and x.x > 0]
    #    self.model.setAttr("UB", tmp, [0]*len(tmp))

    #    self.model.update()
    #    self.model.optimize()

    #    self.model.setAttr("UB", tmp, [GRB.INFINITY]*len(tmp))

    #    # if infeasible, it means that the removed consolidation have a "time window" problem, 
    #    # otherwise it is a "mutual" issue, and the conflicting consolidation should be now given
    #    return self.model.status

    ## force broken consolidation to consolidate and see what else breaks
    #def test_remove_and_resolve(self, arc):
    #    if arc != None:
    #        tmp = [x for k, x in self.x.items() if k[0] == arc and x.x > 0]
    #        self.model.setAttr("UB", tmp, [0]*len(tmp))

    #    self.model.update()
    #    self.model.optimize()

    #    # if infeasible, it means that the removed consolidation have a "time window" problem, 
    #    # otherwise it is a "mutual" issue, and the conflicting consolidation should be now given
    #    return self.model.status


    def get_solution_times(self):
        paths = []
        for k,path in enumerate(self.solution_paths):
            paths.append([])

            for n in path:
                paths[k].append((n, self.model.val(self.t[k][n])))

        return paths

    # get new consolidations
    def get_consolidations(self):
        consolidation = defaultdict(set)

        for k, path_graph in enumerate(self.solution_paths):
            for n1,n2,d in path_graph.edges(data=True):
                for K,q in zip(d['K'],d['q']):
                    consolidation[n1, n2, round(self.model.val(self.t[k][n1,n2,K]), PRECISION)].add(k)

        #for k, path in enumerate(self.solution_paths):
        #    for n1,n2 in pairwise(path):
        #        consolidation[(n1,n2,self.model.val(self.t[k][n1]))].add(k)
         
        return sorted([(c[:2], frozenset(k)) for c,k in consolidation.items() if len(k) > 1])


    # get new consolidations and calculate new solution cost
    def get_solution_cost(self):
        consolidation = {}
        var_cost = 0
        holding_cost = 0

        for k, path_graph in enumerate(self.solution_paths):
            for n1,n2,d in path_graph.edges(data=True):
                for K,q in zip(d['K'],d['q']):
                    a = (n1,n2,round(self.model.val(self.t[k][n1,n2,K]), PRECISION))
                    consolidation[a] = consolidation.get(a, 0) + self.problem.commodities[k]['q']*q
                    var_cost += self.problem.problem.var_cost[k].get((n1,n2),0) * self.problem.commodities[k]['q']*q

                    ## TODO: support split correctly
                    #holding_cost += self.problem.network[n1][n1]['var_cost'] * self.problem.commodities[k]['q']*q * self.model.val(self.h[k][n1,n2,K])

                    #if n2 == self.problem.commodities[k]['b'][0]:
                    #    holding_cost += self.problem.network[n2][n2]['var_cost'] * self.problem.commodities[k]['q']*q * self.model.val(self.h[k][n2,n2,K])


        #for k, path in enumerate(self.solution_paths):
        #    for n1,n2 in zip(path, path[1:]):
        #        a = (n1,n2,round(self.model.val(self.t[k][n1]), PRECISION))
        #        consolidation[a] = consolidation.get(a, 0) + self.problem.commodities[k]['q']
        #        var_cost += self.problem.network[n1][n2]['var_cost'] * self.problem.commodities[k]['q']

        return var_cost + holding_cost + sum(self.problem.network[a][b]['fixed_cost'] * math.ceil(q/self.problem.network[a][b]['capacity']) for (a,b,t), q in consolidation.items())

#    # get broken consolidations
#    def get_broken_consolidations(self):
#        return {k: set(map(operator.itemgetter(1,2), group)) for k,group in itertools.groupby(sorted(k for k,v in self.x.items() if v.x > 0), key=operator.itemgetter(0))}
##        return {k: v.x for k,v in self.x.items() if v.x > 0}

    # get statistics
    def get_statistics(self):
        stats = {}

        if self.model.is_optimal():
            # get path and dispatch times for each commodity
            paths = []
            for k,path in enumerate(self.solution_paths):
                paths.append([])

                for n in path:
                    paths[k].append((n, self.model.val(self.t[k][n])))

            stats['solution_path'] = paths

            # get new consolidations and calculate new solution cost
            consolidation = {}

            for k, path in enumerate(self.solution_paths):
                for n1,n2 in zip(path, path[1:]):
                    a = ((n1,n2), self.model.val(self.t[k][n1]))

                    if a not in consolidation:
                        consolidation[a] = []

                    consolidation[a].append(k)

            stats['cost'] = self.model.objVal()
            stats['consolidation'] = consolidation
            stats['solution_cost'] = self.get_solution_cost()
        else:
            stats['cost'] = None
            stats['consolidation'] = None
            stats['solution_cost'] = None
            stats['solution_path'] = None

        return stats

    def print_solution(self):
        if self.model.status == GRB.status.OPTIMAL:
            print('\nSolution:')

            times = sorted(set([self.t[k][path[n]].x for k, path in enumerate(self.solution_paths) for n in range(len(path) - 1)]))

            # unique t        
            for time in range(len(times)):
                tmp = {}

                for k, path in enumerate(self.solution_paths):
                    for a in zip(path,path[1:]):
                        if self.t[k][a[0]].x == times[time]:
                            if a not in tmp:
                                tmp[a] = []

                            tmp[a].append(k)

                for a,col in tmp.items():
                    print("t={3}, k={0}, a=({1},{2})".format(col, a[0], a[1], times[time]))

            for a,v in self.x.items():
                if v.x != 0:
                    print(v)

    #def time_windows(self, k, path):
    #    t1 = self.problem.commodities[k]['a'][1]
    #    t2 = self.problem.commodities[k]['b'][1]

    #    paths = list(pairwise(path))

    #    early = [0] + list(accumulate((self.problem.network[n[0]][n[1]] for n in paths)))
    #    late  = [0] + list(accumulate((self.problem.network[n[0]][n[1]] for n in reversed(paths))))

    #    return {p: (float(t1 + e), float(t2 - l)) for (p,e,l) in zip(self.solution_paths[k], early, reversed(late))}


    ## given paths & consolidations, find the minimum cost (IP) to drop consolidations
    #def validateIP(self, solution_paths, consolidations):
    #    self.solution_paths = solution_paths
    #    self.consolidations = consolidations  # c={a,K}
        
    #    TW = [self.time_windows(k,P) for k,P in enumerate(self.solution_paths)]

    #    if [t for k,t in enumerate(TW) if t[self.problem.commodities[k]['b'][0]][0] > t[self.problem.commodities[k]['b'][0]][1]]:
    #        return -1 # infeasible

    #    T = {c: range(min(int(TW[k][c[0][0]][0]) for k in c[1]), max(int(TW[k][c[0][0]][1]) for k in c[1]))
    #            for c in consolidations}

    #    ip = self.model = Model("min_cost")
    #    #lp.setParam('OutputFlag', False)
    #    ip.setParam(GRB.param.MIPGap, 0.005)
    #    ip.modelSense = GRB.MINIMIZE

    #    # t - dispatch time at each node in path [k]{n}
    #    y = [{path[i]: ip.addVar(obj=0, lb=0, ub=GRB.INFINITY, name='y' + str(k) + ',' + str(path[i])) 
    #                     for i in range(len(path))}
    #                  for k, path in enumerate(self.solution_paths)]

    #    # x - network flow variables for consolidation x[k,a,t] t=0,1,2,...
    #    x = {(k,a,t): ip.addVar(obj=0, lb=0, ub=1, vtype=GRB.BINARY, name='x' + str((k,a,t)))
    #                    for a, K in consolidations for k in K for t in T[a,K]}

    #    # z - consolidation variables z[c,t] t=0,1,2,...
    #    z = {(c,t): ip.addVar(obj=self.problem.fixed_cost[c[0][0],c[0][1]], lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='z' + str(c) + ',' + str(t))
    #                    for c in consolidations for t in T[c]}

    #    ip.update()

    #    # Constraints
    #    for k, path in enumerate(self.solution_paths):
    #        # origin dispatch time >= origin time, destination dispatch time <= destination time
    #        ip.addConstr(y[k][path[0]] >= self.problem.commodities[k]['a'][1], 'L' + str(k) + ',' + str(path[0]))
    #        ip.addConstr(y[k][path[-1]] <= self.problem.commodities[k]['b'][1], 'U' + str(k) + ',' + str(path[-1]))
            
    #        # dispatch time >= last dispatch + transit time along path
    #        for n1,n2 in zip(path,path[1:]):
    #            ip.addConstr(y[k][n2] >= y[k][n1] + self.problem.network[n1][n2], 'L' + str(k) + ',' + str(n1))

    #        for a,K in consolidations:
    #            if k in K:
    #                ip.addConstr(quicksum(x[k,a,t] for t in T[a,K]) == 1)
    #                ip.addConstr(y[k][a[0]] == quicksum(t*x[k,a,t] for t in T[a,K]))
                    
    #    # consolidating dispatch time
    #    for c in consolidations:
    #        for t in T[c]:
    #            ip.addConstr(z[c,t] * self.problem.capacities[c[0]] >= quicksum(x[k,c[0],t] * self.problem.commodities[k]['q'] for k in c[1]))

    #    ip.update()
    #    ip.optimize()

    #    cost = 0
    #    CD = dict((a,map(itemgetter(1), g)) for a,g in itertools.groupby(sorted(consolidations), itemgetter(0)))

    #    for k, path in enumerate(solution_paths):
    #        for n1,n2 in zip(path, path[1:]):
    #            cost += self.problem.var_cost[n1,n2] * self.problem.commodities[k]['q']

    #            if (n1,n2) not in CD or not [K for K in CD[n1,n2] if k in K]:
    #                cost += self.problem.fixed_cost[n1,n2] * math.ceil(self.problem.commodities[k]['q'] / self.problem.capacities[n1,n2])

    #    #print self.problem.solution[0]
    #    #print ip.objVal, cost

    #    if ip.status == GRB.status.OPTIMAL:
    #        return ip.objVal + cost

    #    return -1

    #def validateLP(self, solution_paths, consolidations):
    #    self.solution_paths = solution_paths
    #    self.consolidations = consolidations

    #    TW = [self.time_windows(k,P) for k,P in enumerate(self.solution_paths)]

    #    if [t for k,t in enumerate(TW) if t[self.problem.commodities[k]['b'][0]][0] > t[self.problem.commodities[k]['b'][0]][1]]:
    #        return False

    #    lp = self.model = Model("find_time")
    #    lp.setParam('OutputFlag', False)
    #    lp.modelSense = GRB.MINIMIZE

    #    # y - dispatch time at each node in path [k]{n}
    #    y = self.y = [{path[i]: lp.addVar(obj=0, lb=0, ub=GRB.INFINITY, name='y' + str(k) + ',' + str(path[i])) 
    #                     for i in range(len(path))}
    #                  for k, path in enumerate(self.solution_paths)]

    #    # x - slack variables for consolidation x[a,k1,k2]
    #    x = self.x = {(a,k1,k2): lp.addVar(obj=self.problem.fixed_cost[a[0], a[1]], lb=0, ub=GRB.INFINITY, name='x' + str(a) + ',' + str((k1,k2)))
    #                    for a, group in self.consolidations for k1 in group for k2 in group if k1 < k2}

    #    lp.update()

    #    # Constraints
    #    for k, path in enumerate(self.solution_paths):
    #        # origin dispatch time >= origin time, destination dispatch time <= destination time
    #        lp.addConstr(y[k][path[0]] >= self.problem.commodities[k]['a'][1], 'L' + str(k) + ',' + str(path[0]))
    #        lp.addConstr(y[k][path[-1]] <= self.problem.commodities[k]['b'][1], 'U' + str(k) + ',' + str(path[-1]))
            
    #        # dispatch time >= last dispatch + transit time along path
    #        for n1,n2 in zip(path,path[1:]):
    #            lp.addConstr(y[k][n2] >= y[k][n1] + self.problem.network[n1][n2], 'L' + str(k) + ',' + str(n1))
    
    #    # consolidating dispatch time is equal + slack variable
    #    for a, group in self.consolidations:
    #        for k1 in group:
    #            for k2 in group:
    #                if k1 < k2:
    #                    lp.addConstr(x[(a,k1,k2)] >= y[k1][a[0]] - y[k2][a[0]], 'a' + str(a) + ',' + str(k1) + ',' + str(k2))
    #                    lp.addConstr(x[(a,k1,k2)] >= y[k2][a[0]] - y[k1][a[0]], 'a' + str(a) + ',' + str(k1) + ',' + str(k2))

    #    lp.update()
    #    lp.optimize()

    #    return lp.status == GRB.status.OPTIMAL

    ## get new consolidations and calculate new solution cost
    #def get_solution_costLP(self):
    #    consolidation = {}
    #    var_cost = 0

    #    for k, path in enumerate(self.solution_paths):
    #        for n1,n2 in zip(path, path[1:]):
    #            a = (n1,n2,self.y[k][n1].x)
    #            consolidation[a] = consolidation.get(a, 0) + self.problem.commodities[k]['q']
    #            var_cost += self.problem.var_cost[n1,n2] * self.problem.commodities[k]['q']

    #    return var_cost + sum(self.problem.fixed_cost[a,b] * math.ceil(q/self.problem.capacities[a,b]) for (a,b,t), q in consolidation.items())

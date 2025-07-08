from gurobipy import *
import networkx as nx

#
# dodgey way to get 2nd shortest path (via IP) and 0.01 tolerance on objective cost
#
def second_shortest_path(network, origin, destination, best_path_cost=None, path=None):
    if origin == destination:
        return 0

    model = Model("path")
    model.setParam('OutputFlag', False)
    model.modelSense = GRB.MINIMIZE

    ##
    ## decision variables
    ##

    arcs = tuplelist(network.edges())

    # x - dispatch along an arc [a]
    x = {a: model.addVar(obj=network[a[0]][a[1]]['weight'], lb=0, ub=1, vtype=GRB.BINARY, name='x' + str(a)) 
            for a in arcs}

    model.update()

    ##
    ## Constraints
    ## 

    # flow constraints
    def inflow(n):
        return [x[a] for a in arcs.select('*', n)]

    def outflow(n):
        return [x[a] for a in arcs.select(n, '*')]

    def r(n):
        # at origin
        if origin == n:
            return -1

        # at destination
        if destination == n:
            return 1

        return 0

    for n in network.nodes():
        i,o = inflow(n), outflow(n)

        if len(i) > 0 or len(o) > 0:
            model.addConstr(quicksum(i) - quicksum(o) == r(n), 'flow' + str(n))
    
    # cannot use same cost
    if best_path_cost != None:
        model.addConstr(quicksum(x[a] * network[a[0]][a[1]]['weight']  for a in arcs) >= best_path_cost + 0.01, 'next_best_path')

    # cannot use same path
    if path != None:
        model.addConstr(quicksum(x[a] for a in zip(path, path[1:])) <= len(path) - 2, 'next_path')

    model.update()
    model.optimize()

    if path != None:
        result = []

        if model.status == GRB.status.OPTIMAL:    
            a = [b for b in arcs.select(origin, '*') if x[b].x > 0][0]
            result.append(origin)

            while a[1] != destination:
                result.append(a[1])
                a = [b for b in arcs.select(a[1], '*') if x[b].x > 0][0]

            result.append(destination)

        return result        

    return model.objVal if model.status == GRB.status.OPTIMAL else None
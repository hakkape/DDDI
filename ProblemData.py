import networkx as nx
import random
import math
import csv
from typing import NamedTuple
from itertools import pairwise
import bisect

class NodeTime(NamedTuple):
    node: int
    time: float

class NodeInterval(NamedTuple):
    node: int
    t1: float
    t2: float

    @property
    def T(self):
        return (self.node, self.t1, self.t2)

class TimedArc(NamedTuple):
    source: NodeInterval
    target: NodeInterval

class Commodity(object):
    __slots__ = ['a', 'b', 'q']

    def __init__(self, a: NodeTime, b: NodeTime, q: float):
        self.a = a
        self.b = b
        self.q = q

class ProblemData(object):
    """description of class"""
    __slots__ = ['commodities', 'network', 'position', 'capacities', 'fixed_cost', 'var_cost', 'solution', 'fixed_paths']

    def __init__(self, commodities: list[Commodity], network: dict[int, dict[int, float]], position=None, capacities={}, fixed_cost={}, var_cost: list[dict[tuple[int, int], float]]=[], solution=None, fixed_paths=None):
        self.commodities = commodities
        self.network = network
        self.position = position
        self.capacities = capacities
        self.fixed_cost = fixed_cost
        self.var_cost = var_cost
        self.solution = solution
        self.fixed_paths = fixed_paths

        if self.var_cost == []:
            self.var_cost = [{} for _ in range(len(self.commodities))]


    ## significant time-points from "New Dynamic Discretization Discovery Strategies Continuous-Time Service Network Design"
    ## TODO: can simplify code by removing take_percentage
    def significant_time_points(self, take_percentage=1.0):
        # initialize with commodity origin/destination time points
        time_points: set[NodeTime] = set((k.a for k in self.commodities))
        time_points.update(k.b for k in self.commodities)

        all_tmp: list[tuple[int, NodeTime]] = []

        # build network graph and get shortest paths
        # we also do this later in the solve, so ideally we should only do it once
        G = nx.DiGraph()

        for a, destinations in self.network.items():
            G.add_edges_from((a, b, {'weight': transit_time}) for b, transit_time in destinations.items())

        shortest_paths = dict(nx.shortest_path_length(G, weight='weight'))

        # earliest arrivals for each node
        for n in G.nodes():
            min_time = min((k.a.time + shortest_paths[k.a.node][n] for k in self.commodities))
            time_points.add(NodeTime(n, min_time))

        # Build per-node sorted time index for fast range checks
        time_index = {n: sorted(tp.time for tp in time_points if tp.node == n) for n in G.nodes()}
        K = range(len(self.commodities))

        for i in sorted(G.nodes()):
            # collect candidate intervals W for node i
            W: list[tuple[float, float, float]] = []

            # precompute arrival upper bounds at node i for each commodity k1
            arrive_i = [k.a.time + shortest_paths[k.a.node][i] for k in self.commodities]
            times_i = time_index.get(i, [])
            sp_i = shortest_paths[i]

            for j in G.successors(i):
                sp_i_j = sp_i[j]
                sp_j = shortest_paths[j]

                for k1 in K:
                    right = arrive_i[k1]

                    for k2 in K:
                        if k1 != k2:
                            dest = self.commodities[k2].b
                            left = dest.time - sp_j[dest.node] - sp_i_j

                            if left < right:
                                # fast check: any existing time point in (left, right] for node i?
                                idx = bisect.bisect_right(times_i, left)
                                if not (idx < len(times_i) and times_i[idx] <= right):
                                    W.append((left, right, self.fixed_cost.get((i, j), 0)))

            # Greedy minimum hitting set for intervals
            if W:
                W.sort()
                tp = W[0][1]
                count = W[0][2]  # cost of first interval
                tmp = []

                for w in W[1:]:
                    count += w[2]  # cost of current interval
                    # skip if tp is valid for this interval
                    if w[0] < tp <= w[1]:
                        continue

                    count -= w[2]
                    tmp.append((count, tp))
                    tp = w[1]
                    count = w[2]

                tmp.append((count, tp))
                all_tmp.extend([(c, NodeTime(i, t)) for c,t in tmp]) 

                # Select the time points with the largest impact
                # tmp.sort(reverse=True)
                # time_points.update((NodeTime(i, tp) for _,tp in tmp[:math.floor(len(tmp)*take_percentage) + 1]))

            # print progress
            print(f"Processed node {i}/{len(G.nodes())}, found {len(time_points)} time points", end='\r')
        print()

        all_tmp.sort(reverse=True)
        time_points.update((tp for _,tp in all_tmp[:math.floor(len(all_tmp)*take_percentage) + 1]))
        print(f"Found {len(time_points)} time points")

        return time_points
    

    ## Scales the time horizon (network and commodities) for this problem
    def scale(self, scale):
        for k,c in enumerate(self.commodities):
            c.a = NodeTime(c.a[0], int(math.ceil(c.a[1]*scale)))
            c.b = NodeTime(c.b[0], int(math.ceil(c.b[1]*scale)))

        for a, destinations in self.network.items():
            for b, transit_time in destinations.items():
                destinations[b] = int(math.ceil(transit_time * scale))

        return self


    ## Pessimistically rounds time as if using coarser time discretization, assumes data is in 1 minute discretization
    def pessimistic_round(self, minutes):
        # copy costs from transit to fix rounding issues
        copy_cost = len(self.fixed_cost) == 0

        for k,c in enumerate(self.commodities):
            c.a = NodeTime(c.a[0], int(math.ceil(c.a[1]/float(minutes))))   # early up
            c.b = NodeTime(c.b[0], int(math.floor(c.b[1]/float(minutes))))  # late down

        # transit times up
        for a, destinations in self.network.items():
            for b, transit_time in destinations.items():
                destinations[b] = int(math.ceil(transit_time/float(minutes)))

                if copy_cost:
                    self.fixed_cost[(a,b)] = transit_time

        return self

    ## Optimistically rounds time as if using coarser time discretization, assumes data is in 1 minute discretization
    def optimistic_round(self, minutes):
        # copy costs from transit to fix rounding issues
        copy_cost = len(self.fixed_cost) == 0

        for k,c in enumerate(self.commodities):
            c.a = NodeTime(c.a[0], int(math.floor(c.a[1]/float(minutes))))   # early down
            c.b = NodeTime(c.b[0], int(math.ceil(c.b[1]/float(minutes))))  # late up

        # transit times down
        for a, destinations in self.network.items():
            for b, transit_time in destinations.items():
                destinations[b] = int(math.floor(transit_time/float(minutes)))

                if copy_cost:
                    self.fixed_cost[(a,b)] = transit_time

        return self

    ## Simply rounds time as if using coarser time discretization, assumes data is in 1 minute discretization
    def simple_round(self, minutes):
        # copy costs from transit to fix rounding issues
        copy_cost = len(self.fixed_cost) == 0

        for k,c in enumerate(self.commodities):
            c.a = NodeTime(c.a[0], int(round(c.a[1]/float(minutes))))   # early
            c.b = NodeTime(c.b[0], int(round(c.b[1]/float(minutes))))  # late

        # transit times
        for a, destinations in self.network.items():
            for b, transit_time in destinations.items():
                destinations[b] = int(round(transit_time/float(minutes)))

                if copy_cost:
                    self.fixed_cost[(a,b)] = transit_time

        return self


    ## Randomizes a previous problem
    def randomize(self, commodity_number=None, commodity_range=(0,10), quantity_range=(0, 2), start_range=(0, 10), origin_set=[], dest_set=[], scope=None, scope_range=(1, 4)):
        p = ProblemData.random_problem(self.network, commodity_number, commodity_range, quantity_range, start_range, origin_set, dest_set, scope, scope_range)
        p.position = self.position
        return p

    ##
    ## Load problem data from common format (Mike Hewitt)
    ##
    @classmethod
    def read_file(cls, filename):
        commodities = []
        network = {}
        positions = []
        capacities = {}
        fixed_cost = {}
        var_cost = {}

        with open(filename, "r") as file:
            while not file.readline().startswith("NODES"):
                pass

            line = file.readline()

            if line.startswith("I"):
                line = file.readline() # skip header

            # read positions
            while not line.startswith("ARCS"):
                tmp = line.split(',')

                if not tmp[2].startswith('-') and not tmp[3].startswith('-'):
                    positions.append([float(tmp[2]), float(tmp[3])])

                line = file.readline()

            line = file.readline()

            if line.startswith("I"):
                line = file.readline() # skip header

            while not line.startswith("COMMODITIES"):
                tmp = line.split(',')

                if int(tmp[1]) not in network:
                    network[int(tmp[1])] = {}

                network[int(tmp[1])][int(tmp[2])] = float(tmp[6])
                
                if float(tmp[5]) >= 0:  # ignore capacities of -1
                    capacities[(int(tmp[1]),int(tmp[2]))] = float(tmp[5])

                fixed_cost[(int(tmp[1]),int(tmp[2]))] = float(tmp[4])
                var_cost[(int(tmp[1]),int(tmp[2]))] = float(tmp[3])

                line = file.readline()

            line = file.readline()

            if line.startswith("I"):
                line = file.readline() # skip header

            while len(line) > 0 and not (line.startswith('horizon') or line.startswith("cost")):
                tmp = line.split(',')
                commodities.append(Commodity(NodeTime(int(tmp[1]), float(tmp[4])), NodeTime(int(tmp[2]), float(tmp[5])), float(tmp[3])))
                line = file.readline()

            ## load solution
            solution_cost = None
            solution_paths = []
            solution_cons = []

            if len(line) > 0 and line.startswith('horizon'):
                line = file.readline()

            if len(line) > 0 and line.startswith('cost'):
                solution_cost = float(line.split('=')[1])
                line = file.readline()  # PATHS
                line = file.readline()

                if line.startswith("Index"):
                    line = file.readline() # skip header

                # load solution paths
                while len(line) > 0 and not line.startswith("CONS"):
                    tmp = line.split(',')
                    solution_paths.append(map(int, tmp[1:]))
                    line = file.readline()

                line = file.readline() # CONS
            
                if line.startswith("Origin"):
                    line = file.readline() # skip header

                while len(line) > 0:
                    tmp = list(map(int, line.split(',')))
                    solution_cons.append((tuple(tmp[:2]), frozenset(tmp[2:])))
                    line = file.readline()

            return ProblemData(commodities, network, positions if positions else None, capacities, fixed_cost, [var_cost]*len(commodities), (solution_cost, solution_paths, solution_cons) if solution_cost is not None else None)

    ##
    ## Save problem data in common format (Mike Hewitt)
    ##
    def save(self, filename, solution=None):
        graph = nx.DiGraph()

        for a, destinations in self.network.items():
            for b, transit_time in destinations.items():
                graph.add_edge(a, b, weight=transit_time, cost=transit_time)

        with open(filename, "w") as file:
            file.write("NODES," + str(len(graph.nodes())) + '\n')
            file.write("INDEX,Name,X-coordinate,Y-coordinate\n")

            # try:
            #     position = self.position if self.position else nx.pygraphviz_layout(graph, prog='neato')
        
            #     for n in graph.nodes():
            #         file.write("{0},{1},{2},{3}\n".format(n, n, position[n][0], position[n][1]))
            # except:
            # don't write position                
            for n in graph.nodes():
                file.write("{0},{1},-,-\n".format(n, n))

            file.write("ARCS," + str(len(graph.edges())) + '\n')
            file.write("Index,Origin,Destination,Variable Cost,Fixed Cost,Capacity,Travel time\n")

            for i, (a,b) in enumerate(graph.edges()):
                file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(i, a, b, self.var_cost[0][(a,b)] if (a,b) in self.var_cost else 0, self.fixed_cost[(a,b)] if (a,b) in self.fixed_cost else self.network[a][b], self.capacities[(a,b)] if (a,b) in self.capacities else 1, self.network[a][b]))

            file.write("COMMODITIES," + str(len(self.commodities)) + '\n')
            file.write("Index,Origin,Destination,Demand/Size,Earliest available time,Latest delivery time\n")

            for k,c in enumerate(self.commodities):
                file.write("{0},{1},{2},{3},{4},{5}\n".format(k, c.a[0], c.b[0], c.q, c.a[1], c.b[1]))

            file.write("horizon={0}\n".format(max(c.b[1] for c in self.commodities) - min(c.a[1] for c in self.commodities)))

            if solution is not None:
                file.write("cost={0}\n".format(solution[0]))

                file.write("PATHS,{0}\n".format(len(solution[1])))
                file.write("Index,Nodes\n")

                for k,p in enumerate(solution[1]):
                    file.write("{0},{1}\n".format(k,",".join(map(str,p))))

                file.write("CONSOLIDATIONS,{0}\n".format(len(solution[2])))
                file.write("Origin,Destination,Commodities\n")

                for (n1,n2),K in solution[2]:
                    file.write("{0},{1},{2}\n".format(n1,n2,",".join(map(str,K))))


    ##
    ## Creates a randomly generated problem
    ##
    ## commodity_number: number of commodities to generate (default chooses random number in commodity_range)
    ## commodity_range: integer (lower, upper]
    ## quantity_range: rational (lower, upper]
    ## start_range: integer [lower, upper)
    ##
    ## scope: scale of time window ~ scope * shortest_path[origin][destination]
    ## scope_range: rational > 1 (lower, upper]
    ##
    ## origin_set: the set of available origins to choose from
    ## dest_set: the set of available destinations to choose from
    ##
    @classmethod
    def random_problem(cls, network, commodity_number=None, commodity_range=(0,10), quantity_range=(0, 2), start_range=(0, 10), origin_set=[], dest_set=[], scope=None, scope_range=(1, 4)):
        if commodity_number is None:
            commodity_number = commodity_range[1] - random.randrange(commodity_range[0], commodity_range[1])

        commodities = []

        # build graph
        graph = nx.DiGraph()

        for a, destinations in network.items():
            for b, transit_time in destinations.items():
                graph.add_edge(a, b, weight=transit_time, cost=transit_time)

        shortest_paths = nx.shortest_path_length(graph, weight='weight')

        # generate commodities
        for k in range(commodity_number):
            # choose valid origin/destination pair
            origin, dest = random.choice(origin_set or graph.nodes()), random.choice(dest_set or graph.nodes())

            while origin == dest or dest not in shortest_paths[origin]:
                origin, dest = random.choice(origin_set or graph.nodes()), random.choice(dest_set or graph.nodes())

            # choose valid time window
            origin_time = random.randrange(start_range[0], start_range[1])
            dest_time = origin_time + int(shortest_paths[origin][dest] * (scope if scope is not None else random.uniform(scope_range[0], scope_range[1])))

            commodities.append(Commodity(NodeTime(origin, origin_time), NodeTime(dest, dest_time), max(0.01, round(quantity_range[1] - random.uniform(quantity_range[0], quantity_range[1]), 2))))

        return ProblemData(commodities, network)


    ##
    ## Load tsp problem data from tw file.  Note this works, but is TERRIBLY slow
    ##
    @classmethod
    def read_tsp(cls, filename):
        commodities = []
        network = {}
        capacities = {}
        fixed_cost = {}
        var_cost = {}

        with open(filename, "r") as file:
            nodes = int(file.readline())

            # complete graph
            network = {i: {j: float(t) for j,t in enumerate(filter(None, file.readline().rstrip().split(' '))) if j != i} for i in range(nodes)}

            for a,G in network.items():
                for b,v in G.items():
                    if v == 0:
                        fixed_cost[a,b] = 1000

            M = float(list(filter(None, file.readline().rstrip().split(' ')))[1])

            for i in range(nodes - 1):
                t = list(filter(None, file.readline().rstrip().split(' ')))

                commodities.append(Commodity(NodeTime(i+1, float(t[0])), NodeTime(0, M), 1/float(nodes+1)))
                commodities.append(Commodity(NodeTime(0, 0), NodeTime(i+1, float(t[1])), 1/float(nodes+1)))

            return ProblemData(commodities, network, None, capacities, fixed_cost, [var_cost]*len(commodities), None)


    ##
    ## Load instance from DDD-arc paper
    ##
    @classmethod
    def read_directory(cls, directory):
        commodities = []
        network = {}
        capacities = {}
        fixed_cost = {}
        var_cost = {}
        fixed_paths = []

        nodes = {}

        with open(directory + '/nodes.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # skip header

            for row in reader:
                nodes[row[0]] = len(nodes)

        commodity_var_cost = {}
        commodity_var_cost_network = []
        commodity_map = {}

        # id,origin,destination,demand,release_time,deadline
        with open(directory + '/commodities.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # skip header

            for row in reader:
                origin,dest = nodes[row[1]], nodes[row[2]]
                commodities.append(Commodity(NodeTime(origin, float(row[4])), NodeTime(dest, float(row[5])), float(row[3])))
                commodity_map[row[0]] = len(commodity_map)
                commodity_var_cost_network.append({})

                # load fixed paths
                if len(row) > 7:
                    fixed_paths.append(list(pairwise([nodes[n.strip(" '")] for n in row[7].strip('[]').split(',')])))

        #commodity,arcs
        with open(directory + '/variable_costs.csv') as csvfile:
            reader = csv.reader(csvfile)
            row = next(reader) # read header
            arc_map = {f: i for i,f in enumerate(row) if f != 'commodity'} 

            for row in reader:
                commodity_var_cost[commodity_map[row[0]]] = {k: row[v] for k,v in arc_map.items()}

        # id,origin,destination,transit_time,capacity,fixed_cost,variable_cost
        with open(directory + '/arcs.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # skip header

            for row in reader:
                origin,dest = nodes[row[1]], nodes[row[2]]

                if origin not in network:
                    network[origin] = {}

                network[origin][dest] = float(row[3])
                
                if float(row[4]) >= 0:  # ignore capacities of -1
                    capacities[(origin,dest)] = float(row[4])

                fixed_cost[(origin,dest)] = float(row[5])

                if row[6] != '': # ignore empty variable costs
                    var_cost[(origin,dest)] = float(row[6])

                for k in range(len(commodities)):
                    commodity_var_cost_network[k][origin,dest] = float(commodity_var_cost[k][row[0]])
    
        return ProblemData(commodities, network, None, capacities, fixed_cost, commodity_var_cost_network, None, fixed_paths)
from gurobipy import tuplelist
from SecondShortestPath import second_shortest_path
import networkx as nx
from ProblemData import ProblemData

class BuildNetwork(object):
    """Builds the network, nodes & arcs for problem"""
    __slots__ = ['commodities', 'network', 'data', 'shortest_paths']

    def __init__(self, problem_data: ProblemData):
        self.data = problem_data
        self.commodities = problem_data.commodities

        # build graph
        self.network = nx.DiGraph()

        for a, destinations in problem_data.network.items():
            for b, transit_time in destinations.items():
                self.network.add_edge(a, b, weight=transit_time, capacity=problem_data.capacities.get((a,b), 1.0), fixed_cost=problem_data.fixed_cost.get((a,b), transit_time), var_cost=problem_data.var_cost[0].get((a,b), 0))

        self.shortest_paths = nx.shortest_path_length(self.network, weight='weight')


    def create(self, time_points, paths=None):
        intervals = self.create_node_intervals(time_points)
        return (intervals, self.create_arcs(intervals, paths))

    ##
    ## Creates time intervals for each physical node based node/timepairs
    ##
    def create_node_intervals(self, nodes):
        output = tuplelist()

        for n in self.network.nodes():
            time_points = sorted(t for _,t in nodes.select(n, '*'))

            for t1,t2 in zip(time_points, time_points[1:]):
                output.append((n, t1, t2))

        return output

    ##
    ## Creates valid arcs (using lower bound) for each commodity and then shares 'redirected' arcs across all commodities
    ##
    def create_arcs(self, intervals, paths=None):
        intervals = tuplelist(sorted(intervals))  # ensure sorted (for performance reasons)
        cache = {n: intervals.select(n,'*', '*') for n in self.network.nodes()}

        redirected_arcs = []

        arcs = {k: tuplelist(self.lower_bound_arcs(c, intervals, cache, redirected_arcs)) if not paths or paths is None else self.fixed_path_arcs(paths[k], c, intervals)
                   for k,c in enumerate(self.commodities)}  # Create arcs ((n1, t1, t2), (n2, t3, t4)) pairs

        redirected_arcs = set(redirected_arcs)

        # add non-standard arcs created by commodities time windows
        for k,c in enumerate(self.commodities):
            origin,dest = c.a, c.b
            origin_to_arc = self.shortest_paths[origin[0]]
 
            #missing_arcs = set(a for k2,v2 in arcs.items() if k != k2 
            #                     for a in v2 if a not in arcs[k] and self.is_arc_valid2(a, origin, dest, origin_to_arc[a[0][0]], self.shortest_paths[a[0][0]][a[1][0]], self.shortest_paths[a[1][0]][dest[0]]))

            missing_arcs = set(a for a in redirected_arcs 
                                 if a not in arcs[k] and self.is_arc_valid2(a, origin, dest, origin_to_arc.get(a[0][0], None), self.shortest_paths[a[0][0]][a[1][0]], self.shortest_paths[a[1][0]].get(dest[0], None)))

            # add missing arcs if valid for k
            for a in missing_arcs:
                arcs[k].append(a)

        return arcs
    
    ##
    ## Creates all valid arcs for given discretization
    ##
    def lower_bound_arcs(self, commodity, intervals, interval_cache, redirected_arcs):
        origin,dest = commodity.a, commodity.b

        # setup storage arcs
        origin_to_arc = self.shortest_paths[origin[0]]
        arcs = [arc for n in self.network.nodes() if n in origin_to_arc and dest[0] in self.shortest_paths[n]
                    for arc in self.iterate_flow(intervals, n) if self.is_arc_valid2(arc, origin, dest, origin_to_arc.get(n, None), 0, self.shortest_paths[n].get(dest[0]))]

        #arcs = [arc for n, o2a, a2d in map(lambda n: (n, origin_to_arc.get(n, None), self.shortest_paths[n].get(dest[0], None)), self.network.nodes()) if o2a != None and a2d != None
        #            for arc in self.iterate_flow(intervals, n) if (arc[0] != None and arc[1] != None and (arc[1][0] != origin[0] or arc[0][0] == origin[0]) and (arc[0][0] != dest[0] or arc[1][0] == dest[0]) and (dest[1] - a2d >= origin[1] + o2a) and (origin[1] + o2a  < arc[0][2]) and (origin[1] + o2a < arc[1][2]) and (arc[1][1] + a2d <= dest[1]) and (arc[0][1] + a2d <= dest[1]))]


        # Faster way to do this is by using following node through intervals keeping a track on last used interval (can't go back in time)
        # that way lookup is much faster (i.e. not O(n^2))

        # for each physical arc
        for e in self.network.edges():
            it = iter(interval_cache[e[1]]) # for each destination interval
            i2 = next(it)

            transit_time = self.network[e[0]][e[1]]['weight']

            if e[0] not in self.shortest_paths[origin[0]] or dest[0] not in self.shortest_paths[e[1]]:
                continue

            origin_to_arc = self.shortest_paths[origin[0]][e[0]]
            arc_to_dest = self.shortest_paths[e[1]][dest[0]]
            arc_to_arc = self.shortest_paths[e[0]][e[1]]

            # for each origin interval
            for i1 in interval_cache[e[0]]:
                time = i1[1] + transit_time

                # the first / previously used interval is also valid for this interval
                if time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                    arcs.append((i1, i2))
                else:
                    # skip to correct interval
                    while i2 is not None and time >= i2[2]:
                        i2 = next(it, None)

                    # keep skipping if invalid, up until latest time
                    while i2 is not None and i1[2] + transit_time >= i2[2]:
                        if (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                            arcs.append((i1,i2))

                            # redirected
                            if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                                redirected_arcs.append((i1,i2))

                            break

                        i2 = next(it, None)

                    # we are done for this arc
                    if i2 is None:
                        break

                    # possibly the last transit time (from above loop) is valid
                    if i1[2] + transit_time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                        arcs.append((i1,i2))

                        # redirected
                        if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                            redirected_arcs.append((i1,i2))


        ## setup transport arcs
        #for i1 in intervals:
        #    for e in self.network.edges(i1[0]):
        #        time = i1[1] + self.network[i1[0]][e[1]]['weight']
        #        arc = (i1, next((i2 for i2 in interval_cache[e[1]] if time < i2[2] and self.is_arc_valid(commodity, (i1, i2))), None))

        #        if arc[1] != None:  # validity check is done in find_best_interval
        #            arcs.append(arc)

        return arcs

    # split the current intervals and then map new/current arcs
    def split_intervals(self, new_timepoints, current_intervals):
        new_intervals = {}
        original_intervals = tuplelist(current_intervals)  # create copy for lookup

        for node,time in sorted(new_timepoints):
            # find current interval that gets split by new timepoint - should always succeed if time >= 0 and time <= T
            interval = next((i for i in current_intervals.select(node, '*', '*') if time >= i[1] and time < i[2]), None)
            original_interval = next((i for i in original_intervals.select(node, '*', '*') if time >= i[1] and time < i[2]), None)
            assert interval is not None and original_interval is not None

            # split interval by renaming first and adding second
            i1 = (node, interval[1], time)
            i2 = (node, time, interval[2])

            if original_interval not in new_intervals:
                new_intervals[original_interval] = [i1, i2]
            else:
                new_intervals[original_interval].remove(interval)
                new_intervals[original_interval].append(i1)
                new_intervals[original_interval].append(i2)

            current_intervals.remove(interval)
            current_intervals.append(i1)
            current_intervals.append(i2)

        return new_intervals

    def create_new_arcs(self, n1, n2, origin, dest, origin_intervals, destination_intervals, redirected_arcs):
        new_arcs = []

        it = iter(destination_intervals) # for each destination interval
        i2 = next(it)

        transit_time = self.network[n1][n2]['weight']

        origin_to_arc = self.shortest_paths[origin[0]][n1]
        arc_to_dest = self.shortest_paths[n2][dest[0]]
        arc_to_arc = self.shortest_paths[n1][n2]

        # for each origin interval
        for i1 in origin_intervals:
            time = i1[1] + transit_time

            # the first / previously used interval is also valid for this interval
            if time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                new_arcs.append((i1, i2))
            else:
                # skip to correct interval
                while i2 is not None and time >= i2[2]:
                    i2 = next(it, None)

                # keep skipping if invalid, up until latest time
                while i2 is not None and i1[2] + transit_time >= i2[2]:
                    if (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                        new_arcs.append((i1,i2))

                        # redirected
                        if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                            redirected_arcs.append((i1,i2))

                        break

                    i2 = next(it, None)

                # we are done for this arc
                if i2 is None:
                    break

                # possibly the last transit time (from above loop) is valid
                if i1[2] + transit_time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1])):
                    new_arcs.append((i1,i2))

                    # redirected
                    if not (i1[1] + transit_time >= i2[1] and i1[1] + transit_time < i2[2]):
                        redirected_arcs.append((i1,i2))

        return list(set(new_arcs))


    def add_timepoints(self, new_timepoints, current_intervals, current_arcs):
        new_intervals = self.split_intervals(new_timepoints, current_intervals)
        interval_cache = {n: sorted(current_intervals.select(n,'*', '*')) for n in self.network.nodes()}

        # create / rename / delete arcs based on the new intervals
        new_arcs, ren_arcs, del_arcs = ([],[],[])
        redirected_arcs = []

        for k,c in enumerate(self.commodities):
            origin,dest = c.a, c.b

            new_arcs.append(set())
            ren_arcs.append({})
            del_arcs.append([])
            
            for original_interval, split_intervals in new_intervals.items():
                split_intervals.sort()  # enforce order

                origin_to_arc = self.shortest_paths[origin[0]].get(original_interval[0], None)
                arc_to_dest = self.shortest_paths[original_interval[0]].get(dest[0], None)

                ## new storage arcs
                storage_intervals = current_intervals.select(original_interval[0], '*', split_intervals[0][1]) + split_intervals + current_intervals.select(original_interval[0], split_intervals[-1][2], '*')  # connect to existing intervals

                storage_arcs = [arc for arc in zip(storage_intervals, storage_intervals[1:]) 
                                    if self.is_arc_valid2(arc, origin, dest, origin_to_arc, 0, arc_to_dest)]

                new_arcs[k].update(storage_arcs)
                current_arcs[k].extend(storage_arcs)

                #
                # outflow arcs
                #
                outflow = current_arcs[k].select(original_interval, '*')
                for arc in outflow:
                    current_arcs[k].remove(arc)
                    ren_arcs[k][arc] = None

                    # redirect storage arcs - arc closest to end of interval
                    if arc[0][0] == arc[1][0]:
                        ren_arcs[k][arc] = next((a for a in reversed(storage_arcs) if a not in ren_arcs[k].values()), None)
                        continue

                    new_interval_arcs = set(self.create_new_arcs(arc[0][0], arc[1][0], origin, dest, split_intervals, interval_cache[arc[1][0]], redirected_arcs))

                    if new_interval_arcs:
                        current_arcs[k].extend(list(new_interval_arcs.difference(new_arcs[k])))
                        new_arcs[k].update(new_interval_arcs)
                        ren_arcs[k][arc] = next((a for a in new_interval_arcs if a not in ren_arcs[k].values()), None)

                #
                # inflow arcs
                #
                for original_interval, split_intervals in new_intervals.items():
                    split_intervals.sort()  # enforce order

                    inflow = current_arcs[k].select('*', original_interval)
                    for arc in inflow:
                        current_arcs[k].remove(arc)
                        ren_arcs[k][arc] = None

                        # redirect storage arcs - closest arc to start of interval
                        if arc[0][0] == arc[1][0]:
                            ren_arcs[k][arc] = next((a for a in storage_arcs if a not in ren_arcs[k].values()), None)
                            continue

                        i1 = arc[0]
                        transit_time = self.network[arc[0][0]][arc[1][0]]['weight']
                        time = i1[1] + transit_time

                        origin_to_arc = self.shortest_paths[origin[0]][arc[0][0]]
                        arc_to_dest = self.shortest_paths[arc[1][0]][dest[0]]
                        arc_to_arc = self.shortest_paths[arc[0][0]][arc[1][0]]

                        # for each origin interval
                        tmp_arc = next(((i1, i2) for i2 in split_intervals if time < i2[2] and (i1 is not None and i2 is not None and (i2[0] != origin[0] or i1[0] == origin[0]) and (i1[0] != dest[0] or i2[0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < i1[2]) and (origin[1] + origin_to_arc + arc_to_arc < i2[2]) and (i2[1] + arc_to_dest <= dest[1]) and (i1[1] + arc_to_arc + arc_to_dest <= dest[1]))), None)

                        if tmp_arc is not None:
                            ren_arcs[k][arc] = tmp_arc
                            current_arcs[k].append(tmp_arc)
                            new_arcs[k].add(tmp_arc)

                            if tmp_arc[0][1] + arc_to_arc < tmp_arc[1][1]:
                                redirected_arcs.append(tmp_arc)
                        else:
                            del_arcs[k].append(arc)


        redirected_arcs = set(redirected_arcs)

        # add non-standard arcs created by commodities time windows
        for k,c in enumerate(self.commodities):
            origin,dest = c.a, c.b
            origin_to_arc = self.shortest_paths[origin[0]]
 
            missing_arcs = set(a for a in redirected_arcs 
                                 if a not in current_arcs[k] and self.is_arc_valid2(a, origin, dest, origin_to_arc.get(a[0][0], None), self.shortest_paths[a[0][0]][a[1][0]], self.shortest_paths[a[1][0]].get(dest[0], None)))

            # add missing arcs if valid for k
            for a in missing_arcs:
                current_arcs[k].append(a)
                new_arcs[k].add(a)

        return new_arcs, ren_arcs, del_arcs, new_intervals

    ##
    ## Creates arcs using a fixed path (used for IP2)
    ##
    def fixed_path_arcs(self, path, commodity, intervals):
        intervals = tuplelist(sorted(intervals))  # ensure sorted (for performance reasons)

        # setup storage arcs
        arcs = tuplelist([arc for n in self.network.nodes() for arc in self.iterate_flow(intervals, n) if self.is_arc_valid(commodity, arc)])
        
        # setup transport arcs
        edges = tuplelist(zip(path,path[1:]))

        for n in intervals:
            for e in edges.select(n[0], '*'):
                arc = (n, self.find_best_interval(commodity, intervals, e[1], n, n[1] + self.network[n[0]][e[1]]['weight']))

                if arc is not None and arc[1] is not None:  # validity check is done in find_best_interval
                    arcs.append(arc)

        return arcs

    # Same as is_arc_valid, but more optimized
    def is_arc_valid2(self, arc, origin, dest, origin_to_arc, arc_to_arc, arc_to_dest):
        return not ((arc[0] is None or arc[1] is None) or                                   # is valid node
                    (origin_to_arc is None or arc_to_dest is None) or                       # invalid path
                    (arc[1][0] == origin[0] and arc[0][0] != origin[0]) or                  # no inflow into origin (except storage arc)
                    (arc[0][0] == dest[0] and arc[1][0] != dest[0]) or                      # no outflow from destination (except storage arc)
                    (dest[1] - arc_to_dest < origin[1] + origin_to_arc + arc_to_arc) or     # cannot route via this arc using shortest paths (assumes transit time in arc)
                    (origin[1] + origin_to_arc >= arc[0][2]) or                             # cannot reach arc in time for dispatch
                    (origin[1] + origin_to_arc + arc_to_arc >= arc[1][2]) or                # arc is invalid due to actual dispatch time window
                    (arc[1][1] + arc_to_dest > dest[1]) or                                  # cannot reach destination in time - from i2
                    (arc[0][1] + arc_to_arc + arc_to_dest > dest[1]))                       # cannot reach destination in time - from i1

        #return (arc[0] != None and arc[1] != None and (arc[1][0] != origin[0] or arc[0][0] == origin[0]) and (arc[0][0] != dest[0] or arc[1][0] == dest[0]) and (dest[1] - arc_to_dest >= origin[1] + origin_to_arc + arc_to_arc) and (origin[1] + origin_to_arc < arc[0][2]) and (origin[1] + origin_to_arc + arc_to_arc < arc[1][2]) and (arc[1][1] + arc_to_dest <= dest[1]) and (arc[0][1] + arc_to_arc + arc_to_dest <= dest[1])) 


    ##
    ## Performs many checks to validate an arc
    ##
    def is_arc_valid(self, commodity, arc):
        origin,dest = commodity.a, commodity.b

        # is valid node
        if arc[0] is None or arc[1] is None:
            return False

        # no inflow into origin (except storage arc)
        if arc[1][0] == origin[0] and arc[0][0] != origin[0]:
            return False

        # no outflow from destination (except storage arc)
        if arc[0][0] == dest[0] and arc[1][0] != dest[0]:
            return False

        # path exists from origin to arc[0] node
        if arc[0][0] not in self.shortest_paths[origin[0]]:
            return False

        # path exists from arc[1] node to destination
        if dest[0] not in self.shortest_paths[arc[1][0]]:
            return False
        #
        # shortest path from origin -> destination using this arc is within time window
        #

        origin_to_arc = self.shortest_paths[origin[0]][arc[0][0]]
        arc_to_dest = self.shortest_paths[arc[1][0]][dest[0]]
        arc_to_arc = self.shortest_paths[arc[0][0]][arc[1][0]]

        # cannot route via this arc using shortest paths
        if dest[1] - arc_to_dest < origin[1] + origin_to_arc + arc_to_arc:
            return False

        # cannot reach arc in time for dispatch
        if origin[1] + origin_to_arc >= arc[0][2]:
            return False

        # arc is invalid due to actual dispatch time window
        if origin[1] + origin_to_arc + arc_to_arc >= arc[1][2]:
            return False

        # cannot reach destination in time - from i2
        if arc[1][1] + arc_to_dest > dest[1]:
            return False

        # cannot reach destination in time - from i1
        if arc[0][1] + arc_to_arc + arc_to_dest > dest[1]:
            return False

        return True

    
    def find_interval(self, nodes, node, time):
        for n in nodes.select(node, '*', '*'):
            if time >= n[1] and time < n[2]:
                return n

        return None

    def find_best_interval(self, c, sorted_intervals, node, n1, time):
        for n in sorted_intervals.select(node, '*', '*'):
            if time < n[2] and self.is_arc_valid(c, (n1, n)):
                return n
        return None
        #return self.find_interval(nodes, node, time)

    # returns a sequence of intervals for a node (in order of flow)
    def iterate_flow(self, sorted_intervals, node):
        tmp = sorted_intervals.select(node, '*', '*')
        return zip(tmp, tmp[1:])

    ##
    ## creates arcs/nodes for parameterized time discretization
    ##
    def discretization_network(self, step=1):
        #S = min([c.a[1] for c in self.commodities]) + 1
        T = max([c.b[1] for c in self.commodities]) + 1

        # Create nodes (n,t) pairs
        return tuplelist([(n,t) for n in self.network.nodes() for t in range(0, int(T+step), step)])

    ##
    ## creates arcs/nodes for time horizon
    ##
    def trivial_network(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # Create nodes (n,t) pairs
        return tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

    ##
    ## creates arcs/nodes for time horizon, plus origin/destination times
    ##
    def simple_network(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # Create nodes (n,t) pairs
        nodes = tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

        for k,c in enumerate(self.commodities):
            if c.a not in nodes:
                nodes.append(c.a)

            if c.b not in nodes:
                nodes.append(c.b)

        return nodes

    ##
    ## creates arcs/nodes for time horizon, plus origin/destination times, plus shortest path for each commodity
    ##
    def shortest_path_network(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # calculate all time points using shortest path for each commodity
        times = []

        for c in self.commodities:
            path = nx.shortest_path(self.network, c.a[0], c.b[0], weight='weight')
            t = c.a[1]
            times.append(c.b)

            for n1, n2 in zip(path, path[1:]):
                times.append((n1, t))
                t += self.network[n1][n2]['weight']

            times.append((c.b[0], t))

        # Create nodes (n,t) pairs
        nodes = tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

        for n in set(times):
            if n not in nodes:
                nodes.append(n)

        return nodes

    ##
    ## creates arcs/nodes for time horizon, plus origin/destination times, plus shortest path selected commodities
    ##
    def shortest_path_network_test(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # calculate all time points using shortest path for each commodity
        times = []

        for c in self.commodities:
            path = nx.shortest_path(self.network, c.a[0], c.b[0], weight='weight')

            shortest_time = float(sum(self.network[n1][n2]['weight'] for n1, n2 in zip(path, path[1:])))
            if (c.b[1] - c.a[1]) / shortest_time > 1.6:
                continue

            t = c.a[1]
            times.append(c.b)

            for n1, n2 in zip(path, path[1:]):
                times.append((n1, t))
                t += self.network[n1][n2]['weight']

            times.append((c.b[0], t))

        # Create nodes (n,t) pairs
        nodes = tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

        for n in set(times):
            if n not in nodes:
                nodes.append(n)

        return nodes

    ##
    ## creates arcs/nodes for time horizon, plus origin/destination times, plus shortest paths to all nodes (for each commodity)
    ##
    def all_shortest_path_network(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # calculate all time points using shortest paths
        shortest_paths = self.shortest_paths
        times = []

        for n in self.network.nodes():
            for c in self.commodities:
                if n in shortest_paths[c.a[0]] and c.b[0] in shortest_paths[n]:
                    earliest_to_n = c.a[1] + shortest_paths[c.a[0]][n]
                    latest_from_n = c.b[1] - shortest_paths[n][c.b[0]]

                    times.append((n, earliest_to_n))

                    # dont consider infeasible paths
                    if latest_from_n >= earliest_to_n:
                        times.append((n, latest_from_n))

        # Create nodes (n,t) pairs
        nodes = tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

        for n in set(times):
            if n not in nodes:
                nodes.append(n)

        return nodes

    ##
    ## creates arcs/nodes for time horizon, plus origin/destination times, plus 1st & 2nd shortest paths for each commodity
    ##
    def second_shortest_path_network(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # calculate all time points using shortest paths
        times = []

        for c in self.commodities:
            path = nx.shortest_path(self.network, c.a[0], c.b[0], weight='weight')
            t = c.a[1]
            times.append(c.b)

            for n1, n2 in zip(path, path[1:]):
                times.append((n1, t))
                t += self.network[n1][n2]['weight']

            times.append((c.b[0], t))

            # second shortest path
            path = second_shortest_path(self.network, c.a[0], c.b[0], path=path)[1]
            t = c.a[1]

            for n1, n2 in zip(path, path[1:]):
                times.append((n1, t))
                t += self.network[n1][n2]['weight']

            times.append((c.b[0], t))

        # Create nodes (n,t) pairs
        nodes = tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

        for n in set(times):
            if n not in nodes:
                nodes.append(n)

        return nodes

    ##
    ## creates arcs/nodes for time horizon, plus origin/destination times, plus 1st & 2nd shortest paths for each commodity
    ##
    def all_second_shortest_path_network(self):
        T = max([c.b[1] for c in self.commodities]) + 1

        # calculate all time points using shortest paths
        shortest_paths = self.shortest_paths
        times = []

        second_shortest_paths = {a: {b: second_shortest_path(self.network, a,b, shortest_paths[a][b])[0] 
                                     for b in self.network.nodes() if b in shortest_paths[a]}
                                 for a in self.network.nodes()}

        for n in self.network.nodes():
            for c in self.commodities:
                if n in shortest_paths[c.a[0]] and c.b[0] in shortest_paths[n]:
                    # shortest path
                    earliest_to_n = c.a[1] + shortest_paths[c.a[0]][n]
                    latest_from_n = c.b[1] - shortest_paths[n][c.b[0]]

                    times.append((n, earliest_to_n))

                    # dont consider infeasible paths
                    if latest_from_n >= earliest_to_n:
                        times.append((n, latest_from_n))

                    # second shortest path
                    v1 = second_shortest_paths[c.a[0]][n]
                    v2 = second_shortest_paths[n][c.b[0]]
                    
                    if v1 is not None and v2 is not None:
                        earliest_to_n = c.a[1] + v1
                        latest_from_n = c.b[1] - v2

                        if earliest_to_n < T:
                            times.append((n, earliest_to_n))

                        # dont consider infeasible paths
                        if latest_from_n >= earliest_to_n:
                            times.append((n, latest_from_n))

        # Create nodes (n,t) pairs
        nodes = tuplelist([(n,t) for n in self.network.nodes() for t in [0,T]])

        for n in set(times):
            if n not in nodes:
                nodes.append(n)

        return nodes



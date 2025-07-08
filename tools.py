import itertools
import functools
from operator import itemgetter

# zips a sequence on itself - "s -> (s0,s1), (s1,s2), (s2, s3), ..."
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

# zips a sequence on itself - "s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ..."
def triple(iterable):
    a, b, c = itertools.tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)

# itertools.accumulate is in python 3.x
def accumulate(iterator):
    total = 0
    for item in iterator:
        total += item
        yield total

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    return [_ for _ in iterable if pred(_) == False], [_ for _ in iterable if pred(_) == True]
#    t1, t2 = itertools.tee(iterable)
#    return list(itertools.filterfalse(pred, t1)), list(filter(pred, t2))

def limit_shortest_paths(G, source, target, weight='weight', cutoff=None):
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
                length -= G[visited[-1]][v][weight]
        else:
            t = G[visited[-1]][child][weight]

            if length + t <= cutoff:
                if child == target:
                    yield visited + [target]
                elif child not in visited:
                    visited.append(child)
                    stack.append(iter(G[child]))
                    length += t

# finds paths that are less than one metric, but greater than the other. i.e. cheap paths but greater than time window
def limit_path_range(G, source, target, lt_weight='weight', gt_weight='weight', less_than=None, greater_than=None):
    lt = 0
    gt = 0

    visited = [source]
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            v = visited.pop()
            if visited:
                lt -= G[visited[-1]][v][lt_weight]
                gt -= G[visited[-1]][v][gt_weight]
        else:
            t1 = G[visited[-1]][child][lt_weight]
            t2 = G[visited[-1]][child][gt_weight]    

            if lt + t1 < less_than:
                if child == target:
                    if gt + t2 > greater_than:
                        yield visited + [target]
                elif child not in visited:
                    visited.append(child)
                    stack.append(iter(G[child]))
                    lt += t1
                    gt += t2


# Anon objects
class Abj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


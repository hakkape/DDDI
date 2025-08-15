import itertools
import networkx as nx

# Improve type support for networkx
from collections.abc import Collection, Iterable, Hashable
from typing import Any, Generic, TypeVar

NodeType = TypeVar("NodeType", bound=Hashable)

class TypedDiGraph(Generic[NodeType]):
    def __init__(self) -> None:
        self.nx_graph = nx.DiGraph()

    @property
    def G(self):
        return self.nx_graph

    def add_edge(self, u_of_edge: NodeType, v_of_edge: NodeType, **attr: Any) -> None:
        self.nx_graph.add_edge(u_of_edge, v_of_edge, **attr)

    def edges(self, **kwargs: Any) -> Collection[tuple[NodeType, NodeType]]:
        return self.nx_graph.edges(data=False, **kwargs)  # type: ignore[no-any-return]

    def edges_data(self, **kwargs: Any) -> Iterable[tuple[NodeType, NodeType, dict[Any, Any]]]:
        return self.nx_graph.edges(data=True, **kwargs)  # type: ignore[no-any-return]

    def add_edges_from(self, edges_for_adding: Iterable[tuple[NodeType, NodeType, dict[Any, Any]]]) -> None:
        self.nx_graph.add_edges_from(edges_for_adding)

    def add_weighted_edges_from(self, edges_for_adding: Iterable[tuple[NodeType, NodeType, float]]) -> None:
        self.nx_graph.add_weighted_edges_from(edges_for_adding)

    def edge_data(self, n1: NodeType, n2: NodeType) -> dict[Any, Any]:
        return self.nx_graph.edges[(n1, n2)]
    
    def has_edge(self, u: NodeType, v: NodeType) -> bool:
        return self.nx_graph.has_edge(u, v)

    def remove_edge(self, u: NodeType, v: NodeType) -> None:
        self.nx_graph.remove_edge(u, v)

    def out_edges_data(self, n: NodeType, **kwargs: Any) -> Iterable[tuple[NodeType, NodeType, dict[Any, Any]]]:
        return self.nx_graph.out_edges(n, data=True, **kwargs)  # type: ignore[no-any-return]

    def in_edges_data(self, n: NodeType, **kwargs: Any) -> Iterable[tuple[NodeType, NodeType, dict[Any, Any]]]:
        return self.nx_graph.in_edges(n, data=True, **kwargs)  # type: ignore[no-any-return]

    def out_edges(self, n: NodeType, **kwargs: Any) -> Iterable[tuple[NodeType, NodeType]]:
        return self.nx_graph.out_edges(n, data=False, **kwargs)  # type: ignore[no-any-return]

    def in_edges(self, n: NodeType, **kwargs: Any) -> Iterable[tuple[NodeType, NodeType]]:
        return self.nx_graph.in_edges(n, data=False, **kwargs)  # type: ignore[no-any-return]

    def add_node(self, n: NodeType, **attr: Any) -> None:
        self.nx_graph.add_node(n, **attr)

    def add_nodes_from(self, nodes_for_adding: Iterable[NodeType]) -> None:
        self.nx_graph.add_nodes_from(nodes_for_adding)

    def nodes(self, **kwargs: Any) -> Collection[NodeType]:
        return self.nx_graph.nodes(data=False, **kwargs)  # type: ignore[no-any-return]

    def nodes_data(self, **kwargs: Any) -> Iterable[tuple[NodeType, dict[Any, Any]]]:
        return self.nx_graph.nodes(data=True, **kwargs)  # type: ignore[no-any-return]

    def node_data(self, n: NodeType) -> dict[Any, Any]:
        return self.nx_graph.nodes[n]

    def remove_node(self, n: NodeType) -> None:
        self.nx_graph.remove_node(n)

    def copy(self) -> "TypedDiGraph[NodeType]":
        new_graph = TypedDiGraph[NodeType]()
        new_graph.nx_graph = self.nx_graph.copy()
        return new_graph
    
    def subgraph(self, nodes: Iterable[NodeType]) -> "TypedDiGraph[NodeType]":
        new_graph = TypedDiGraph[NodeType]()
        new_graph.nx_graph = self.nx_graph.subgraph(nodes)
        return new_graph

    def has_path(self, source: NodeType, target: NodeType) -> bool:
        return nx.has_path(self.nx_graph, source, target)

    def has_node(self, n: NodeType) -> bool:
        return self.nx_graph.has_node(n)

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
    return [_ for _ in iterable if not pred(_)], [_ for _ in iterable if pred(_)]
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


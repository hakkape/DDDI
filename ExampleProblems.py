from ProblemData import ProblemData, Commodity, NodeTime
import inspect

class ExampleProblems(object):
    """Provides a list of example problems for testing"""

    # simple 3 node example (simple fail)
    simple = ProblemData(
        [
            Commodity(NodeTime(1, 0), NodeTime(2, 4), 0.5),
            Commodity(NodeTime(0, 0), NodeTime(2, 3), 0.5),
            Commodity(NodeTime(1, 1), NodeTime(0, 4), 0.5),
        ],
        {
            0: {1: 2, 2: 1},
            1: {0: 2, 2: 2},
            2: {0: 1, 1: 2},
        },
    )

    # simple 3 node example (simple fail)
    n3c3 = ProblemData(
        [
            Commodity(NodeTime(1, 4), NodeTime(2, 38), 0.5),
            Commodity(NodeTime(0, 6), NodeTime(2, 32), 0.5),
            Commodity(NodeTime(1, 8), NodeTime(0, 35), 0.5),
        ],
        {
            0: {1: 20, 2: 10},
            1: {0: 20, 2: 20},
            2: {0: 10, 1: 20},
        },
    )

    # simple 3 node example (simple fail), with var costs
    n3c3v = ProblemData(
        [
            Commodity(NodeTime(1, 0.4), NodeTime(2, 3.8), 0.5),
            Commodity(NodeTime(0, 0.6), NodeTime(2, 3.2), 0.5),
            Commodity(NodeTime(1, 0.8), NodeTime(0, 3.5), 0.5),
        ],
        {
            0: {1: 2.0, 2: 1.0},
            1: {0: 2.0, 2: 2.0},
            2: {0: 1.0, 1: 2.0},
        },
        None,
        {},
        {},
        [
            {
                (0, 1): 1,
                (0, 2): 1,
                (1, 0): 1,
                (1, 2): 1,
                (2, 0): 1,
                (2, 1): 1,
            }
        ] * 3,
    )

    # 3 node example with non-overlapping time windows and fails discretization > 1 time step
    n3c6 = ProblemData(
        [
            Commodity(NodeTime(1, 0), NodeTime(2, 38), 0.25),
            Commodity(NodeTime(0, 7), NodeTime(2, 32), 0.25),
            Commodity(NodeTime(1, 9), NodeTime(0, 35), 0.25),
            Commodity(NodeTime(2, 39), NodeTime(1, 78), 0.25),
            Commodity(NodeTime(0, 43), NodeTime(1, 72), 0.25),
            Commodity(NodeTime(0, 29), NodeTime(2, 75), 0.25),
        ],
        {
            0: {1: 19, 2: 11},
            1: {0: 19, 2: 23},
            2: {0: 11, 1: 23},
        },
    )

    # 4 node example (shortest path fail)
    n4c3 = ProblemData(
        [
            Commodity(NodeTime(2, 30), NodeTime(1, 160), 0.25),
            Commodity(NodeTime(3, 20), NodeTime(0, 150), 0.25),
            Commodity(NodeTime(1, 0), NodeTime(2, 180), 0.25),
        ],
        {
            0: {1: 60, 2: 40},
            1: {0: 60, 2: 70, 3: 50},
            2: {0: 40, 1: 70, 3: 30},
            3: {1: 50, 2: 30},
        },
    )

    # Continuous time
    continuous = ProblemData(
        [
            Commodity(NodeTime(2, 30.5), NodeTime(1, 160.7), 0.25),
            Commodity(NodeTime(3, 20.2), NodeTime(0, 150.3), 0.25),
            Commodity(NodeTime(1, 0.6), NodeTime(2, 180.1), 0.25),
        ],
        {
            0: {1: 60.3, 2: 40.1},
            1: {0: 60.3, 2: 70.8, 3: 50.7},
            2: {0: 40.1, 1: 70.8, 3: 30.3},
            3: {1: 50.7, 2: 30.3},
        },
    )

    # time travel fail - consolidation is infeasible, but time windows are valid
    time_travel_consolidations = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(3, 200), 0.25),
            Commodity(NodeTime(2, 0), NodeTime(1, 200), 0.25),
        ],
        {
            0: {1: 9},
            1: {2: 10},
            2: {3: 11},
            3: {0: 12},
        },
    )

    # time travel - with bad times cyclic tests
    time_travel_consolidations2 = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(3, 40), 0.25),
            Commodity(NodeTime(2, 0), NodeTime(1, 30), 0.25),
        ],
        {
            0: {1: 1},
            1: {2: 2},
            2: {3: 3},
            3: {0: 4},
        },
    )

    # complex 5 node example (second shortest path fail) - NOT YET!
    ##
    ## Why does shortest path fail, but not trivial etc????
    n5c4_hard = ProblemData(
        [
            Commodity(NodeTime(2, 0), NodeTime(3, 50), 0.25),
            Commodity(NodeTime(3, 0), NodeTime(1, 50), 0.25),
            Commodity(NodeTime(1, 0), NodeTime(3, 50), 0.25),
            Commodity(NodeTime(0, 0), NodeTime(2, 50), 0.25),
        ],
        {
            0: {1: 3, 2: 10, 4: 11},
            1: {0: 3, 3: 7, 4: 11},
            2: {0: 10, 3: 10, 4: 11},
            3: {1: 7, 2: 10, 4: 11},
            4: {0: 11, 1: 11, 2: 11, 3: 11},
        },
    )

    # more complex 5 node example (simple fail)
    n5c4 = ProblemData(
        [
            Commodity(NodeTime(1, 0), NodeTime(2, 40), 0.25),
            Commodity(NodeTime(0, 0), NodeTime(2, 30), 0.25),
            Commodity(NodeTime(3, 0), NodeTime(0, 30), 0.25),
            Commodity(NodeTime(1, 10), NodeTime(0, 40), 0.25),
        ],
        {
            0: {1: 20, 2: 10, 4: 10},
            1: {0: 20, 3: 10, 4: 10},
            2: {0: 10, 3: 10, 4: 10},
            3: {1: 10, 2: 10, 4: 10},
            4: {0: 10, 1: 10, 2: 10, 3: 10},
        },
    )

    # simple path length fail
    path_fail = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 35), 0.5),
        ],
        {
            0: {1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
            1: {2: 10, 3: 10, 4: 10, 5: 10},
            2: {3: 10, 4: 10, 5: 10},
            3: {4: 10, 5: 10},
            4: {5: 10},
            5: {},
        },
        None,
        {},
        {
            (0, 1): 1,
            (0, 2): 20,
            (0, 3): 30,
            (0, 4): 40,
            (0, 5): 50,
            (1, 2): 1,
            (1, 3): 20,
            (1, 4): 30,
            (1, 5): 40,
            (2, 3): 1,
            (2, 4): 20,
            (2, 5): 30,
            (3, 4): 1,
            (3, 5): 20,
            (4, 5): 1,
        },
    )

    # path fail, that does not require time points at earliest dispatch, i.e. (2,[8-10]) will work
    path_early_dispatch = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(4, 17), 0.5),
        ],
        {
            0: {1: 5, 2: 5, 3: 5},
            1: {2: 5},
            2: {3: 5, 4: 5},
            3: {4: 5},
            4: {},
        },
        None,
        {},
        {
            (0, 3): 100,
            (0, 2): 100,
            (2, 4): 100,
        },
    )

    # path fail, requiring more than one time point
    path_multiple = ProblemData(
        [
            Commodity(NodeTime(0, 3), NodeTime(3, 10), 0.5),
        ],
        {
            0: {1: 3, 2: 2, 3: 2, 4: 1},
            1: {2: 1, 3: 4},
            2: {3: 4, 5: 1},
            3: {},
            4: {1: 1},
            5: {3: 2},
        },
        None,
        {},
        {
            (0, 4): 100,
            (0, 3): 100,
            (0, 2): 100,
            (2, 5): 100,
            (1, 3): 100,
        },
    )

    # path fail requires multiple time points to fix.  Could also force with one commodity, but cheaper path [0,1,4,3,2]
    simple_path_fail = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(2, 30), 0.5),
            Commodity(NodeTime(0, 0), NodeTime(1, 40), 0.5),
            Commodity(NodeTime(1, 0), NodeTime(4, 40), 0.5),
            Commodity(NodeTime(4, 0), NodeTime(3, 40), 0.5),
            Commodity(NodeTime(3, 0), NodeTime(2, 40), 0.5),
        ],
        {
            0: {6: 5, 3: 10, 4: 10, 5: 5},
            1: {0: 10, 2: 10, 4: 10},
            2: {1: 10, 3: 10, 4: 10},
            3: {0: 10, 2: 10, 4: 10},
            4: {0: 10, 1: 10, 2: 10, 3: 10},
            5: {1: 4},
            6: {1: 5},
        },
        None,
        {},
        {
            (0, 5): 10,
        },
    )

    # path fail requires multiple time points to fix.  Could also force with one commodity, but cheaper path [0,1,4,3,2]
    # this works with one point (3,t) 10<t<=19
    simple_path_fail_single_commodity = ProblemData(
        [
            Commodity(NodeTime(0, 0.8), NodeTime(6, 32), 0.5),
        ],
        {
            0: {1: 5, 4: 10.5, 2: 5},
            1: {3: 5},
            2: {3: 4},
            3: {0: 10, 6: 17, 4: 10},
            4: {0: 10, 3: 10, 6: 10, 5: 10.5},
            5: {6: 10, 4: 10},
            6: {3: 10, 5: 10, 4: 10},
        },
        None,
        {},
        {
            (0, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (5, 6): 1,
        },
    )

    # path fail. Testing when multiple equal shortest paths
    path_fail_single_commodity_equal_shortest_paths = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 31), 0.5),
        ],
        {
            0: {2: 10, 3: 10.5, 1: 5},
            1: {2: 5},
            2: {0: 10, 5: 17, 3: 10},
            3: {0: 10, 2: 10, 5: 10, 4: 10.5},
            4: {5: 10, 3: 10},
            5: {2: 10, 4: 10, 3: 10},
        },
        None,
        {},
        {
            (0, 2): 1,
            (2, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
        },
    )

    # path fail requires multiple time points to fix, with equal shortest paths (early late).
    path_fail_single_commodity_multiple_timepoints_shortest_paths = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(6, 35), 0.5),
        ],
        {
            0: {2: 10, 4: 10, 3: 10, 1: 5},
            1: {2: 5},
            2: {0: 10, 5: 10, 3: 10},
            3: {0: 10, 2: 10, 5: 10, 4: 10},
            4: {0: 10, 5: 10, 3: 10, 6: 1},
            5: {2: 10, 4: 10, 3: 10, 6: 10},
            6: {5: 10},
        },
        None,
        {},
        {
            (0, 2): 1,
            (2, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (5, 6): 1,
            (4, 6): 1000,
        },
    )

    # path fail requires 4 time points to fix (using transit times)
    # 3 time points using cumulative shortest paths
    # ? time point using shortest path + transit e^k + Y_{o^k,n1} + T_{n1,n2}
    path_fail_single_commodity_multiple_timepoints = ProblemData(
        [
            Commodity(NodeTime(0, 3), NodeTime(8, 48), 0.5),
        ],
        {
            0: {1: 5, 5: 10, 4: 10, 2: 5},
            1: {3: 5},
            2: {3: 4},
            3: {6: 10, 4: 10},
            4: {0: 10, 3: 10, 6: 9, 5: 14},
            5: {0: 10, 6: 10, 4: 14, 7: 1},
            6: {3: 10, 5: 10, 4: 9, 7: 10, 8: 1},
            7: {6: 10, 8: 10},
            8: {7: 10},
        },
        None,
        {},
        {
            (0, 1): 1,
            (1, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (5, 6): 1,
            (6, 7): 1,
            (5, 7): 1000,
            (6, 8): 1000,
        },
    )

    # path fail requires 3 time points to fix (using transit times)
    # 2 time points using cumulative shortest paths
    # 1 time point using shortest path + transit e^k + Y_{o^k,n1} + T_{n1,n2}
    # when this was not triangle (removed node 8) it caused multiple time points
    path_fail_single_commodity_3_timepoints = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(6, 35), 0.5),
        ],
        {
            0: {8: 5, 4: 10, 3: 10, 1: 5},
            1: {2: 4},
            2: {0: 10, 5: 10, 3: 10},
            3: {0: 10, 2: 10, 5: 10, 4: 14},
            4: {0: 10, 5: 10, 3: 14, 6: 1},
            5: {2: 10, 4: 10, 3: 10, 6: 10},
            6: {5: 10},
            8: {2: 5},
        },
        None,
        {},
        {
            (0, 2): 1,
            (2, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (5, 6): 1,
            (4, 6): 1000,
        },
    )

    # FAILED to fix path issue using transit times - infinite loop (problem is with 10.'2')
    infinite_path_fail_single_commodity_3_timepoints = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(6, 35), 0.5),
        ],
        {
            0: {2: 9, 4: 10, 3: 10, 1: 5},
            1: {2: 4},
            2: {0: 10, 5: 10, 3: 10.2},
            3: {0: 10, 2: 10, 5: 10, 4: 10},
            4: {0: 10, 5: 10, 3: 10, 6: 1},
            5: {2: 10, 4: 10, 3: 10, 6: 10},
            6: {5: 10},
        },
        None,
        {},
        {
            (0, 2): 1,
            (2, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (5, 6): 1,
            (4, 6): 1000,
        },
    )

    # current W approach does not break consolidation [with timepoints.extend([(4, 20), (3, 30)])]
    multiple_point_consolidation_fail = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 45), 0.5),
            Commodity(NodeTime(0, 0), NodeTime(1, 35), 0.5),
            Commodity(NodeTime(1, 0), NodeTime(2, 35), 0.5),
            Commodity(NodeTime(2, 0), NodeTime(3, 35), 0.5),
            Commodity(NodeTime(3, 0), NodeTime(5, 35), 0.5),
        ],
        {
            0: {1: 10, 2: 10},
            1: {2: 10},
            2: {5: 10, 4: 10, 3: 4},
            3: {4: 5},
            4: {5: 10},
        },
        [
            [0, -0.5],
            [1, 0],
            [2, 0],
            [3, 0.5],
            [3.5, -0.5],
            [4, 0],
        ],
        {},
        {
            # (2,4): 5
        },
    )

    # simple path length fail - broken
    simple_window_fail = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(4, 40), 0.25),
            Commodity(NodeTime(3, 0), NodeTime(4, 15), 0.25),
        ],
        {
            0: {1: 10, 2: 15},
            1: {2: 10},
            2: {3: 10},
            3: {4: 10},
        },
        None,
        {},
        {
            (0, 2): 21,
        },
    )

    # simple path length fail
    simple_window_fail2 = ProblemData(
        [
            Commodity(NodeTime(5, 0), NodeTime(0, 50), 0.25),
            Commodity(NodeTime(4, 15), NodeTime(3, 25), 0.25),
        ],
        {
            0: {},
            1: {0: 10},
            2: {1: 10, 0: 1},
            3: {2: 10},
            4: {3: 10},
            5: {4: 10},
        },
        None,
        {},
        {
            (2, 0): 45,
        },
    )

    # window fail - Uses cheap longer paths. 
    window_fail = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 50), 0.25),
            Commodity(NodeTime(6, 0), NodeTime(5, 50), 0.25),
            Commodity(NodeTime(3, 0), NodeTime(4, 15), 0.25),
        ],
        {
            0: {1: 10, 2: 15, 3: 25},
            1: {2: 10},
            2: {3: 10},
            3: {4: 10},
            4: {5: 10},
            5: {},
            6: {7: 10, 8: 1, 3: 1},
            7: {8: 10},
            8: {3: 10},
        },
        None,
        {},
        {
            (0, 3): 45,
            (0, 2): 45,
            (3, 4): 9,
        },
    )

    # Window fail - Requires multiple timepoints to break invalid consolidation (uses cheap longer paths to force issue)
    window_fail2 = ProblemData(
        [
            Commodity(NodeTime(5, 0), NodeTime(0, 50), 0.25),
            Commodity(NodeTime(5, 0), NodeTime(8, 50), 0.25),
            Commodity(NodeTime(4, 15), NodeTime(3, 25), 0.25),
        ],
        {
            0: {},
            1: {0: 10},
            2: {1: 10, 0: 1},
            3: {2: 10, 0: 1, 6: 10, 8: 1},
            4: {3: 10},
            5: {4: 10},
            6: {7: 10, 8: 1},
            7: {8: 10},
            8: {},
        },
        None,
        {},
        {
            (3, 0): 45,
            (2, 0): 45,
            (4, 3): 9,
            (3, 8): 45,
            (2, 8): 45,
        },
    )

    # example where 2 conflicting consolidations are either side of reported
    mutex_simple = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(2, 20), 0.25),
            Commodity(NodeTime(1, 0), NodeTime(3, 30), 0.25),
            Commodity(NodeTime(2, 0), NodeTime(4, 50), 0.25),
            Commodity(NodeTime(3, 25), NodeTime(4, 35), 0.25),
        ],
        {
            0: {1: 10},
            1: {2: 10},
            2: {3: 9},
            3: {4: 10},
        },
    )

    # a mutual consolidation with short paths at start/end, does it still break?
    # time points [(14,13),(3,18),(4,23),(5,28),(6,33),(7,38),(8,43),(9,48),(10,53)]
    middle_mutual = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(11, 60), 0.25),
            Commodity(NodeTime(12, 3), NodeTime(7, 40), 0.25),
            Commodity(NodeTime(15, 20), NodeTime(8, 41), 0.25),
        ],
        {
            0: {1: 5, 2: 1, 3: 1},
            1: {2: 5, 3: 1},
            2: {3: 5},
            3: {4: 5},
            4: {5: 5, 6: 1, 7: 1},
            5: {6: 5, 7: 1},
            6: {7: 5},
            7: {8: 5},
            8: {9: 5, 10: 1, 11: 1},
            9: {10: 5, 11: 1},
            10: {11: 5},
            12: {13: 5, 14: 1, 3: 1},
            13: {14: 5, 3: 1},
            14: {3: 5},
            15: {16: 5, 17: 1, 7: 1},
            16: {17: 5, 7: 1},
            17: {7: 5},
        },
        None,
        {},
        {
            (0, 3): 100,
            (0, 2): 100,
            (1, 3): 100,
            (12, 3): 100,
            (12, 14): 100,
            (13, 3): 100,
            (4, 7): 100,
            (4, 6): 100,
            (5, 7): 100,
            (8, 11): 100,
            (8, 10): 100,
            (9, 11): 100,
            (15, 7): 100,
            (15, 17): 100,
            (16, 7): 100,
        },
    )

    # this problem is designed to show that we need to recusively add timepoints to stop mutual consolidations
    # doesn't quite work as expected.  want to stop a simple 'shortest path' from root/tail 'early'>'late' times.
    # * need to incorporate shortest path to consolidation some how
    recursive_mutual = ProblemData(
        [
            Commodity(NodeTime(0, 3), NodeTime(14, 39), 0.25),
            Commodity(NodeTime(11, 1), NodeTime(17, 43), 0.25),
            Commodity(NodeTime(14, 5), NodeTime(20, 49), 0.25),
            Commodity(NodeTime(17, 0), NodeTime(10, 53), 0.25),
            Commodity(NodeTime(20, 0), NodeTime(23, 52), 0.25),
            # Commodity(NodeTime(21,22), NodeTime(7,40), 0.25)
        ],
        {
            0: {1: 5, 2: 1, 3: 1},
            1: {2: 5, 3: 1},
            2: {3: 5},
            11: {12: 5, 13: 1, 3: 1},
            12: {13: 5, 3: 1},
            13: {3: 5},
            3: {4: 5},
            4: {5: 5, 16: 5, 15: 1, 14: 1},
            5: {6: 5, 19: 5, 18: 1, 17: 1},
            6: {7: 5, 22: 5, 21: 1, 20: 1},
            7: {8: 5, 9: 1, 10: 1, 25: 5, 24: 1, 23: 1},
            8: {9: 5, 10: 1},
            9: {10: 5},
            14: {15: 5, 16: 1, 4: 1},
            15: {16: 5, 4: 1, 14: 5},
            16: {4: 5, 14: 1, 15: 5},
            17: {18: 5, 19: 1, 5: 1},
            18: {19: 5, 5: 1, 17: 5},
            19: {5: 5, 17: 1, 18: 5},
            20: {21: 5, 22: 1, 6: 1},
            21: {22: 5, 6: 1, 20: 5},
            22: {6: 5, 20: 1, 21: 5, 23: 1},
            25: {24: 5, 23: 1},
            24: {23: 5},
        },
        None,
        {},
        {
            (0, 3): 100,
            (0, 2): 100,
            (1, 3): 100,
            (11, 3): 100,
            (11, 13): 100,
            (12, 3): 100,
            (4, 14): 100,
            (14, 4): 100,
            (4, 15): 100,
            (15, 4): 100,
            (16, 14): 100,
            (14, 16): 100,
            (5, 17): 100,
            (17, 5): 100,
            (5, 18): 100,
            (18, 5): 100,
            (19, 17): 100,
            (17, 19): 100,
            (6, 20): 100,
            (20, 6): 100,
            (6, 21): 100,
            (21, 6): 100,
            (22, 20): 100,
            (20, 22): 100,
            (7, 10): 100,
            (7, 9): 100,
            (8, 10): 100,
            (7, 23): 100,
            (7, 24): 100,
            (25, 23): 100,
            (22, 23): 100,
        },
    )

    # a simple time window consolidation with short paths at start/end, does it still break?
    middle_window = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(11, 60), 0.25),
            Commodity(NodeTime(12, 3), NodeTime(7, 40), 0.25),
            Commodity(NodeTime(15, 15), NodeTime(11, 51), 0.25),
        ],
        {
            0: {1: 5, 2: 1, 3: 1},
            1: {2: 5, 3: 1},
            2: {3: 5},
            3: {4: 5},
            4: {5: 5, 6: 1, 7: 1},
            5: {6: 5, 7: 1},
            6: {7: 5},
            7: {8: 5},
            8: {9: 5, 10: 1, 11: 1},
            9: {10: 5, 11: 1},
            10: {11: 5},
            12: {13: 5, 14: 1, 3: 1},
            13: {14: 5, 3: 1},
            14: {3: 5},
            15: {16: 5, 17: 1, 7: 1},
            16: {17: 5, 7: 1},
            17: {7: 5},
        },
        None,
        {},
        {
            (0, 3): 100,
            (0, 2): 100,
            (1, 3): 100,
            (12, 3): 100,
            (12, 14): 100,
            (13, 3): 100,
            (4, 7): 100,
            (4, 6): 100,
            (5, 7): 100,
            (8, 11): 100,
            (8, 10): 100,
            (9, 11): 100,
            (15, 7): 100,
            (15, 17): 100,
            (16, 7): 100,
        },
    )

    # cycle with shortest paths at start/end
    middle_cycle = ProblemData(
        [
            Commodity(NodeTime(4, 0), NodeTime(9, 90), 0.25),
            Commodity(NodeTime(10, 0), NodeTime(15, 400), 0.25),
            Commodity(NodeTime(0, 0), NodeTime(1, 80), 0.25),
        ],
        {
            0: {1: 9},
            1: {2: 10, 14: 1, 15: 1},
            2: {3: 11},
            3: {0: 12, 7: 5, 9: 1},
            4: {5: 5, 0: 1, 6: 1},
            5: {6: 5, 0: 1},
            6: {0: 5},
            7: {8: 5, 9: 1},
            8: {9: 5},
            10: {11: 5, 2: 1, 12: 1},
            11: {12: 5, 2: 1},
            12: {2: 5},
            13: {14: 5, 15: 1},
            14: {15: 5},
        },
        None,
        {},
        {
            (4, 0): 100,
            (4, 6): 100,
            (5, 0): 100,
            (3, 9): 100,
            (3, 8): 100,
            (7, 9): 100,
            (1, 14): 100,
            (1, 15): 100,
            (13, 15): 100,
            (10, 2): 100,
            (10, 12): 100,
            (11, 2): 100,
            (0, 1): 1000,
            (2, 3): 1000,
        },
    )

    ms_test_conflict = ProblemData(
        [
            Commodity(NodeTime(5, 10), NodeTime(7, 30), 0.25),
            Commodity(NodeTime(0, 5), NodeTime(2, 25), 0.25),
            Commodity(NodeTime(6, 0), NodeTime(4, 50), 0.25),
            Commodity(NodeTime(1, 0), NodeTime(4, 45), 0.25),
        ],
        {
            0: {1: 10},
            1: {2: 10},
            2: {3: 10},
            3: {4: 10},
            5: {6: 10},
            6: {7: 10},
            7: {3: 10},
        },
    )

    ms_test1 = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 16), 0.5),
            Commodity(NodeTime(2, 0), NodeTime(7, 16), 0.5),
        ],
        {
            0: {1: 3, 3: 6},
            1: {3: 4},
            2: {1: 3, 3: 6},
            3: {4: 2},
            4: {5: 6, 6: 4, 7: 6},
            6: {5: 3, 7: 3},
        },
    )

    ms_test2 = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 16), 0.5),
            Commodity(NodeTime(2, 0), NodeTime(7, 15), 0.5),
        ],
        {
            0: {1: 3, 3: 6},
            1: {3: 4},
            2: {1: 3, 3: 6},
            3: {4: 2},
            4: {5: 6, 6: 4, 7: 6},
            6: {5: 3, 7: 3},
        },
    )

    ms_test3 = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 20), 0.5),
            Commodity(NodeTime(2, 2), NodeTime(7, 18), 0.5),
        ],
        {
            0: {1: 3, 3: 6},
            1: {3: 4},
            2: {1: 3, 3: 6},
            3: {4: 2},
            4: {5: 6, 6: 4, 7: 6},
            6: {5: 3, 7: 3},
        },
    )

    ms_test4 = ProblemData(
        [
            Commodity(NodeTime(0, 27), NodeTime(3, 60), 0.5),
            Commodity(NodeTime(1, 20), NodeTime(7, 80), 0.5),
            Commodity(NodeTime(5, 50), NodeTime(8, 85), 0.5),
        ],
        {
            0: {1: 10},
            1: {2: 10},
            2: {3: 10, 4: 10},
            3: {},
            4: {6: 10},
            5: {6: 10},
            6: {7: 10},
            7: {8: 10},
            8: {},
        },
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [20, 10],
            [10, 20],
            [0, 30],
            [10, 30],
            [10, 40],
            [20, 40],
        ],
    )

    ms_test5 = ProblemData(
        [
            Commodity(NodeTime(0, 37), NodeTime(1, 50), 0.5),
            Commodity(NodeTime(0, 20), NodeTime(3, 85), 0.5),
            Commodity(NodeTime(2, 60), NodeTime(3, 75), 0.5),
        ],
        {
            0: {1: 10},
            1: {2: 20},
            2: {3: 10},
            3: {},
        },
        [
            [0, 0],
            [0, 10],
            [0, 30],
            [0, 40],
        ],
    )

    dod = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(5, 100), 0.5),
            Commodity(NodeTime(2, 2), NodeTime(7, 180), 0.5),
        ],
        {
            0: {1: 21, 4: 21, 5: 6},
            1: {0: 21, 2: 21, 7: 6},
            2: {1: 21, 3: 21, 9: 6},
            3: {2: 21, 4: 21, 11: 6},
            4: {0: 21, 3: 21, 13: 6},
            5: {0: 6, 6: 7, 14: 7},
            6: {5: 7, 7: 7, 15: 4},
            7: {1: 6, 6: 7, 8: 7},
            8: {7: 7, 9: 7, 16: 4},
            9: {2: 6, 8: 7, 10: 7},
            10: {9: 7, 11: 7, 17: 4},
            11: {3: 6, 10: 7, 12: 7},
            12: {11: 7, 13: 7, 18: 4},
            13: {4: 6, 12: 7, 14: 7},
            14: {5: 7, 13: 7, 19: 4},
            15: {6: 4, 16: 6, 19: 6},
            16: {8: 4, 15: 6, 17: 6},
            17: {10: 4, 16: 6, 18: 6},
            18: {12: 4, 17: 6, 19: 6},
            19: {14: 4, 15: 6, 18: 6},
        },
        [
            [200, 375],
            [385, 239],
            [316, 21],
            [85, 21],
            [15, 239],
            [200, 315],
            [254, 254],
            [327, 219],
            [287, 148],
            [279, 70],
            [200, 85],
            [120, 70],
            [110, 153],
            [73, 220],
            [147, 255],
            [231, 223],
            [251, 162],
            [198, 124],
            [147, 163],
            [169, 223],
        ],
    )

    visualize_solution_example = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(4, 50), 0.2),
            Commodity(NodeTime(0, 0), NodeTime(4, 50), 0.2),
            Commodity(NodeTime(1, 0), NodeTime(4, 50), 0.2),
            Commodity(NodeTime(1, 0), NodeTime(5, 50), 0.2),
        ],
        {
            0: {2: 5},
            1: {2: 5},
            2: {3: 5},
            3: {4: 5, 5: 5},
        },
    )

    # tests mutual consolidations from diferent root paths 
    mutual_2_root_nodes = ProblemData(
        [
            Commodity(NodeTime(0, 7), NodeTime(3, 12), 0.2),
            Commodity(NodeTime(0, 0), NodeTime(4, 17), 0.2),
            Commodity(NodeTime(1, 0), NodeTime(4, 17), 0.2),
            Commodity(NodeTime(1, 1), NodeTime(2, 6), 0.2),
            Commodity(NodeTime(3, 10), NodeTime(4, 15), 0.2),
        ],
        {
            0: {3: 5},
            1: {2: 5},
            2: {3: 5},
            3: {4: 5, 5: 5},
        },
        None,
        {},
        {
            (0, 3): 45,
        },
    )

    # multiple time points required from different commodity paths to break solution
    tp_along_multiple_commodities_path = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(2, 20), 0.2),
            Commodity(NodeTime(1, 0), NodeTime(3, 30), 0.2),
            Commodity(NodeTime(2, 0), NodeTime(4, 50), 0.2),
            Commodity(NodeTime(3, 25), NodeTime(4, 35), 0.2),
        ],
        {
            0: {1: 10},
            1: {2: 10},
            2: {3: 9},
            3: {4: 10},
        },
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
    )

    # used in the price of discretization
    paper_example = ProblemData(
        [
            Commodity(NodeTime(1, 20), NodeTime(2, 130), 0.25),
            Commodity(NodeTime(1, 80), NodeTime(0, 140), 0.25),
            Commodity(NodeTime(0, 0), NodeTime(2, 100), 0.25),
        ],
        {
            0: {1: 17, 2: 5, 3: 3},
            1: {0: 17, 2: 9, 3: 3},
            2: {0: 5, 1: 9},
            3: {0: 3, 1: 3},
        },
        [[0, 0], [1, 1], [1, 0], [0, 1]],
    )

    paper_example2 = ProblemData(
        [
            Commodity(NodeTime(1, 2), NodeTime(2, 19), 0.5),
            Commodity(NodeTime(1, 0), NodeTime(2, 22), 0.5),
        ],
        {
            0: {1: 9, 2: 4},
            1: {0: 9, 2: 11},
            2: {0: 4, 1: 11},
        },
    )

    # try to exploit allow splitting, split commodity takes the same path but at different times
    split_freight = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(1, 5), 0.9),
            Commodity(NodeTime(0, 0), NodeTime(1, 10), 0.2),
            Commodity(NodeTime(0, 5), NodeTime(1, 10), 0.9),
        ],
        {
            0: {1: 2},
            1: {0: 2},
        },
    )

    # try to exploit allow splitting, split commodity takes different paths at same time
    split_freight_path = ProblemData(
        [
            Commodity(NodeTime(0, 0), NodeTime(1, 5), 0.9),
            Commodity(NodeTime(0, 0), NodeTime(1, 10), 0.2),
            Commodity(NodeTime(0, 1), NodeTime(2, 10), 0.9),
        ],
        {
            0: {1: 5, 2: 5},
            1: {},
            2: {1: 1},
        },
    )

    #mh_test = ProblemData.read_file('tests/1zone_3.txt')

    # get all class members via reflection
    @staticmethod
    def all_problems():
        return inspect.getmembers(ExampleProblems, lambda m: isinstance(m, ProblemData))

import numpy as np

from stl_constrained_zpc.scripts.reachability.Interval import Interval


class IntervalOperations:
    """
    Class for performing operations on intervals. This class provides methods to compute the intersection of two intervals.

    Attributes:
        None

    Methods:
        intersection_intervals(interval_1, interval_2):
            Computes the intersection of two intervals represented as numpy arrays.

        intersection_intervals_v2(interval_1, interval_2):
            Computes the intersection of two intervals represented as Interval objects.
    """
    def __init__(self):
        pass

    def intersection_intervals(self, interval_1, interval_2):
        l1 = min(interval_1[0], interval_1[1])
        r1 = max(interval_1[0], interval_1[1])        

        l2 = min(interval_2[0], interval_2[1])
        r2 = max(interval_2[0], interval_2[1])        

        if (l2 > r1 or r2 < l1):
            pass

        # Else update the intersection69
        else:
            l1 = max(l1, l2)
            r1 = min(r1, r2)

        return np.array([l1, r1])
    
    def intersection_intervals_v2(self, interval_1, interval_2):
        l1 = min(interval_1.inf, interval_1.sup)
        r1 = max(interval_1.inf, interval_1.sup)        

        l2 = min(interval_2.inf, interval_2.sup)
        r2 = max(interval_2.inf, interval_2.sup)        

        if (l2 > r1 or r2 < l1):
            pass

        # Else update the intersection69
        else:
            l1 = max(l1, l2)
            r1 = min(r1, r2)

        return Interval(l1, r1)



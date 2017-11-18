import numpy as np
import math
import random

CONSENSUS_THRESHOLD = 2000

MAX_INLIER_DISTANCE = 0.1

def ransac(points, iterations):
    # sample = points[random.sample(range(len(points)), 100000)]
    sample = points
    (best_plane, best_consensus) = ([], [])
    for _ in range(iterations):
        (plane, consensus) = fit_plane(sample)
        if (len(consensus) > CONSENSUS_THRESHOLD and len(consensus) > len(best_consensus)):
            (best_plane, best_consensus) = (plane, consensus)

    return best_consensus


# equation of a plane in python hints (incomplete)
# aimed at 2017/18 L3 SSA students at DU
def fit_plane(points):
    # how to - select 3 non-colinear points
    cross_product_check = np.array([0,0,0]);
    while np.all(cross_product_check == 0):
        (P1,P2,P3) = points[random.sample(range(len(points)), 3)]
        # make sure they are non-collinear
        cross_product_check = np.cross(P1-P2, P2-P3);

    # how to - calculate plane coefficients from these points

    coefficients_abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))

    # check plane is roughly horizontal
    if (coefficients_abc[1] < 0.5):
        return ([],[])

    coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0]+coefficients_abc[1]*coefficients_abc[1]+coefficients_abc[2]*coefficients_abc[2])

    # how to - measure distance of all points from plane given the plane coefficients calculated

    # dist = abs((np.dot(sample, coefficients_abc) - 1)/coefficient_d)

    dist = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d)
    (row, _ )= np.where(dist < MAX_INLIER_DISTANCE)
    inliers = points[row]
    # avg_dist = np.average(dist[row])

    return (np.array([P1,P2,P3]), inliers)

############################################

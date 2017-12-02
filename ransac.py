import numpy as np
import math
import random

CONSENSUS_THRESHOLD = 2000 # number of consensus points required for a plane to be considered
ITERATIONS = 200 # number of ransac iterations to run each time
MAX_INLIER_DISTANCE = 0.1 # consider all points within this distance to be inliers


def ransac(points, previous_plane):
    """ Fit a plane to a set of 3D points using RANdom SAmple and Consensus

    Arguments:
        points         -- list of 3D points
        previous_plane -- best plane estimate for previous pair of images
                          used to improve results by considering an average
                          between this and best ransac plane found for current image pair
    """
    # this can actually happen if you downsample significantly
    if (len(points) < 4):
        return None, None, None

    (best_normal, best_plane, best_consensus) = ([], [] ,[])
    for _ in range(ITERATIONS):
        # randomly select a plane and return its normal and list of consensus points
        (normal, plane, consensus) = fit_plane(points)
        # update best plane estimate if number of consensus points is sufficiently high
        if (len(consensus) > CONSENSUS_THRESHOLD and len(consensus) > len(best_consensus)):
            (best_normal, best_plane, best_consensus) = (normal,  plane, consensus)

    # consider an average between best plane for this image pair and its predecessor
    if (previous_plane != []):
        # create the average plane from the midpoints of their X,Y,Z components
        average_plane = []
        for p,q in zip(best_plane, previous_plane):
            midpoint =  [(min(a,b) + (max(a,b) - min(a,b))/2)  for a,b in zip(p,q)]
            average_plane.append(midpoint)
        # use the average plane if it has a better consensus
        (normal, plane, consensus) = fit_plane(points, average_plane)
        if (len(consensus) > CONSENSUS_THRESHOLD and len(consensus) > len(best_consensus)):
            (best_normal, best_plane, best_consensus) = (normal, plane, consensus)

    return best_normal, best_plane, best_consensus


def fit_plane(points, plane=[]):
    """ 3D plane fitting taken from DUO
        Modified to reject vertical planes and return consensus points

    Arguments:
        points -- list of 3D points
        plane  -- 3D plane to be tested
    """
    # if plane is not supplied, randomly select one from points
    if (plane == []):
        cross_product_check = np.array([0,0,0])
        while np.all(cross_product_check == 0):
            (P1,P2,P3) = points[random.sample(range(len(points)), 3)]
            # make sure they are non-collinear
            cross_product_check = np.cross(P1-P2, P2-P3);
    else:
        (P1, P2, P3) = plane

    # how to - calculate plane coefficients from these points
    try:
        coefficients_abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))
    except np.linalg.linalg.LinAlgError:
        return ([], [], [])

    # check plane is roughly horizontal
    # used to prevent fitting of sides of cars, hedges etc
    if (coefficients_abc[1] < 0.5):
        return ([],[],[])

    coefficient_d = np.linalg.norm(coefficients_abc)

    # extract list of points that are within MAX_INLIER_DISTANCE of the plane
    dist = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d)
    (row, _ )= np.where(dist < MAX_INLIER_DISTANCE)
    inliers = points[row]

    normal = coefficients_abc / coefficient_d
    normal = normal.flatten()
    return (normal, [P1,P2,P3], inliers)

############################################

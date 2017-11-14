import planar_fitting
import random

CONSENSUS_THRESHOLD = 20000
MAX_INLIER_DISTANCE = 0.1

def ransac(points, iterations):
    # sample = points[random.sample(range(len(points)), 1000000)]
    sample = points
    (best_plane, best_consensus, best_dist) = ([], [], 999)
    for i in range(iterations):
        (plane, consensus, dist) = planar_fitting.fit_plane(sample, MAX_INLIER_DISTANCE)
        if (len(consensus) > CONSENSUS_THRESHOLD and dist > best_dist):
            (best_plane, best_consensus, best_dist) = (plane, consensus, dist)
    print(best_dist)
    print(best_plane)
    # print(best_consensus)
    return best_plane, best_consensus

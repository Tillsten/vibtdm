import numpy as np

def norm(vects : np.ndarray):
    """
    Normalize vectors.

    Parameters
    ----------
    vects : ndarray
        Array with the shape (3,N), where N is the number
        of vectors.

    Return
    ------
    ndarray
        Array of the normalized vectors
    """
    return np.linalg.norm(vects, axis=0)

def project_on_plane(points : np.ndarray):
    """
    Project an array of points on a single plane.

    Parameters
    ----------
    points : ndarray
        Array of points

    Returns
    -------

    """
    points_mean = np.mean(points, 1)
    shifted_points = points - points_mean[:, None]
    u, s, v = np.linalg.svd(points)
    print(u.shape, v.shape)
    return v[0], v[1], u[2]

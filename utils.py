import numpy as np


def interpolate_contour_2d(array, n)->np.ndarray:
    """
    Interpolate a 2d-contour array with n points.
    Works with non-equal contour point distances too.
    The created array has equal distances inbetween points.
    :param array: the 2d contour array.
    :param n: number of points in the end contour.
    :return: new 2d contour array.
    """
    array = np.atleast_2d(array)
    #np.diff calculates v_i+1 - v_i
    #np.linalg.norm calculates length of each v
    #IMPORTANT -> np.diff creates n-1 points
    len_intra_points = np.linalg.norm(np.diff(array, axis=0), axis=1)

    #len_intra_points create an array of length n-1
    #we need an array of length n
    contour_s = np.zeros(len(array))
    contour_s[1:] = np.cumsum(len_intra_points)

    total_length = contour_s[-1]

    new_contour_s = np.linspace(0, total_length, n)

    #now we interpolate x and y points separately
    x_new = np.interp(new_contour_s, contour_s, array[:,0])
    y_new = np.interp(new_contour_s, contour_s, array[:,1])

    new_array = np.stack([x_new, y_new], axis=1)
    return new_array


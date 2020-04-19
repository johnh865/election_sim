# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy.stats import truncnorm
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate.ndgriddata import _ndim_coords_from_arrays



def ltruncnorm(loc, scale, size, random_state=None):
    """
    Truncated normal random numbers, cut off at locations less than 0.
    
    Parameters
    -----------
    loc : float 
        Center coordinate of gaussian distribution
    scale : float 
        Std deviation scale
    size : int
        Number of random numbers to generate
    random_state : None or numpy.random.RandomState 
        Random number seeding object, or None.
        
    Returns
    ---------
    out : array shaped (size)
        Output samples
    """
    if scale == 0:
        return np.ones(size) * loc
    
    xmin = -loc / scale
    t = truncnorm(xmin, 1e6)
    s = t.rvs(size=size, random_state=random_state)
    s = s * scale  + loc
    return s



class NearestManhattanInterpolator(NearestNDInterpolator):
    """
    NearestManhattanInterpolator(x, y)

    Nearest-neighbour interpolation in N dimensions using Manhatten 
    p=1 norm. 

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    x : (Npoints, Ndims) ndarray of floats
        Data point coordinates.
    y : (Npoints,) ndarray of float or complex
        Data values.
    """    

    def __call__(self, *args):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        """
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)
        dist, i = self.tree.query(xi, p=1)
        return self.values[i]


def vectorized_choice(p, n, items=None):
    """
    Randomly choose integers
    
    Parameters
    ------------
    p : array
        Weights on choices
    n : int
        Number of choices
    items : array
        Choices
        
    """
    s = p.cumsum(axis=1)
    r = np.random.rand(p.shape[0], n, 1)
    
    q = np.expand_dims(s, 1) >= r
    k = q.argmax(axis=-1)
    if items is not None:
        k = np.asarray(items)[k]
    return k




p = np.ones((20, 5))
#p[:, 1] = 0

p = p / np.sum(p, axis=1)[:, None]
n = 4
out = vectorized_choice(p, n)
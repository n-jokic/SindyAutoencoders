import numpy as np
from scipy.special import binom
from scipy.integrate import odeint
import itertools as itertools


def sindy_library(X, poly_order, include_sine=False):
    m, n = X.shape
    l = library_size(n, poly_order, include_sine, True)
    library = np.ones((m, l))
    index = 1

    library[:, index:index+n] = X  # Linear terms
    index += n

    for order in range(2, poly_order + 1):
        for indices in itertools.product(range(n), repeat=order):
            library[:, index] = np.prod(X[:, indices], axis=1)
            index += 1

    if include_sine:
        library[:, index:] = np.sin(X)
    
    return library

def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 1 if include_constant else 0
    
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))

    if use_sine:
        l += n
    return l


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i])[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x

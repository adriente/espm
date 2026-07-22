import numpy as np

from espm.conf import dicotomy_tol, log_shift, maxit_dichotomy


def dichotomy_simplex(
    num, denum, log_shift=log_shift, tol=dicotomy_tol, maxit=maxit_dichotomy, safe=False
):
    """
    Function to solve the num/(x+denum) -1 = 0 equation. Here, x is the Lagragian multiplier which is used to apply the simplex constraint.
    The first part consists in finding a and b such that num/(a+denum) -1 > 0 and num/(b+denum) -1  < 0.
    The second part applies the dichotomy algorithm to solve the equation.
    """
    # The function has exactly one root at the right of the first singularity (the singularity at min(denum))

    # num = num.astype("float64")
    # denum = denum.astype("float64")

    # do some test

    if safe:
        assert (num >= 0).all()
        assert (denum >= 0).all()
        assert (np.sum(num, axis=0) > 0).all()
    if log_shift > 0:
        # Check that a solution is possible
        if denum.shape[0] * log_shift >= 1:
            raise ValueError("No solution exists!")
        denum_max = num / log_shift
    else:
        denum_max = np.inf
    # Ideally we want to do this, but we have to exclude the case where num==0.
    # a = np.max(num/2 - denum, axis=0)
    val = num / 2 - denum
    val[num <= 0] = -np.inf
    a = np.max(val, axis=0)

    # r = np.sum(num/denum, axis=0)
    # b = np.zeros(r.shape)
    # b[r>=1] = (len(num) * np.max(num, axis=0)/0.5 - np.min(denum, axis=0))[r>=1]
    b = len(num) * np.max(num, axis=0) / 0.5 - np.min(denum, axis=0)

    def func(x):
        new_x = x + denum
        return np.sum(np.maximum(num / new_x, log_shift), axis=0) - 1

    return dicotomy(a, b, func, maxit, tol, safe=safe)


def dichotomy_simplex_acc(
    a, b, minus_c, log_shift=log_shift, tol=dicotomy_tol, maxit=maxit_dichotomy, safe=False
):
    """
    Function to solve the dicotomy for the function:
    f(nu) = n_p * nu_k + 2a - sum_p sqrt ( (b_p + nu)^2 - 4 a c_p) + sum_p b_p

    The first part consists in finding nu_max and nu_min such that f(nu_max) > 0 and f(nu_min) < 0.
    The second part applies the dichotomy algorithm to solve the equation.
    """
    # do some test
    if safe:
        assert a >= 0
        assert (minus_c >= 0).all()

    if log_shift > 0:
        # Check that a solution is possible
        if b.shape[0] * log_shift >= 1:
            raise ValueError("No solution exists!")

    n_p = len(b)
    nu_max = n_p * np.max(b**2 / a + 2 * a + 2 * (b + minus_c), axis=0) * 1.5 + 1e-3
    nu_min = -(2 * a + np.sum(b, axis=0)) / n_p * 1.1 - 1e-3

    def func(x):
        return 2 * a - np.sum(
            np.maximum(
                np.sqrt((b + x) ** 2 + 4 * a * minus_c) - x - b, log_shift * 2 * a
            ),
            axis=0,
        )

    return dicotomy(nu_max, nu_min, func, maxit, tol, safe=safe)


def dichotomy_simplex_projected_gradient(
    a, log_shift=log_shift, tol=dicotomy_tol, maxit=maxit_dichotomy, safe=False
):
    r"""
    Function to solve the dicotomy for the function:

    .. :math::

        f(\nu) = \sum_p \max \left( \alpha_p+\nu, \epsilon right) - 1 = 0

    The first part consists in finding nu_max and nu_min such that f(nu_max) > 0 and f(nu_min) < 0.
    The second part applies the dichotomy algorithm to solve the equation.
    """

    if log_shift > 0:
        # Check that a solution is possible
        if a.shape[0] * log_shift >= 1:
            raise ValueError("No solution exists!")

    nu_min = -np.max(a, axis=0)
    nu_max = 1 / a.shape[0] - np.min(a, axis=0)

    def func(x):
        return np.sum(np.maximum(a + x, log_shift), axis=0) - 1

    return dicotomy(nu_max, nu_min, func, maxit, tol, safe=safe)


def dicotomy(a, b, func, maxit, tol, safe=False):
    """
    Dicotomy algorithm searching for func(x)=0.

    Parameters
    ----------

    a : float or numpy array
        Lower bound of the interval such that func(a) > 0
    b : float or numpy array
        Upper bound of the interval such that func(b) < 0
    func : function
        Function to solve
    maxit : int
        Maximum number of iterations - the algorithm stops if |func(sol)| < tol
    tol : float
        Tolerance - the algorithm stops if |func(sol)| < tol


    Returns
    -------
    new : float or numpy array
        Solution of the equation func(new) = 0

    This algorithm works for number or numpy array of any size.
    """
    # Copy arrays to prevent mutating arguments in-place
    if isinstance(a, np.ndarray):
        a = a.copy()
    if isinstance(b, np.ndarray):
        b = b.copy()

    func_max = func(a)
    func_min = func(b)

    if safe:
        assert np.sum(func_min >= 0) == 0
        assert np.sum(func_max <= 0) == 0
        assert np.sum(np.isnan(func_max)) == 0
        assert np.sum(np.isnan(func_min)) == 0

    # Dichotomy algorithm to solve the equation
    it = 0
    new = (a + b) / 2
    func_new = func(new)
    
    if isinstance(func_max, np.ndarray):
        func_a = func_max.copy()
    else:
        func_a = func_max

    while np.max(np.abs(func_new)) > tol:
        it = it + 1
        
        # if f(a)*f(new) <0 then f(new) < 0 --> store in b
        minus_bool = func_a * func_new <= 0

        # if f(a)*f(new) > 0 then f(new) > 0 --> store in a
        plus_bool = np.logical_not(minus_bool)

        if isinstance(new, np.ndarray):
            b[minus_bool] = new[minus_bool]
            a[plus_bool] = new[plus_bool]
            if isinstance(func_a, np.ndarray):
                func_a[plus_bool] = func_new[plus_bool]
            else:
                func_a = func_new if plus_bool else func_a
        else:
            if minus_bool:
                b = new
            else:
                a = new
                func_a = func_new

        new = (a + b) / 2
        func_new = func(new)
        if it >= maxit:
            print(
                "Dicotomy stopped for maximum number of iterations with an error of : {}".format(
                    np.max(np.abs(func_new))
                )
            )
            break

    return new

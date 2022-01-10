# -*- coding: utf-8 -*-
import sys
import scipy as sp
import scipy.linalg as splin
from numpy import linalg as LA

def sunsal(M,y,AL_iters=1000,lambda_0=0.,positivity=False,addone=False,tol=1e-4,x0 = None,verbose=False):


    """
     SUNSAL -> sparse unmixing via variable splitting and augmented
     Lagrangian methods
    --------------- Description --------------------------------------------
     SUNSAL solves the following l2-l1 optimization  problem
     [size(M) = (L,p); size(X) = (p,N)]; size(Y) = (L,N)]
            min  (1/2) ||M X-y||^2_F + lambda ||X||_1
             X
     where ||X||_1 = sum(sum(abs(X)).
       CONSTRAINTS ACCEPTED:
       1) POSITIVITY:  X >= 0;
       2) ADDONE:  sum(X) = ones(1,N);
       NOTES:
          1) The optimization w.r.t each column of X is decoupled. Thus,
             SUNSAL solves N simultaneous problems.
          2) SUNSAL solves the following  problems:
             a) BPDN - Basis pursuit denoising l2-l1
                       (lambda > 0, POSITIVITY = 'no', ADDONE, 'no')
             b) CBPDN - Constrained basis pursuit denoising l2-l1
                       (lambda > 0, POSITIVITY = 'yes', ADDONE, 'no')
             c) CLS   - Constrained least squares
                        (lambda = 0, POSITIVITY = 'yes', ADDONE, 'no')
             c) FCLS   - Fully constrained least squares
                        (lambda >=0 , POSITIVITY = 'yes', ADDONE, 'yes')
                         In this case, the regularizer ||X||_1  plays no role,
                         as it is constant.
    -------------------- Line of Attack  -----------------------------------
     SUNSAL solves the above optimization problem by introducing a variable
     splitting and then solving the resulting constrained optimization with
     the augmented Lagrangian method of multipliers (ADMM).
            min  (1/2) ||M X-y||^2_F + lambda ||Z||_1
             X,Z
            subject to: sum(X) = ones(1,N)); Z >= 0; X = Z
     Augmented Lagrangian (scaled version):
          L(X,Z,D) = (1/2) ||M X-y||^2_F + lambda ||Z||_1 + mu/2||X-Z-D||^2_F
     where D are the scale Lagrange multipliers
     ADMM:
         do
           X  <-- arg min L(X,Z,D)
                       X, s.t: sum(X) = ones(1,N));
           Z  <-- arg min L(X,Z,D)
                       Z, s.t: Z >= 0;
           D  <-- D - (X-Z);
         while ~stop_rulde
    For details see
    [1] J. Bioucas-Dias and M. Figueiredo, "Alternating direction algorithms
    for constrained sparse regression: Application to hyperspectral unmixing",
    in 2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal
    Processing-WHISPERS'2010, Raykjavik, Iceland, 2010.
    ------------------------------------------------------------------------
    ====================== Required inputs =============
     M - [L(channels) x p(endmembers)] mixing matrix
     y - matrix with  L(channels) x N(pixels).
         each pixel is a linear mixture of p endmembers
         signatures y = M*x + noise,
    ====================== Optional inputs =============================
     AL_ITERS - Minimum number of augmented Lagrangian iterations - Default: 1000
     lambda_0 - regularization parameter. lambda is either a scalar
              or a vector with N components (one per column of x)
              Default: 0.
     positivity  = {True, False}; Enforces the positivity constraint: X >= 0 - Default: False
     addone  = {True, False}; Enforces the positivity constraint: X >= 0 - Default: False
     tol    - tolerance for the primal and  dual residuals - Default: 1e-4;
     verbose   = {True, False}; Default: False
    =========================== Outputs ==================================
    x      estimated mixing matrix [pxN]
    res_p  primal residual
    res_d  dual residual
    i      number of iteration until convergence
    ------------------------------------------------------------------
    Author: Jose Bioucas-Dias, 2009
    -------------------------------------------------------------------------
    Copyright (July, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
    SUNSAL is distributed under the terms of the GNU General Public License 2.0.
    Permission to use, copy, modify, and distribute this software for any purpose without fee is hereby granted, provided that this entire notice is included in all copies of any software which is or includes a copy or modification of this software and in all copies of the supporting documentation for such software.
    This software is being provided "as is", without any express or implied warranty.  In particular, the authors do not make any representation or warranty of any kind concerning the merchantability of this software or its fitness for any particular purpose."
    ---------------------------------------------------------------------
    Software translated from matlab to python by Adrien Lagrange (ad.lagrange@gmail.com), 2018.
    """

    #--------------------------------------------------------------
    # test for number of required parametres
    #--------------------------------------------------------------
    [LM,p] = M.shape # mixing matrixsize
    [L,N] = y.shape # data set size
    if LM != L:
        sys.exit('mixing matrix M and data set y are inconsistent')

    ##
    #--------------------------------------------------------------
    # Local variables
    #--------------------------------------------------------------

    #--------------------------------------------------------------
    # Read the optional parameters
    #--------------------------------------------------------------
    AL_iters = int(AL_iters)
    if (AL_iters < 0 ):
        sys.exit('AL_iters must a positive integer')

    # If lambda is scalar convert it into vector
    lambda_0 = ( lambda_0 * sp.ones((N,p)) ).T
    if (lambda_0<0).any():
        sys.exit('lambda_0 must be positive')

    # compute mean norm
    norm_m = splin.norm(M)*(25+p)/float(p)
    # rescale M and Y and lambda
    M = M/norm_m
    y = y/norm_m
    lambda_0 = lambda_0/norm_m**2

    if x0 is not None:
        if (x0.shape[0]==p) or (x0.shape[0]==N):
            sys.exit('initial X is not inconsistent with M or Y')


    #---------------------------------------------
    # just least squares
    #---------------------------------------------
    if (lambda_0.sum() == 0) and (not positivity) and (not addone):
        z = sp.dot(splin.pinv(M),y)
        # primal and dual residues
        res_p = 0.
        res_d = 0.
        return z,res_p,res_d,None

    #---------------------------------------------
    # least squares constrained (sum(x) = 1)
    #---------------------------------------------
    SMALL = 1e-12;
    if (lambda_0.sum() == 0) and (addone) and (not positivity):
        F = sp.dot(M.T,M)
        # test if F is invertible
        if LA.cond(F) > SMALL:
            # compute the solution explicitly
            IF = splin.inv(F);
            z = sp.dot(sp.dot(IF,M.T),y) - (1./IF.sum())*sp.dot(sp.sum(IF,axis=1,keepdims=True) , ( sp.dot(sp.dot(sp.sum(IF,axis=0,keepdims=True),M.T),y) - 1.))
            # primal and dual residues
            res_p = 0
            res_d = 0

            return z,res_p,res_d,None
        else:
            sys.exit('Bad conditioning of M.T*M')


    #---------------------------------------------
    #  Constants and initializations
    #---------------------------------------------
    mu_AL = 0.01
    mu = 10*lambda_0.mean() + mu_AL

    [UF,SF] = splin.svd(sp.dot(M.T,M))[:2]
    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
    x_aux = sp.sum(Aux,axis=1,keepdims=True)
    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))


    yy = sp.dot(M.T,y)

    #---------------------------------------------
    #  Initializations
    #---------------------------------------------

    # no intial solution supplied
    if x0 is None:
       x = sp.dot( sp.dot(IF,M.T) , y)
    else:
        x = x0

    z = x
    # scaled Lagrange Multipliers
    d  = 0*z

    #---------------------------------------------
    #  AL iterations - main body
    #---------------------------------------------
    tol1 = sp.sqrt(N*p)*tol
    tol2 = sp.sqrt(N*p)*tol
    i=1
    res_p = sp.inf
    res_d = sp.inf
    maskz = sp.ones(z.shape)
    mu_changed = 0

    #--------------------------------------------------------------------------
    # constrained  leat squares (CLS) X >= 0
    #--------------------------------------------------------------------------
    if (lambda_0.sum() ==  0)  and (not addone):
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF,yy + mu*(z+d))
            # Lagrange multipliers update
            d -= (x-z)
            
            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    # Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    # x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    # IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

        #--------------------------------------------------------------------------
        # Fully constrained  leat squares (FCLS) X >= 0
        #--------------------------------------------------------------------------
    elif (lambda_0.sum() ==  0)  and addone:
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF1,yy + mu*(z+d)) + x_aux
            # Lagrange multipliers update
            d -= (x-z)
            
            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

        #--------------------------------------------------------------------------
        # generic SUNSAL: lambda > 0
        #--------------------------------------------------------------------------
    else:
        # implement soft_th
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            nu = x-d
            z = sp.sign(nu) * sp.maximum(sp.absolute(nu) - lambda_0/mu,0)
            # teste for positivity
            if positivity:
                z = sp.maximum(z,0)
            # teste for sum-to-one
            if addone:
                x = sp.dot(IF1,yy+mu*(z+d)) + x_aux
            else:
                x = sp.dot(IF,yy+mu*(z+d))
            # Lagrange multipliers update
            d -= (x-z)
            
            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

    return x,res_p,res_d,i
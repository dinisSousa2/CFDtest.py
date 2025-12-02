import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import math
import json
import sys
import argparse
import os
import glob
import time
from scipy.sparse import diags, kron, eye, lil_matrix, linalg, block_diag
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg 
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import spilu, LinearOperator
import pyamg
import sys


def cg_solver(p, S, App, nx, ny):
    
    # Flatten the source term (excluding ghost cells)
    b = S[1:-1, 1:-1].ravel()
    
    # Initial guess (flattened, excluding ghost cells)
    p0 = p[1:-1, 1:-1].ravel()
    
    # Solve using PCG with diagonal preconditioner
    M = diags(1.0/App.diagonal())  # Jacobi preconditioner
    
    p_flat, info = cg(App, b, x0=p0, rtol=1e-06, M=M)
    
    if info != 0:
        print(f"PCG did not converge: info = {info}")
    # Compute residual norm
    residual = np.abs(App @ p_flat - b).reshape((ny,nx))
    #print(f'DimentionsResi_p = {np.shape(residual)}')
    
    # Reshape solution and maintain ghost cells
    p_new = p.copy()
    p_new[1:-1, 1:-1] = p_flat.reshape((ny, nx))
    
    return p_new, residual

def amg_solver(p, S, Ap, nx, ny, ml):
    """
    Solves the linear system Ap x = b using PyAMG as a solver.
    """
    # Flatten source term (excluding ghost cells)
    b = S[1:-1, 1:-1].ravel()

    # Initial guess
    x0 = p[1:-1, 1:-1].ravel()

    # Setup PyAMG multigrid solver
    #ml = pyamg.smoothed_aggregation_solver(Ap)

    # Solve the system using AMG
    x = ml.solve(b, x0=x0, tol=1e-6, cycle ='V')


    # Compute residual norm
    #residual = np.linalg.norm(Ap @ x - b)
    residual = np.abs(Ap @ x - b).reshape((ny,nx))

    # Insert result back into pressure field
    p_new = p.copy()
    p_new[1:-1, 1:-1] = x.reshape((ny, nx))

    return p_new, residual
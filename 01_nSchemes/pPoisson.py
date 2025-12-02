import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../00_AUX'))

from linearSolvers import cg_solver
from linearSolvers import amg_solver

sys.path.append(os.path.join(os.path.dirname(__file__), '../02_SIM'))
from mainSimulation import apply_BC

def pressure_correction(u, v, Ap, p, dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName):
    div = np.zeros_like(p)
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1])/dx + (v[2:, 1:-1] - v[1:-1, 1:-1])/dy
    
    prhs = div / dt
    
    # Use PCG instead of SOR     p, S, Ap, nx, ny
    if pressureProjection == 'AMG':
        p_new, residual = amg_solver(p.copy(), prhs, Ap, nx, ny, ml)
    elif pressureProjection == 'CG':
        p_new, residual = cg_solver(p.copy(), prhs, Ap, nx, ny)

    # Apply pressure correction to velocities
    u_corr = u.copy()
    v_corr = v.copy()

    dpdx = (p_new[1:-1, 2:-1] - p_new[1:-1, 1:-2])/dx
    dpdy = (p_new[2:-1, 1:-1] - p_new[1:-2, 1:-1])/dy

    u_corr[1:-1, 2:-1] = u[1:-1, 2:-1] - dt * dpdx
    v_corr[2:-1, 1:-1] = v[2:-1, 1:-1] - dt * dpdy
    
    u_corr, v_corr = apply_BC(u_corr, v_corr, Utop, caseName)

    return u_corr, v_corr, p_new, dpdx, dpdy, residual
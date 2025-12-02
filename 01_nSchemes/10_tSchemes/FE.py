import sys
import os
# Go up two levels from current file, then into 02_SIM
sys.path.append(os.path.join(os.path.dirname(__file__), '../../02_SIM'))
from mainSimulation import apply_BC

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from navierStokes import NS_operator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pPoisson import pressure_correction



def run_FE(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, pressureProjection, caseName):
    # Stage 1: Compute explicit RHS (advection + diffusion only)
    k1u, k1v = NS_operator(u, v, nu, dx, dy, nx, ny)
    u1 = u + dt * k1u
    v1 = v + dt * k1v
    u1, v1 = apply_BC(u1, v1, Utop, caseName)
    ut, vt, p, _, _, _ = pressure_correction(u1, v1, App, p, dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

    u, v = ut.copy(), vt.copy()
    u, v = apply_BC(u, v, Utop, caseName)

    return u, v
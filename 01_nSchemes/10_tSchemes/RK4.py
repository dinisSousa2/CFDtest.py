import sys
import os
# Go up two levels from current file, then into 02_SIM
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02_SIM'))
from mainSimulation import apply_BC

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from navierStokes import NS_operator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pPoisson import pressure_correction

def run_RK4(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, pressureProjection, caseName):
    # Stage 1: Compute explicit RHS (advection + diffusion only)
    k1u, k1v = NS_operator(u, v, nu, dx, dy, nx, ny)
    u1 = u + 0.5*dt * k1u
    v1 = v + 0.5*dt * k1v
    u1, v1 = apply_BC(u1, v1, Utop, caseName)
    #u1, v1, _, _, _, _ = pressure_correction(u1, v1, App, p, 0.5*dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

    # Stage 2: Repeat with u1, v1
    k2u, k2v = NS_operator(u1, v1, nu, dx, dy, nx, ny)
    u2 = u + 0.5*dt * k2u
    v2 = v + 0.5*dt * k2v
    u2, v2 = apply_BC(u2, v2, Utop, caseName)
    #u2, v2, _, _, _, _ = pressure_correction(u2, v2, App, p, 0.5*dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

    # Stage 3: Full step
    k3u, k3v = NS_operator(u2, v2, nu, dx, dy, nx, ny)
    u3 = u + dt * k3u
    v3 = v + dt * k3v
    u3, v3 = apply_BC(u3, v3, Utop, caseName)
    #u3, v3, _, _, _, _ = pressure_correction(u3, v3, App, p, dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

    # Stage 4: Final explicit RHS
    k4u, k4v = NS_operator(u3, v3, nu, dx, dy, nx, ny)

    # RK4 weighted update (without pressure yet)
    ut = u + (dt/6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    vt = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    ut, vt = apply_BC(ut, vt, Utop, caseName)

    # FINAL pressure correction
    ut, vt, p, _, _, _ = pressure_correction(ut, vt, App, p, dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

    # Update variables
    u, v = ut.copy(), vt.copy()
    u, v = apply_BC(u, v, Utop, caseName)

    return u, v
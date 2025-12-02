import numpy as np
import math
from scipy.sparse import eye, diags
from scipy.sparse.linalg import cg
import sys
import os
# Go up two levels from current file, then into 02_SIM
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02_SIM'))
from mainSimulation import apply_BC

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from navierStokes import compute_explicit, compute_implicit

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pPoisson import pressure_correction

def run_IMEX(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, c, A, A_hat, b, b_hat, Iu, Iv, Lu, Lv, Su, pressureProjection, caseName):

    ui_int = u[1:-1, 2:-1]
    vi_int = v[2:-1, 1:-1]

    s = A.shape[0]

    K_hat = []
    U = []
    K = []
    
    p0 = p.copy()
    
    #Variable Initialization
    u0_int = u.copy()
    v0_int = v.copy()
    Hu0 = np.zeros_like(u0_int)
    Hv0 = np.zeros_like(v0_int)
    Ku0_hat_int = np.zeros_like(u0_int)
    Kv0_hat_int = np.zeros_like(v0_int)

    Uut0 = np.zeros_like(u)
    Uvt0 = np.zeros_like(v)
    
    p0 = p.copy()
    Ku0_hat = np.zeros_like(u) #[ny+2, nx+2]
    Kv0_hat = np.zeros_like(v) #[ny+2, nx+2]

    u0_int = u[1:-1, 2:-1]
    v0_int = v[2:-1, 1:-1]

    Ku0_hat, Kv0_hat = compute_explicit(u, v, dx, dy, nx, ny)

    Ku0_hat_int = Ku0_hat[1:-1, 2:-1] #[ny, nx-1]
    Kv0_hat_int = Kv0_hat[2:-1, 1:-1] #[ny-1, nx]

    Hu0 = u0_int + dt * A_hat[1,0] * Ku0_hat_int #Only the interior points of u and Ku_hat [ny, nx-1]
    Hv0 = v0_int + dt * A_hat[1,0] * Kv0_hat_int #Only the interior points of v and  Kv_hat [ny-1, nx]

    Mu = Iu - nu * dt * A[0,0] * Lu #[ny * (nx-1), ny * (nx-1)]
    Mv = Iv - nu * dt * A[0,0] * Lv #[(ny-1) * nx, (ny-1) * nx]

    b0_int = Hu0.ravel() + nu * dt * A[0,0] * Su

    Mpu = diags(1.0/Mu.diagonal())  # Jacobi preconditioner
    Uu0_vec, info = cg(Mu, b0_int, x0=u0_int.ravel(), rtol=1e-06, M=Mpu)

    Mpv = diags(1.0/Mv.diagonal())  # Jacobi preconditioner
    Uv0_vec, info = cg(Mv, Hv0.ravel(), x0=v0_int.ravel(), rtol=1e-06, M=Mpv)

    Uut0_int = Uu0_vec.reshape(u0_int.shape) #[ny, nx-1]
    Uvt0_int = Uv0_vec.reshape(v0_int.shape) #[ny-1, nx]

    Uut0[1:-1, 2:-1] = Uut0_int
    Uvt0[2:-1, 1:-1] = Uvt0_int
    
    Uut0, Uvt0 = apply_BC(Uut0, Uvt0, Utop, caseName)

    #dt0 = (A[0,0]) * dt

    Uu0, Uv0, _, _, _, _ = pressure_correction(Uut0, Uvt0, App, p0, dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)
    #print(f'dpdx0len = {np.shape(dpdx0)}')
    Uu0, Uv0 = apply_BC(Uu0, Uv0, Utop, caseName)

    Ku0, Kv0 = compute_implicit(Uu0, Uv0, nu, dx, dy, nx, ny)

    K_hat.append((Ku0_hat, Kv0_hat))
    U.append((Uu0, Uv0))
    K.append((Ku0, Kv0))

    for i in range(1, s):
        pi = p.copy()
        Mui = Iu - nu * dt * A[i,i] * Lu
        Mvi = Iv - nu * dt * A[i,i] * Lv

        Kui_hat, Kvi_hat = compute_explicit(U[i-1][0], U[i-1][1], dx, dy, nx, ny)
        K_hat.append((Kui_hat, Kvi_hat))

        Zui = u + dt * A_hat[i+1, i] * K_hat[i][0]
        Zvi = v + dt * A_hat[i+1, i] * K_hat[i][1]
        
        for j in range(i):
            Zui += dt * (A[i,j] * K[j][0] + A_hat[i+1,j] * K_hat[j][0])
            Zvi += dt * (A[i,j] * K[j][1] + A_hat[i+1,j] * K_hat[j][1])

        Zui_int = np.zeros([ny, nx-1])
        Zvi_int = np.zeros([ny-1, nx])
        Zui_int = Zui[1:-1, 2:-1]
        Zvi_int = Zvi[2:-1, 1:-1]

        bi_int = Zui_int.ravel() + nu * dt * A[i,i] * Su

        # Use CG for both systems (consistent with first step)
        Mpui = diags(1.0/Mui.diagonal())  # Jacobi preconditioner
        Uui_vec, info_u = cg(Mui, bi_int, x0=ui_int.ravel(), rtol=1e-06, M=Mpui)

        Mpvi = diags(1.0/Mvi.diagonal())  # Jacobi preconditioner  
        Uvi_vec, info_v = cg(Mvi, Zvi_int.ravel(), x0=vi_int.ravel(), rtol=1e-06, M=Mpvi)

        Uuit_int = Uui_vec.reshape(Zui_int.shape) #[ny, nx-1]
        Uvit_int = Uvi_vec.reshape(Zvi_int.shape) #[ny-1, nx]

        Uuit = np.zeros_like(u)
        Uvit = np.zeros_like(v)
        Uuit[1:-1, 2:-1] = Uuit_int
        Uvit[2:-1, 1:-1] = Uvit_int
        Uuit, Uvit = apply_BC(Uuit, Uvit, Utop, caseName)
        
        #dti = (A[i,i]) * dt

        Uui, Uvi, _, dpdxi, dpdyi, _ = pressure_correction(Uuit, Uvit, App, pi, dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

        Uui, Uvi = apply_BC(Uui, Uvi, Utop, caseName)
        U.append((Uui, Uvi))

        Kui, Kvi = compute_implicit(Uui, Uvi, nu, dx, dy, nx, ny)
    
        #Kui[1:-1, 2:-1] = Kui[1:-1, 2:-1] - dpdxi
        #Kvi[2:-1, 1:-1] = Kvi[2:-1, 1:-1] - dpdyi

        K.append((Kui, Kvi))

    #Compute the last K_hat
    Kus_hat, Kvs_hat = compute_explicit(U[-1][0], U[-1][1], dx, dy, nx, ny)
    K_hat.append((Kus_hat, Kvs_hat))

    uhast = u + dt * b_hat[-1] * K_hat[-1][0] 
    vhast = v + dt * b_hat[-1] * K_hat[-1][1]

    for i in range(s):
        uhast += dt * (b[i] * K[i][0] + b_hat[i] * K_hat[i][0])
        vhast += dt * (b[i] * K[i][1] + b_hat[i] * K_hat[i][1])

    # uhast_int = np.zeros([ny, nx-1])
    # vhast_int = np.zeros([ny-1, nx])
    # uhast_int = uhast[1:-1, 2:-1]
    # vhast_int = vhast[2:-1, 1:-1]

    u, v = apply_BC(u, v, Utop, caseName)
    u, v, p, _, _, _ = pressure_correction(uhast, vhast, App, p.copy(), dt, dx, dy, nx, ny, Utop, ml, pressureProjection, caseName)

    u, v = apply_BC(u, v, Utop, caseName)

    return u, v
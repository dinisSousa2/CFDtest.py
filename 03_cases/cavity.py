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


sys.path.append(os.path.join(os.path.dirname(__file__), '../02_SIM'))
from mainSimulation import do_time_integration, apply_BC

#===Cavity Case==================================================================
def gen_Ap_cavity(nx, ny, dx, dy):
    total_size = ny * nx
    
    # =========================
    # Generate Apx (x-direction)
    # =========================
    inv_dx2 = 1.0 / (dx ** 2)
    
    # Main diagonal
    md_Apx = np.full(total_size, -2.0 * inv_dx2)
    md_Apx[::nx] = -1.0 * inv_dx2       # First element of each x-block
    md_Apx[nx-1::nx] = -1.0 * inv_dx2   # Last element of each x-block
    
    # Upper/lower diagonals
    upper_x = np.ones(total_size - 1) * inv_dx2
    upper_x[nx-1::nx] = 0.0
    lower_x = upper_x.copy()
    
    Apx = diags([lower_x, md_Apx, upper_x], [-1, 0, 1], format='csr')
    
    # =========================
    # Generate Apy (y-direction)
    # =========================
    inv_dy2 = 1.0 / (dy ** 2)
    
    # Main diagonal
    md_Apy = np.zeros(total_size)
    
    # Fill interior rows (-2/dy²)
    md_Apy[nx:-nx] = -2.0 * inv_dy2
    
    # Fill first and last ny rows (-1/dy²)
    md_Apy[:nx] = -1.0 * inv_dy2      # First y-block
    md_Apy[-nx:] = -1.0 * inv_dy2     # Last y-block
    
    # Off-diagonals (+1/dy² spaced nx apart)
    upper_y = np.ones(total_size - nx) * inv_dy2
    lower_y = upper_y.copy()
    
    Apy = diags([lower_y, md_Apy, upper_y], [-nx, 0, nx], format='csr')
    
    # =========================
    # Combine matrices
    # =========================
    Ap = Apx + Apy
    #ml = pyamg.smoothed_aggregation_solver(Ap)
    ml = pyamg.ruge_stuben_solver(Ap)
    
    return Ap, ml

def gen_Lu_cavity(nx, ny, dx, dy, Utop):
    """
    Generate Laplacian operators for a staggered grid of size ny × (nx-1).

    Parameters:
        nx (int): Number of x grid points (including boundaries)
        ny (int): Number of y grid points (including boundaries)
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction

    Returns:
        Lu (csr_matrix): Combined Laplacian operator (1/dx² Lux + 1/dy² Luy)
        Lux (csr_matrix): x-direction component (unscaled)
        Luy (csr_matrix): y-direction component (unscaled)
    """
    nux = nx - 1
    N = ny * nux  # total number of u points

    # --- x-direction Laplacian block (1D) ---
    main_diag_x = -2 * np.ones(nux)
    off_diag_x = np.ones(nux - 1)
    Lux1D = diags([off_diag_x, main_diag_x, off_diag_x], offsets=[-1, 0, 1], format='csr')

    # 2D x-direction Laplacian using Kronecker product
    Iy = identity(ny, format='csr')
    Lux = kron(Iy, Lux1D, format='csr')

    # --- y-direction Laplacian block (1D) with boundary handling ---
    main_diag_y = -2 * np.ones(ny)
    main_diag_y[0] = -3
    main_diag_y[-1] = -3
    off_diag_y = np.ones(ny - 1)
    Luy1D = diags([off_diag_y, main_diag_y, off_diag_y], offsets=[-1, 0, 1], format='csr')

    # 2D y-direction Laplacian using Kronecker product
    Ix = identity(nux, format='csr')
    Luy = kron(Luy1D, Ix, format='csr')

    # --- Combine the Laplacians ---
    Lu = (1 / dx**2) * Lux + (1 / dy**2) * Luy

    Su = np.zeros(N)
    Su[(N-nux):] = 2*Utop /dy/dy 
    return Lu, Su, Lux, Luy

def gen_Lv_cavity(nx, ny, dx, dy):
    """
    Generate Laplacian operators for a staggered grid of size (ny-1) × nx (for v).

    Parameters:
        nx (int): Number of x grid points (including boundaries)
        ny (int): Number of y grid points (including boundaries)
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction

    Returns:
        Lv (csr_matrix): Combined Laplacian operator (1/dx² Lvx + 1/dy² Lvy)
        Lvx (csr_matrix): x-direction component (unscaled)
        Lvy (csr_matrix): y-direction component (unscaled)
    """
    nvx = nx
    nvy = ny - 1
    Nv = nvx * nvy  # total number of v points

    # --- x-direction Laplacian block (1D) with boundary treatment ---
    main_diag_x = -2 * np.ones(nvx)
    main_diag_x[0] = -3
    main_diag_x[-1] = -3
    off_diag_x = np.ones(nvx - 1)
    Lvx1D = diags([off_diag_x, main_diag_x, off_diag_x], offsets=[-1, 0, 1], format='csr')

    # 2D x-direction Laplacian using Kronecker product
    Iy = identity(nvy, format='csr')
    Lvx = kron(Iy, Lvx1D, format='csr')

    # --- y-direction Laplacian block (1D) ---
    main_diag_y = -2 * np.ones(nvy)
    off_diag_y = np.ones(nvy - 1)
    Lvy1D = diags([off_diag_y, main_diag_y, off_diag_y], offsets=[-1, 0, 1], format='csr')

    # 2D y-direction Laplacian using Kronecker product
    Ix = identity(nvx, format='csr')
    Lvy = kron(Lvy1D, Ix, format='csr')

    # --- Combine the Laplacians ---
    Lv = (1 / dx**2) * Lvx + (1 / dy**2) * Lvy

    return Lv, Lvx, Lvy


#===FIXED: run_cavity function with proper variable updates and post-processing================================
def run_cavity(lx, ly, nx, ny, Re, Utop, t_end, nit, caseName, script_dir):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    dx = lx/nx
    dy = ly/ny

    print(f'dx= {dx}')
    print(f'dy= {dy}')

    dt = np.float64(t_end / nit)
    print(f'dt = {dt}')
    

    nu = lx * Utop / Re
    print(f'nu = {nu}')

    App, ml = gen_Ap_cavity(nx, ny, dx, dy)

    #print(f"Starting simulation for {nit} iterations...")
    p = np.zeros([ny+2, nx+2])
    u = np.zeros([ny+2, nx+2])
    v = np.zeros([ny+2, nx+2])
    u_old = np.zeros_like(u)
    v_old = np.zeros_like(v)

    u[-1,:] = Utop

    Ek0 = 1e-6
    Ek_store = [Ek0]
    t_sim = [0.0]

    # Create output directories
    frames_dir = os.path.join(script_dir, 'postProcessing')
    #frames_dir2 = os.path.join(script_dir, 'divergence_frames')
    os.makedirs(frames_dir, exist_ok=True)
    #os.makedirs(frames_dir2, exist_ok=True)

    start_time = time.time()
    
    #print(f"Starting simulation for {nit} iterations...")
    
    for cont_it in range(1, nit+1):
        t_current = cont_it * dt
        
        # Apply boundary conditions
        u, v = apply_BC(u, v, Utop, caseName)
        
        # Do time integration - FIXED: Capture returned values
        u, v = do_time_integration(u, v, p, dt, dx, dy, nu, App, Utop, ml)
        
        # Calculate kinetic energy for monitoring
        u_center = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])
        v_center = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])
        Eki = 0.5 * np.sum(u_center**2 + v_center**2) * dx * dy

        Ek_store.append(Eki)
        Ek0 = Eki

        t = np.float64(cont_it * dt) 
        t_sim.append(t)
        
        # Post-processing every N steps
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Cavity Case ran SUCCESSFULLY")
    
    return u, v, p, Ek_store, t_sim
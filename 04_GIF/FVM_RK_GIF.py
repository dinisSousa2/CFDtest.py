#/*--------------------------------------------------------------------------------------------------------------*\                                                                                                                |
#|  ====== ===  ===    ======== ======== =======    ===  === ========                                             |
#|   || //  ||   \\     ||   \\  ||       ||         ||   \\  ||   \\                                             |
#|   ||//   ||   ||     ||   ||  ||  ===  ||==||     ||   ||  ||====||                                            |
#|   ||\\   ||   ||     ||   ||  ||   ||      ||     ||   ||  ||                                                  |
#|   || //   \\__||     ||__//    \\__||  ____//      \\__||  ||                                                  |
#\*--------------------------------------------------------------------------------------------------------------*/

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
from scipy.sparse.linalg import LinearOperator, cg # Corrected import for cg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import spilu, LinearOperator
import pyamg


class Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    dt: float


def save_residual_data(filename, times, data, header_info, storage_dir):
    filepath = os.path.join(storage_dir, filename)
    header = f"Time (t)\t{header_info}\nRe = {Re}, nx = {nx}, ny = {ny}, dt = {dt:.6f}"
    np.savetxt(
        filepath,
        np.column_stack((times, data)),
        header=header,
        delimiter='\t',
        fmt='%.18f',
        comments=''
    )
    print(f"Saved {filename} to: {filepath}")

# Function to save residual data to a file
def save_residual_data(filename, times, data, header_info, storage_dir):
    filepath = os.path.join(storage_dir, filename)
    header = f"Time (t)\t{header_info}\nRe = {Re}, nx = {nx}, ny = {ny}, dt = {dt:.6f}"
    np.savetxt(
        filepath,
        np.column_stack((times, data)),
        header=header,
        delimiter='\t',
        fmt='%.18f',
        comments=''
    )
    print(f"Saved {filename} to: {filepath}")


#=== Check Points ========================================================================================================================
def save_checkpoint(t, u, v, p, Ek_store, t_sim, residuals_u, residuals_v, residuals_p, 
                   residuals_omega, residuals_Ek, Uwall_mean_history, checkpoint_dir):
    """Save simulation state to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_Re{Re}_t{t:.4f}.npz")
    
    np.savez_compressed(
        checkpoint_file,
        t=t,
        u=u,
        v=v,
        p=p,
        Ek_store=np.array(Ek_store),
        t_sim=np.array(t_sim),
        residuals_u=np.array(residuals_u),
        residuals_v=np.array(residuals_v),
        residuals_p=np.array(residuals_p),
        residuals_omega=np.array(residuals_omega),
        residuals_Ek=np.array(residuals_Ek),
        Uwall_mean_history=np.array(Uwall_mean_history),
        Re=Re,
        nx=nx,
        ny=ny,
        dt=dt,
        nu=nu,
        Utop=Utop
    )
    print(f"Checkpoint saved at t={t:.4f} to {checkpoint_file}")

def load_checkpoint(checkpoint_path):
    """Load simulation state from a checkpoint file."""
    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        
        # Extract all variables
        t = float(data['t'])
        u = data['u']
        v = data['v']
        p = data['p']
        Ek_store = list(data['Ek_store'])
        t_sim = list(data['t_sim'])
        residuals_u = list(data['residuals_u'])
        residuals_v = list(data['residuals_v'])
        residuals_p = list(data['residuals_p'])
        residuals_omega = list(data['residuals_omega'])
        residuals_Ek = list(data['residuals_Ek'])
        Uwall_mean_history = list(data['Uwall_mean_history'])
        
        # Get parameters (these might not change between runs)
        Re = int(data['Re'])
        nx = int(data['nx'])
        ny = int(data['ny'])
        dt = float(data['dt'])
        nu = float(data['nu'])
        Utop = float(data['Utop'])
        
        print(f"Loaded checkpoint from t={t:.4f}")
        return (t, u, v, p, Ek_store, t_sim, residuals_u, residuals_v, residuals_p,
                residuals_omega, residuals_Ek, Uwall_mean_history, Re, nx, ny, dt, nu, Utop)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
#=========================================================================================================================================
# --- Add argument parsing ---
parser = argparse.ArgumentParser(description='Run lid-cavity simulation and post-process.')
parser.add_argument('--output_dir', type=str,
                    help='Directory to save output plots and data.',
                    default=None) # Set default to None initially
args = parser.parse_args()

# Determine the base directory for outputs
if args.output_dir:
    output_base_dir = args.output_dir
else:
    # Fallback to current script's directory if no output_dir is provided
    # This might be your original behavior for local testing
    output_base_dir = os.path.dirname(os.path.abspath(__file__))

# Modify where script_dir is defined globally or passed to postProcess
# If postProcess takes script_dir as an argument, you'll pass output_base_dir
# If script_dir is a global variable used by postProcess, then:
script_dir = output_base_dir # This will make all your outputs go to output_base_dir
print(f"Output files will be saved to: {script_dir}")
#==========================================================================================================================

def generate_Ap(nx, ny, dx, dy):
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


def apply_BC(u, v, Utop):
    # BC on the u velocity
    u[:,1] = 0.0 #Left Wall Impermiable
    u[:,-1] = 0.0 #Right Wall Impermiable
    u[0,:] =  -1.0 * u[1,:]   #Bottom Wall BC, Including Ghost Cells
    u[-1,:] = 2.0 * Utop - u[-2,:] #Top Wall BC, including Ghost Cell

    # BC on the v velocity
    v[-1,:] = 0.0 #Top Wall Impermiable
    v[1,:] = 0.0 #Bottom Wall Impermiable
    v[:,0] =  -1.0 * v[:,1] #Left Wall BC, including Ghost Cell
    v[:,-1] =  -1.0 * v[:,-2] #Right Wall
    return u, v

def compute_rhs(u, v, nu, dx, dy):
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    
    # Compute u-component RHS
    for i in range(2, nx+1):
        for j in range(1, ny+1):
            ue = 0.5 * (u[j,i+1] + u[j,i])
            uw = 0.5 * (u[j,i] + u[j,i-1])
            un = 0.5 * (u[j+1,i] + u[j,i])
            us = 0.5 * (u[j,i] + u[j-1,i])

            vn = 0.5 * (v[j+1, i-1] + v[j+1, i])
            vs = 0.5 * (v[j, i-1] + v[j, i])

            convection = -(ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy
            diffusion = nu * ((u[j,i-1] - 2.0 * u[j,i] + u[j,i+1])/dx/dx + 
                            (u[j-1,i] - 2.0 * u[j,i] + u[j+1,i])/dy/dy)
            ut[j,i] = convection + diffusion
    
    # Compute v-component RHS
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            ve = 0.5 * (v[j,i+1] + v[j,i])
            vw = 0.5 * (v[j,i] + v[j,i-1])
            ue = 0.5 * (u[j,i+1] + u[j-1,i+1])
            uw = 0.5 * (u[j,i] + u[j-1,i])

            vn = 0.5 * (v[j+1, i] + v[j, i])
            vs = 0.5 * (v[j, i] + v[j-1, i])

            convection = -(ue*ve - uw*vw)/dx - (vn*vn - vs*vs)/dy
            diffusion = nu * ((v[j,i+1] - 2.0 * v[j,i] + v[j,i-1])/dx/dx + 
                            (v[j+1,i] - 2.0 * v[j,i] + v[j-1,i])/dy/dy)
            vt[j,i] = convection + diffusion
    
    return ut, vt

def NS_operator(u, v, nu, dx, dy, dpdx, dpdy):
    diff_u = np.zeros_like(u)
    conv_u = np.zeros_like(u)

    diff_v = np.zeros_like(v)
    conv_v = np.zeros_like(v)


    RHS_u = np.zeros_like(u)
    RHS_v = np.zeros_like(v)

    NS_u = np.zeros_like(u)
    NS_v = np.zeros_like(v)


    # Compute u-component RHS
    for i in range(2, nx+1):
        for j in range(1, ny+1):
            ue = 0.5 * (u[j,i+1] + u[j,i])
            uw = 0.5 * (u[j,i] + u[j,i-1])
            un = 0.5 * (u[j+1,i] + u[j,i])
            us = 0.5 * (u[j,i] + u[j-1,i])

            vn = 0.5 * (v[j+1, i-1] + v[j+1, i])
            vs = 0.5 * (v[j, i-1] + v[j, i])

            conv_u[j,i] = -(ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy
            diff_u[j,i] = nu * ((u[j,i-1] - 2.0 * u[j,i] + u[j,i+1])/dx/dx + 
                            (u[j-1,i] - 2.0 * u[j,i] + u[j+1,i])/dy/dy)

            RHS_u = conv_u + diff_u
            
    # Compute v-component RHS
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            ve = 0.5 * (v[j,i+1] + v[j,i])
            vw = 0.5 * (v[j,i] + v[j,i-1])
            ue = 0.5 * (u[j,i+1] + u[j-1,i+1])
            uw = 0.5 * (u[j,i] + u[j-1,i])

            vn = 0.5 * (v[j+1, i] + v[j, i])
            vs = 0.5 * (v[j, i] + v[j-1, i])

            conv_v[j,i] = -(ue*ve - uw*vw)/dx - (vn*vn - vs*vs)/dy
            diff_v[j,i] = nu * ((v[j,i+1] - 2.0 * v[j,i] + v[j,i-1])/dx/dx + 
                            (v[j+1,i] - 2.0 * v[j,i] + v[j-1,i])/dy/dy)

            RHS_v = conv_v + diff_v

    NS_u[1:-1, 2:-1] = RHS_u[1:-1, 2:-1] - dpdx
    NS_v[2:-1, 1:-1] = RHS_v[2:-1, 1:-1] - dpdy

    return NS_u, NS_v, RHS_u, RHS_v

def pressure_correction(u, v, Ap, p, dt, dx, dy, ml):
    div = np.zeros_like(p)
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1])/dx + (v[2:, 1:-1] - v[1:-1, 1:-1])/dy
    
    prhs = div / dt
    
    # Use PCG instead of SOR     p, S, Ap, nx, ny
    p_new, residual = amg_solver(p.copy(), prhs, Ap, nx, ny, ml)
    
    # Apply pressure correction to velocities
    u_corr = u.copy()
    v_corr = v.copy()

    dpdx = (p_new[1:-1, 2:-1] - p_new[1:-1, 1:-2])/dx
    dpdy = (p_new[2:-1, 1:-1] - p_new[1:-2, 1:-1])/dy

    u_corr[1:-1, 2:-1] = u[1:-1, 2:-1] - dt * dpdx
    v_corr[2:-1, 1:-1] = v[2:-1, 1:-1] - dt * dpdy
    
    u_corr, v_corr = apply_BC(u_corr, v_corr, Utop)

    return u_corr, v_corr, p_new, dpdx, dpdy, residual


#===Check Point Setup===============================================================================================================
checkpoint_interval = 5000  # Save checkpoint every N steps
checkpoint_dir = os.path.join(script_dir, "checkpoints")
latest_checkpoint = None

#===Domain Setup====================================================================================================================
lx, ly = 1.0, 1.0
nx, ny = 16, 16
nelm = nx * ny
print(f'Nx = {nx}')
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
dx, dy = lx/nx, ly/ny
print(f' dx= {dx}')
xx, yy = np.meshgrid(x, y)

#===BC for Pressure=================================================================================================================
App, ml = generate_Ap(nx, ny, dx, dy)
print(f'ml = {ml}')

#===Flow parameters=================================================================================================================
Utop = 1.0  
Ubot = 0.0 
Vleft = 0.0 
Vright = 0.0 
Re = 100            # Reynolds number
nu = lx * Utop / Re  # Kinematic viscosity = 1 / Re, in practise

#===Time parameters=================================================================================================================
t = 0.0
t_end = 10
nit = int(100)
#nit = math.ceil(1.45*t_end*nu* (1.0/dx/dx + 1.0/dy/dy))
dt = np.float64(t_end / nit)
#print(dt)

#dt1 = 0.5/nu /(1.0/dx/dx + 1.0/dy/dy)
#dt2 = 2.0 * nu / (Utop**2)


#dt = min(dt1, dt2)
# Initialize arrays
p = np.zeros([ny+2, nx+2])
u = np.zeros([ny+2, nx+2])
v = np.zeros([ny+2, nx+2])
u_old = np.zeros_like(u)
v_old = np.zeros_like(v)

u[-1,:] = Utop

p_int = np.zeros([ny, nx])
u_int = np.zeros([ny, nx-1])
v_int = np.zeros([ny-1, nx])


Ek0 = 1e-6
dV = dx * dy
Ek_store = [Ek0]
t_sim = [0.0]

residuals_u = []
residuals_v = []
residuals_p = []
residuals_Ek = []
residuals_omega = []

resNSu = []
resNSv = []
resNSp = []

Uwallt = [0.0]
Uwall_mean_history = []  # To store mean Uwall over time
residuals_Uwall = []     # To store residuals of Uwall
counter = 0

# Time when wall starts moving (non-dimensional time)
t_start_moving = 0 * dt
Uwall_store = []  # To track how Uwall changes over time
# Create directory to store frame images
frames_dir = os.path.join(script_dir, "frames")
frames_dir2 = os.path.join(script_dir, "frames2")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(frames_dir2, exist_ok=True)

#nsave = 5 #Number of time steps to save
dt_i = dt

if os.path.exists(checkpoint_dir):
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_Re*.npz")))
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]

if latest_checkpoint:
    print(f"Found checkpoint file: {latest_checkpoint}")
    result = load_checkpoint(latest_checkpoint)
    if result:
        (t, u, v, p, Ek_store, t_sim, residuals_u, residuals_v, residuals_p,
         residuals_omega, residuals_Ek, Uwall_mean_history, 
         Re, nx, ny, dt, nu, Utop) = result
        
        # Recreate grid parameters since they weren't saved
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        dx, dy = lx/nx, ly/ny
        xx, yy = np.meshgrid(x, y)
        
        # Reinitialize Ap, Ae, Aw, An, As arrays
        Ap = np.zeros([ny+2, nx+2])
        Ae = 1.0/dx/dx * np.ones([ny+2, nx+2])
        Aw = 1.0/dx/dx * np.ones([ny+2, nx+2])
        An = 1.0/dy/dy * np.ones([ny+2, nx+2])
        As = 1.0/dy/dy * np.ones([ny+2, nx+2])
        
        # Boundary conditions for pressure
        Aw[1:-1, 1] = 0.0 
        Ae[1:-1, -2] = 0.0 
        An[-2, 1:-1] = 0.0
        As[1, 1:-1] = 0.0 
        Ap = - (Ae + Aw + An + As)
        
        # Set old variables for residual calculation
        u_old, v_old, p_old = u.copy(), v.copy(), p.copy()
        omega = np.zeros((ny, nx))  # Will be recalculated in first iteration
        omega_old = omega.copy()
        Uwall_old = abs(u[-1,1:] + u[-2,1:]) / 2
        Ek0 = Ek_store[-1]
        
        print(f"Restarted simulation from t={t:.4f}")
    else:
        print("Failed to load checkpoint, starting from scratch")
        # Initialize as before
else:
    print("No checkpoint found, starting new simulation")
    # Initialize as before

#=== RK4 AMG ===========================================================================================================================
start_time = time.time()
cont_it = int(0)

for cont_it in range(1, nit+1):
    if t < t_start_moving:
        Uwall = 0.0
    else:
        # Linear ramp-up over 0.2 time units (can adjust this duration)
        ramp_duration = 0.2
        if t < t_start_moving + ramp_duration:
            Uwall = Utop * (t - t_start_moving) / ramp_duration
        else:
            Uwall = Utop

    p0 = p.copy()
    u, v = apply_BC(u, v, Utop)

    # Stage 1: Compute explicit RHS (advection + diffusion only)
    k1u, k1v = compute_rhs(u, v, nu, dx, dy)
    u1 = u + 0.5*dt * k1u
    v1 = v + 0.5*dt * k1v
    u1, v1 = apply_BC(u1, v1, Utop)
    u1, v1, p1, _, _, _ = pressure_correction(u1, v1, App, p0, 0.5*dt, dx, dy, ml)  # Project here

    # Stage 2: Repeat with u1, v1
    k2u, k2v = compute_rhs(u1, v1, nu, dx, dy)
    u2 = u + 0.5*dt * k2u
    v2 = v + 0.5*dt * k2v
    u2, v2 = apply_BC(u2, v2, Utop)
    u2, v2, p2, _, _, _ = pressure_correction(u2, v2, App, p0, 0.5*dt, dx, dy, ml)  # Project again

    # Stage 3: Full step
    k3u, k3v = compute_rhs(u2, v2, nu, dx, dy)
    u3 = u + dt * k3u
    v3 = v + dt * k3v
    u3, v3 = apply_BC(u3, v3, Utop)
    u3, v3, p3, _, _, _ = pressure_correction(u3, v3, App, p0, dt, dx, dy, ml)  # Project again

    # Stage 4: Final explicit RHS
    k4u, k4v = compute_rhs(u3, v3, nu, dx, dy)

    # RK4 weighted update (without pressure yet)
    ut = u + (dt/6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    vt = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    ut, vt = apply_BC(ut, vt, Utop)

    # FINAL pressure correction (optional, but helps stability)
    ut, vt, p, _, _, Rp = pressure_correction(ut, vt, App, p0, dt, dx, dy, ml)

    # Update variables
    u, v = ut.copy(), vt.copy()
    u, v = apply_BC(u,v, Utop)

    #NS_u, NS_v, _, _ = NS_operator(u, v, nu, dx, dy, dpdxn, dpdyn)

    Ru = np.zeros([ny, nx+1])
    Rv = np.zeros([ny+1, nx])
    
    #Ru = (u[1:-1, 2:-1]-u_old[1:-1, 2:-1]) / dt - NS_u[1:-1, 2:-1]
    #Rv = (v[2:-1, 1:-1]-v_old[2:-1, 1:-1]) / dt - NS_v[2:-1, 1:-1]
    #NS_p = pressure_residual(p, u, v, dx, dy, dt, nx, ny, Ap, Ae, Aw, An, As)

    Ru = abs(u[1:-1, 2:-1]-u_old[1:-1, 2:-1]) / np.max(u_old[1:-1, 2:-1])
    Rv = abs(v[2:-1, 1:-1]-v_old[2:-1, 1:-1]) / np.max(u_old[1:-1, 2:-1])


    Uwalli = abs(u[-1,1:] + u[-2,1:]) / 2 
    Uwallt.append(Uwalli)

    # Calculate and store mean Uwall
    Uwall_mean = np.mean(Uwalli)
    Uwall_mean_history.append(Uwall_mean)

    #Vorticity Calculations
    dvdx = np.zeros((ny, nx))
    dudy = np.zeros((ny, nx))
    omega = np.zeros((ny, nx))
    # Calculate dvdx for all j and i at once
    dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)

    # Calculate dudy for all j and i at once
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)

    # Compute vorticity
    omega = dvdx - dudy
    
    ui_centers = 0.5*(u[1:-1,1:-1] + u[1:-1,2:])  # Average u to cell centers
    vi_centers = 0.5*(v[1:-1,1:-1] + v[2:,1:-1])
    
    Eki = 0.5 * np.sum(ui_centers**2 + vi_centers**2) / nelm
    #tol = abs(Ek0 - Eki) / Ek0
    Ek_store.append(Eki)
    
    if len(Ek_store) > 1:
        du = np.linalg.norm(u - u_old) / np.linalg.norm(u_old)  if 'u_old' in locals() else 1.0
        dv = np.linalg.norm(v - v_old) / np.linalg.norm(v_old) if 'v_old' in locals() else 1.0
        dp = np.linalg.norm(p - p_old) / np.linalg.norm(p_old) if 'p_old' in locals() else 1.0
        domega = np.linalg.norm(omega - omega_old) / np.linalg.norm(omega_old) if 'omega_old' in locals() else 1.0
        dEk = abs(Ek0 - Eki) / Ek0  if len(Ek_store) > 1 else 1.0
        
        
        residuals_u.append(du)
        residuals_v.append(dv)
        residuals_p.append(dp)
        residuals_omega.append(domega)
        residuals_Ek.append(dEk)


        dur = np.max(np.abs(Ru))  # Maximum absolute residual for u-momentum
        dvr = np.max(np.abs(Rv))  # Maximum absolute residual for v-momentum
        dpr = np.max(np.abs(Rp))  # Maximum absolute residual for p
        #dur = np.linalg.norm(NS_u)
        #dvr = np.linalg.norm(NS_v)

        resNSu.append(dur)
        resNSv.append(dvr) 
        resNSp.append(dpr)
    
    Ek0 = Eki
    u_old, v_old, p_old = u.copy(), v.copy(), p.copy()
    omega_old = omega.copy()
    Uwall_old = Uwalli.copy()
    counter += 1

    # Save frame every N steps
    save_interval = 2  # adjust as needed

    if int(t / dt) % save_interval == 0:
        #===== POST-PROCESSING =============================================================================================
        # Use mathtext (built-in LaTeX-like rendering)
        plt.rcParams.update({
            "mathtext.fontset": "cm",     # Computer Modern (LaTeX default)
            "font.family": "serif",       # Serif font to match LaTeX
            "font.size": 14
        })
        # 1. Create proper grid for visualization
        # For cell centers (pressure, velocity magnitude)
        x_centers = np.linspace(dx/2, lx-dx/2, nx)
        y_centers = np.linspace(dy/2, ly-dy/2, ny)
        xx_centers, yy_centers = np.meshgrid(x_centers, y_centers)

        # For staggered grid visualization
        x_u = np.linspace(0, lx, nx+1)  # u-velocity points
        y_u = np.linspace(dy/2, ly-dy/2, ny)
        xx_u, yy_u = np.meshgrid(x_u, y_u)

        x_v = np.linspace(dx/2, lx-dx/2, nx)
        y_v = np.linspace(0, ly, ny+1)  # v-velocity points
        xx_v, yy_v = np.meshgrid(x_v, y_v)

        # 2. Calculate velocity magnitude at cell centers (properly interpolated)
        u_centers = 0.5*(u[1:-1,1:-1] + u[1:-1,2:])  # Average u to cell centers
        v_centers = 0.5*(v[1:-1,1:-1] + v[2:,1:-1])  # Average v to cell centers
        velocity_magnitude = np.sqrt(u_centers**2 + v_centers**2)

        # ===== Plot 1: Velocity Magnitude =================================================================================
        #plt.figure(figsize=(6,6))
        #plt.contourf(xx_centers, yy_centers, velocity_magnitude, 20, cmap='jet')
        #plt.colorbar(label='Velocity Magnitude')
        #plt.title(f'Velocity Magnitude (Re={Re})')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.tight_layout()

        fig, ax = plt.subplots(figsize=(6,6))

        # Plot the contour with fixed range 0 to 1 - use levels to force the full range
        levels = np.linspace(0, 1, 21)  # 20 contour levels from 0 to 1
        cf = ax.contourf(xx_centers, yy_centers, velocity_magnitude, levels=levels, cmap='jet', vmin=0, vmax=1)

        # Create an axes for the colorbar that matches the height of the contour plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cf, cax=cax)

        # Set fixed ticks from 0 to 1 with step 0.2
        cbar_ticks = np.arange(0, 1.1, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        cbar.set_ticks(cbar_ticks)

        # Fix the LaTeX formatting
        cbar.set_ticklabels([r'$' + f'{tick:.1f}' + '$' for tick in cbar_ticks])

        # Set colorbar label with proper formatting
        cbar.set_label(r"$\left|\mathbf{V^*}\right|\,[-]$", fontsize=18)

        # Optional: make the tick numbers bigger
        cbar.ax.tick_params(labelsize=16)

        # Increase label and title sizes
        ax.set_title(rf"$Re = {Re},\quad t^* = {t:.3f}\,[-]$", fontsize=18)
        ax.set_xlabel(r"$x^*\,[-]$", fontsize=18)
        ax.set_ylabel(r"$y^*\,[-]$", fontsize=18)

        # Make axis tick numbers LaTeX
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%g$"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%g$"))

        ax.set_aspect('equal', adjustable='box')  # Square aspect ratio
        plt.tight_layout()

        # Save velocity magnitude plot

        frame_filename = os.path.join(frames_dir, f'frame_{int(t / dt):05d}.png')
        plt.savefig(frame_filename, dpi=150)
        plt.close()



        # --- Absolute Divergence -----------------------------------------------------------------------------------------
        divu = np.zeros_like(p)
        divu[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx + (v[2:, 1:-1] - v[1:-1, 1:-1]) / dy
        div_mag = np.abs(divu[1:-1, 1:-1])

        # Compute max absolute divergence and its location
        max_val = np.max(div_mag)  # This is the ACTUAL maximum value (e.g., 93.3)
        print(f'MAXVAL = {max_val}')
        j_max, i_max = np.unravel_index(np.argmax(np.abs(divu)), divu.shape)

        # Get coordinates of max divergence
        i_abs = i_max
        j_abs = j_max 

        fig, ax = plt.subplots(figsize=(5,5))

        # Plot the original data
        im = ax.imshow(div_mag, origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm')

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(r"$\left|\nabla^*\mathbf{V^*}\right|\,[-]$", fontsize=16)

        # Determine the appropriate order of magnitude for scaling
        if max_val > 0:
            order_of_magnitude = np.floor(np.log10(max_val))
            scale_factor = 10 ** (-order_of_magnitude)
            scaled_max = max_val * scale_factor
            
            # Create nice ticks from 0 to scaled_max (rounded up)
            nice_max = np.ceil(scaled_max)
            
            # Use integer ticks instead of linspace for better readability
            if nice_max <= 10:
                # For small ranges, use integer ticks
                cbar_ticks = np.arange(0, nice_max + 1, 1)
            else:
                # For larger ranges, use fewer ticks but still integers
                step = max(1, int(np.ceil(nice_max / 8)))
                cbar_ticks = np.arange(0, nice_max + step, step)
            
            # Scale the data and update the image (for visualization only)
            scaled_data = div_mag * scale_factor
            im.set_data(scaled_data)
            im.set_clim(0, nice_max)
            
            # Update colorbar ticks
            cbar.set_ticks(cbar_ticks)
            # FIXED: Use proper formatting to avoid scientific notation
            cbar.set_ticklabels([r'$' + f'{int(tick)}' + '$' if tick.is_integer() else r'$' + f'{tick:.1f}' + '$' for tick in cbar_ticks])
            
            # Add the scaling factor text to colorbar
            if order_of_magnitude != 0:
                exponent = int(order_of_magnitude)
                scale_text = r'$\times 10^{' + str(exponent) + '}$'
                cbar.ax.text(0.5, 1.02, scale_text, transform=cbar.ax.transAxes, 
                            fontsize=14, ha='center', va='bottom')
        else:
            # Handle case where all values are zero or negative
            cbar.set_ticklabels([r'$0$'])

        # Format the ACTUAL max value for the title (not the scaled one!)
        if max_val > 0:
            # Find the appropriate order of magnitude for the actual max value
            actual_order = np.floor(np.log10(max_val))
            if actual_order != 0:
                actual_mantissa = max_val / (10 ** actual_order)
                max_val_formatted = rf"{actual_mantissa:.3f} \times 10^{{{int(actual_order)}}}"
            else:
                max_val_formatted = f"{max_val:.3f}"
        else:
            max_val_formatted = "0.000"

        # Optional: make the tick numbers bigger
        cbar.ax.tick_params(labelsize=14)

        # Title and labels with properly formatted ACTUAL max value
        ax.set_title(rf"$Re = {Re}$" 
                    "\n" 
                    rf"$|\nabla^* \mathbf{{V^*}}|_{{max}} = {max_val_formatted}\;[-]$" 
                    "\n"
                    rf"$i_{{max}} = {i_abs:.0f},\ j_{{max}} = {j_abs:.0f}$", fontsize=14)
        ax.set_xlabel(r"$x^*\,[-]$", fontsize=16)
        ax.set_ylabel(r"$y^*\,[-]$", fontsize=16)

        # Make axis tick numbers LaTeX
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%g$"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%g$"))

        # Square aspect ratio and layout
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        # Save divergence magnitude plot
        frame_filename2 = os.path.join(frames_dir2, f'frame_{int(t / dt):05d}.png')
        plt.savefig(frame_filename2, dpi=150)
        plt.close()


    t = np.float64(cont_it * dt) 
    t_sim.append(t)
    dt_i = min(dt, t_end - t)


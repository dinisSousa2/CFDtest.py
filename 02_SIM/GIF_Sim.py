#/*===============================================================================================================*\                                                                                                                |
#|  ======== ======== =======                                                                                      |
#|   ||   \\  ||       ||                                                                                          |
#|   ||   ||  ||  ===  ||==||                                                                                      |
#|   ||   ||  ||   ||      ||                                                                                      |
#|   ||__//    \\__||  ____//                                                                                      |
#\*===============================================================================================================*/
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


#====================================================================================================================
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
    output_base_dir = os.path.dirname(os.path.abspath(__file__))

script_dir = output_base_dir
print(f"Output files will be saved to: {script_dir}")

#========Add directories paths to main simulation==============================================================
sys.path.append(os.path.join(os.path.dirname(__file__), '../00_AUX'))
from postProcess import postProcessing
from linearSolvers import cg_solver
from linearSolvers import amg_solver


#=======Main Simulation Settings=======================================================================================
def load_simulation_settings(filename='20_simSettings.txt'):
    """Load simulation settings from text file as dictionary"""
    # Get the absolute path to the settings file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    print(f"Looking for settings file at: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in script directory: {os.listdir(script_dir)}")
        sys.exit(1)
    
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('*/'):
                continue
            # Parse key = value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to evaluate the value (handles numbers, strings, etc.)
                try:
                    settings[key] = eval(value)
                except:
                    # If eval fails, keep as string
                    settings[key] = value
    return settings

# Load settings with error handling
try:
    sim_settings = load_simulation_settings('20_simSettings.txt')
    
    # Access values
    caseName = sim_settings['caseName']
    lx = sim_settings['lx']
    ly = sim_settings['ly'] 
    nx = sim_settings['nx']
    ny = sim_settings['ny']
    t_end = sim_settings['tend']
    nit = sim_settings['nit']
    Re = sim_settings['Re']
    mode = sim_settings['mode']

    spatialScheme = sim_settings['spatialScheme']
    temporalScheme = sim_settings['temporalScheme']
    pressureProjection = sim_settings['pressureProjection']
    velSolver = sim_settings['velSolver']

    Utop = sim_settings['Utop']


    print(f"Running {caseName} case with Re={Re}, grid={nx}x{ny}, sScheme {spatialScheme}, tScheme {temporalScheme}, pSolver {pressureProjection}")

except Exception as e:
    print(f"ERROR loading settings: {e}")
    sys.exit(1)

#===Cavity Case==================================================================
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



def NS_operator(u, v, nu, dx, dy):
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

#===FIXED: run_cavity function with proper variable updates and post-processing================================
def run_cavity(lx, ly, nx, ny, Re, Utop):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    dx = lx/nx
    dy = ly/ny

    print(f' dx= {dx}')
    print(f' dy= {dy}')

    dt = np.float64(t_end / nit)
    print(f'dt = {dt}')

    nu = lx * Utop / Re
    print(f'nu = {nu}')

    App, ml = generate_Ap(nx, ny, dx, dy)

    p = np.zeros([ny+2, nx+2])
    u = np.zeros([ny+2, nx+2])
    v = np.zeros([ny+2, nx+2])

    u[-1,:] = Utop

    # Create output directories
    frames_dir = os.path.join(script_dir, 'velocity_frames')
    frames_dir2 = os.path.join(script_dir, 'divergence_frames')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(frames_dir2, exist_ok=True)

    start_time = time.time()
    
    #print(f"Starting simulation for {nit} iterations...")
    
    for cont_it in range(1, nit+1):
        t_current = cont_it * dt
        
        # Apply boundary conditions
        u, v = apply_BC(u, v, Utop)
        
        # Do time integration - FIXED: Capture returned values
        u, v = do_time_integration(u, v, p, dt, dx, dy, nu, App, Utop, ml)
        
        # Calculate kinetic energy for monitoring
        u_center = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])
        v_center = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])
        Ek = 0.5 * np.sum(u_center**2 + v_center**2) * dx * dy
        
        # Post-processing every N steps
        if cont_it % max(1, nit//20) == 0 or cont_it == 1:
            #print(f"Iteration {cont_it}/{nit}, t={t_current:.4f}, Ek={Ek:.6e}")
            
            # Call post-processing
            try:
                postProcessing(u, v, p, nx, ny, lx, ly, dx, dy, dt, t_current, Re, frames_dir, frames_dir2)
            except Exception as e:
                print(f"Warning: Post-processing failed: {e}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Cavity Case ran SUCCESSFULLY")
    
    return u, v, p

#===FIXED: run_simulation function with proper returns=======================================================
def run_simulation():
    if caseName == 'cavity':
        u_final, v_final, p_final = run_cavity(lx, ly, nx, ny, Re, Utop)
        return u_final, v_final, p_final
    else:
        print(f"Unknown case: {caseName}")
        return None, None, None

def do_time_integration(u, v, p, dt, dx, dy, nu, App, Utop, ml):
    if temporalScheme == 'RK4':
        u, v = run_RK4(u, v, p, dt, dx, dy, nu, App, Utop, ml)  # FIXED: Capture return values

    return u, v

def run_RK4(u, v, p, dt, dx, dy, nu, App, Utop, ml):
    # Stage 1: Compute explicit RHS (advection + diffusion only)
    k1u, k1v = NS_operator(u, v, nu, dx, dy)
    u1 = u + 0.5*dt * k1u
    v1 = v + 0.5*dt * k1v
    u1, v1 = apply_BC(u1, v1, Utop)
    u1, v1, _, _, _, _ = pressure_correction(u1, v1, App, p, 0.5*dt, dx, dy, ml)  # Project here

    # Stage 2: Repeat with u1, v1
    k2u, k2v = NS_operator(u1, v1, nu, dx, dy)
    u2 = u + 0.5*dt * k2u
    v2 = v + 0.5*dt * k2v
    u2, v2 = apply_BC(u2, v2, Utop)
    u2, v2, _, _, _, _ = pressure_correction(u2, v2, App, p, 0.5*dt, dx, dy, ml)  # Project again

    # Stage 3: Full step
    k3u, k3v = NS_operator(u2, v2, nu, dx, dy)
    u3 = u + dt * k3u
    v3 = v + dt * k3v
    u3, v3 = apply_BC(u3, v3, Utop)
    u3, v3, _, _, _, _ = pressure_correction(u3, v3, App, p, dt, dx, dy, ml)  # Project again

    # Stage 4: Final explicit RHS
    k4u, k4v = NS_operator(u3, v3, nu, dx, dy)

    # RK4 weighted update (without pressure yet)
    ut = u + (dt/6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    vt = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    ut, vt = apply_BC(ut, vt, Utop)

    # FINAL pressure correction (optional, but helps stability)
    ut, vt, p, _, _, Rp = pressure_correction(ut, vt, App, p, dt, dx, dy, ml)

    # Update variables
    u, v = ut.copy(), vt.copy()
    u, v = apply_BC(u, v, Utop)

    return u, v

#===MAIN EXECUTION BLOCK - ADD THIS AT THE VERY END==========================================
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("STARTING LID-DRIVEN CAVITY SIMULATION")
        print("=" * 70)
        
        # Run the simulation
        u_final, v_final, p_final = run_simulation()
        
        if u_final is not None:
            # Final post-processing
            print("Performing final post-processing...")
            frames_dir = os.path.join(script_dir, 'velocity_frames')
            frames_dir2 = os.path.join(script_dir, 'divergence_frames')
            dt_sim = t_end / nit
            postProcessing(u_final, v_final, p_final, nx, ny, lx, ly, lx/nx, lx/ny, dt_sim, t_end, Re, frames_dir, frames_dir2)
            
            print("=" * 70)
            print("SIMULATION COMPLETED SUCCESSFULLY!")
            print(f"Results saved in: {script_dir}")
            print("=" * 70)
        else:
            print("Simulation failed to produce results.")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
#/*====================================================================================================================*\                                                                                                                |
#|  ======== ======== =======   ======== ======== ========  ======== ======== ======== ========   ======== ========     |
#|   ||   \\  ||       ||        ||       ||       ||   \\     ||     ||       ||         ||       ||   ||  \\  //      |
#|   ||   ||  ||  ===  ||==||    ||       ||====   ||   ||     ||     ||====   ||==||     ||       ||===||   \\//       |
#|   ||   ||  ||   ||      ||    ||       ||       ||   ||     ||     ||           ||     ||       ||         ||        |
#|   ||__//    \\__||  ____//    ||_____  ||       ||__//      ||     ||_____  ____//     ||   ::  ||         ||        |
#\*====================================================================================================================*/
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

#===Boundary Conditions==================================================================
def apply_BC(u, v, Utop, caseName):
    if caseName == 'cavity':
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
    elif caseName == '2DpipeFlow':
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



#===FIXED: run_simulation function with proper returns=======================================================
def run_simulation():
    if caseName == 'cavity':
        sys.path.append(os.path.join(os.path.dirname(__file__), '../03_cases'))
        from cavity import run_cavity

        u_final, v_final, p_final, Ek_store, t_sim = run_cavity(lx, ly, nx, ny, Re, Utop, t_end, nit, caseName, script_dir)
        return u_final, v_final, p_final, Ek_store, t_sim
    else:
        print(f"Unknown case: {caseName}")
        return None, None, None

def do_time_integration(u, v, p, dt, dx, dy, nu, App, Utop, ml):
    try:

        if temporalScheme == 'RK4':
            tschemes_path = os.path.join(os.path.dirname(__file__), '..', '01_nSchemes', '10_tSchemes')
            if tschemes_path not in sys.path:
                sys.path.append(tschemes_path)
            from RK4 import run_RK4

            u, v = run_RK4(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, pressureProjection, caseName)
            #print('RK4 done')
            return u, v
        
        elif temporalScheme == 'FE':
            tschemes_path = os.path.join(os.path.dirname(__file__), '..', '01_nSchemes', '10_tSchemes')
            if tschemes_path not in sys.path:
                sys.path.append(tschemes_path)
            from FE import run_FE

            u, v = run_FE(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, pressureProjection, caseName)
            return u, v
        elif temporalScheme == 'ASC222':
            tschemes_path = os.path.join(os.path.dirname(__file__), '..', '01_nSchemes', '10_tSchemes')
            if tschemes_path not in sys.path:
                sys.path.append(tschemes_path)
            from IMEX import run_IMEX

            gamma = (2-math.sqrt(2))/2
            d222 = 1- (1/(2*gamma))
            #print(f'gamma = {gamma}')
            A = np.array([
                [gamma,   0     ],
                [1-gamma,   gamma]
            ])
            b = A[-1]
            c = np.sum(A, axis=1)

            # Explicit (A_hat, b_hat)
            A_hat = np.array([
                [0,     0,     0],
                [gamma,   0,     0],
                [d222, 1-d222,  0]
            ])
            b_hat = A_hat[-1]
            c_hat = np.sum(A_hat, axis=1)

            if caseName == 'cavity':
                sys.path.append(os.path.join(os.path.dirname(__file__), '../03_cases'))
                from cavity import gen_Lu_cavity, gen_Lv_cavity

                Lu, Su, _, _ = gen_Lu_cavity(nx, ny, dx, dy, Utop)
                Lv, _, _ = gen_Lv_cavity(nx, ny, dx, dy)
                Iu = eye(ny * (nx-1))
                Iv = eye((ny-1) * nx)

            u, v = run_IMEX(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, c, A, A_hat, b, b_hat, Iu, Iv, Lu, Lv, Su, pressureProjection, caseName)
            return u, v  # FIXED: Ensure return statement

        elif temporalScheme == 'ASC443':
            tschemes_path = os.path.join(os.path.dirname(__file__), '..', '01_nSchemes', '10_tSchemes')
            if tschemes_path not in sys.path:
                sys.path.append(tschemes_path)
            from IMEX import run_IMEX

            A = np.array([
                [0.5,   0,     0,     0],
                [1/6,   0.5,   0,     0],
                [-0.5,  0.5,   0.5,   0],
                [1.5,  -1.5,   0.5,   0.5]
            ])
            b = A[-1]
            c = np.sum(A, axis=1)

            # Explicit (A_hat, b_hat)
            A_hat = np.array([
                [0,     0,     0,     0, 0],
                [0.5,   0,     0,     0, 0],
                [11/18, 1/18,  0,     0, 0],
                [5/6,  -5/6,   1/2,     0, 0],
                [1/4, 7/4, 3/4, -7/4, 0]
            ])
            b_hat = A_hat[-1]
            c_hat = np.sum(A_hat, axis=1)
            if caseName == 'cavity':
                sys.path.append(os.path.join(os.path.dirname(__file__), '../03_cases'))
                from cavity import gen_Lu_cavity, gen_Lv_cavity
                
                Lu, Su, _, _ = gen_Lu_cavity(nx, ny, dx, dy, Utop)
                Lv, _, _ = gen_Lv_cavity(nx, ny, dx, dy)
                Iu = eye(ny * (nx-1))
                Iv = eye((ny-1) * nx)

            u, v = run_IMEX(u, v, p, dt, dx, dy, nx, ny, nu, App, Utop, ml, c, A, A_hat, b, b_hat, Iu, Iv, Lu, Lv, Su, pressureProjection, caseName)
            return u, v  # FIXED: Ensure return statement

        else:
            print(f"Unknown Time Integrator: {temporalScheme}")
            return u, v

    except Exception as e:
        print(f"Error in do_time_integration: {e}")
        import traceback
        traceback.print_exc()
        return u, v  # Return original values on error


#===MAIN EXECUTION BLOCK==========================================
if __name__ == "__main__":
    try:
        print("""/*====================================================================================================================*\\
|  ======== ======== =======   ======== ======== ========  ======== ======== ======== ========   ======== ========     |
|   ||   \\\\  ||       ||        ||       ||       ||   \\\\     ||     ||       ||         ||       ||   ||  \\\\  //      |
|   ||   ||  ||  ===  ||==||    ||       ||====   ||   ||     ||     ||====   ||==||     ||       ||===||   \\\\//       |
|   ||   ||  ||   ||      ||    ||       ||       ||   ||     ||     ||           ||     ||       ||         ||        |
|   ||__//    \\\\__||  ____//    ||_____  ||       ||__//      ||     ||_____  ____//     ||   ::  ||         ||        |
\\*====================================================================================================================*/""")
    
        print("=" * 120)
        print("STARTING LID-DRIVEN CAVITY SIMULATION")
        print("=" * 120)
        
        # Run the simulation
        u_final, v_final, p_final, Ek_store, t_sim = run_simulation()
        
        if u_final is not None:
            # Final post-processing - USE THE SAME DIRECTORY FOR BOTH
            print("====Performing final post-processing====")
            frames_dir = os.path.join(script_dir, 'postProcessing')  # Single directory
            dt_sim = t_end / nit
            
            # Call postProcessing with the same directory for both parameters
            postProcessing(u_final, v_final, p_final, nx, ny, lx, ly, lx/nx, ly/ny, dt_sim, t_end, Re, frames_dir, frames_dir, Ek_store, t_sim)
            
            print("=" * 120)
            print("***SIMULATION COMPLETED SUCCESSFULLY!***")
            print(f"Results saved in: {frames_dir}")
            print("=" * 120)
        else:
            print("Simulation failed to produce results.")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
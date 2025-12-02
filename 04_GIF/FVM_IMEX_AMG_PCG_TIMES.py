#/*--------------------------------------------------------------------------------------------------------------*\                                                                                                                |
#|  ====== ===  ===    ======== ======== =======    ===  === ========                                             |
#|   || //  ||   \\     ||   \\  ||       ||         ||   \\  ||   \\                                             |
#|   ||//   ||   ||     ||   ||  ||  ===  ||==||     ||   ||  ||====||                                            |
#|   ||\\   ||   ||     ||   ||  ||   ||      ||     ||   ||  ||                                                  |
#|   || //   \\__||     ||__//    \\__||  ____//      \\__||  ||                                                  |
#\*--------------------------------------------------------------------------------------------------------------*/

import numpy as np
import matplotlib.pyplot as plt
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

# Function to save residual data to a file
def postProcess(u, v, dx, dy, nx, ny, lx, ly, x, y, xx, yy, t,  Re, nu, Utop, p, omega, Ek_store, residuals_u, residuals_v, residuals_p, residuals_Ek, t_sim, Uwall_mean_history, script_dir, resNSu,resNSv, resNSp, Ru, Rv, Rp):

    #===== POST-PROCESSING =============================================================================================

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

    plt.figure(figsize=(6,6))
    plt.contourf(xx_centers, yy_centers, velocity_magnitude, 20, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')
    plt.title(f'Velocity Magnitude (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save velocity magnitude plot
    velmag_path = os.path.join(script_dir, f'O1_VelocityMagnitude_Re{Re}.png')
    plt.savefig(velmag_path, dpi=300, bbox_inches='tight')
    print(f"Velocity magnitude plot saved to: {velmag_path}")
    plt.close()



    # ===== Plot 2: Streamlines ========================================================================================
    plt.figure(figsize=(6,6))
    plt.streamplot(xx_centers, yy_centers, u_centers, v_centers,
                color='black', density=1.5, linewidth=1, arrowsize=1)
    plt.title(f'Streamlines (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save streamline plot
    streamline_path = os.path.join(script_dir, f'O2_Streamlines_Re{Re}.png')
    plt.savefig(streamline_path, dpi=300, bbox_inches='tight')
    print(f"Streamline plot saved to: {streamline_path}")
    plt.close()

    # ===== Plot 3: Staggered Grid Visualization ========================================================================
    plt.figure(figsize=(8,8))

    # Plot pressure points (cell centers)
    plt.scatter(xx_centers, yy_centers, c='red', s=20, label='Pressure points')

    # Plot u-velocity points (staggered in x)
    plt.scatter(xx_u, yy_u, c='blue', marker='s', s=30, label='u-velocity points')

    # Plot v-velocity points (staggered in y)
    plt.scatter(xx_v, yy_v, c='green', marker='^', s=30, label='v-velocity points')

    # Add grid lines
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.title(f'Staggered Grid Layout (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save staggered grid plot
    staggered_path = os.path.join(script_dir, f'O3_StaggeredGrid_Re{Re}.png')
    plt.savefig(staggered_path, dpi=300, bbox_inches='tight')
    print(f"Staggered grid plot saved to: {staggered_path}")
    plt.close()

    # ===== Plot 4: Combined Velocity Magnitude + Streamlines ==========================================================
    plt.figure(figsize=(6,6))
    plt.contourf(xx_centers, yy_centers, velocity_magnitude, 20, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')
    plt.streamplot(xx_centers, yy_centers, u_centers, v_centers,
                color='white', density=1.5, linewidth=1, arrowsize=1)
    plt.title(f'Velocity Magnitude with Streamlines (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save combined plot
    combined_path = os.path.join(script_dir, f'O16_Combined_Re{Re}.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {combined_path}")
    plt.close()


    # ===== Plot 5: Staggered Velocity Components ========================================================================
    plt.figure(figsize=(8,6))

    # Subsample for clearer visualization
    skip = 2  # Plot every 2nd point (adjust as needed)

    # Prepare u-velocity components (horizontal arrows)
    u_plot = u[1:-1,1:-1][::skip,::skip]
    xx_u_plot = xx_u[::skip,::skip][:u_plot.shape[0], :u_plot.shape[1]]
    yy_u_plot = yy_u[::skip,::skip][:u_plot.shape[0], :u_plot.shape[1]]

    # Prepare v-velocity components (vertical arrows)
    v_plot = v[1:-1,1:-1][::skip,::skip]
    xx_v_plot = xx_v[::skip,::skip][:v_plot.shape[0], :v_plot.shape[1]]
    yy_v_plot = yy_v[::skip,::skip][:v_plot.shape[0], :v_plot.shape[1]]

    # Plot u-components (horizontal)
    plt.quiver(xx_u_plot, yy_u_plot, 
            u_plot, np.zeros_like(u_plot),
            scale=30, color='red', width=0.003, 
            label='u-velocity (horizontal)')

    # Plot v-components (vertical)
    plt.quiver(xx_v_plot, yy_v_plot, 
            np.zeros_like(v_plot), v_plot,
            scale=30, color='blue', width=0.003,
            label='v-velocity (vertical)')

    # Add grid and styling
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.title(f'Staggered Velocity Components (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot
    vectors_path = os.path.join(script_dir, f'O4_VelocityVectors_Re{Re}.png')
    plt.savefig(vectors_path, dpi=300, bbox_inches='tight')
    print(f"Velocity vectors plot saved to: {vectors_path}")
    plt.close()
    #-------------------------------------------------------------------------------------------------------------------
    # --- Absolute Divergence -----------------------------------------------------------------------------------------
    divu = np.zeros_like(p)
    divu[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx + (v[2:, 1:-1] - v[1:-1, 1:-1]) / dy
    div_mag = np.abs(divu[1:-1, 1:-1])

    # Compute max absolute divergence and its location
    max_div = np.max(div_mag)
    j_max, i_max = np.unravel_index(np.argmax(np.abs(divu)), divu.shape)

    # Get coordinates of max divergence
    i_abs = i_max
    j_abs = j_max 

    plt.figure(figsize=(5, 5))
    plt.imshow(div_mag, origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm', 
            vmin=0, vmax=np.max(div_mag))
    plt.colorbar(label='|divV|')
    plt.title(f'Absolute Divergence (Re={Re})\nMax: {np.max(div_mag):.3e}\ni={i_abs:.0f}, j={j_abs:.0f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    div_path = os.path.join(script_dir, f'O6_Re{Re}_{nx}_{ny}.png')
    plt.savefig(div_path, dpi=300, bbox_inches='tight')
    print(f"Divergence magnitude plot saved to: {div_path}")
    plt.close()
    #plt.show()

    # --- Kinetic Energy Over Time ------------------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t_sim, Ek_store, 'b-', linewidth=2, label='Kinetic Energy')
    plt.xlabel('Simulation Time (t)')
    plt.ylabel('Kinetic Energy (Ek)')
    plt.title(f'Kinetic Energy Evolution (Re = {Re})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot to the script's directory
    plot_filename = f"O5_Ek_Re{Re}_{nx}_{ny}_{dt:.2f}.png"           # Dynamic filename
    plot_path = os.path.join(script_dir, plot_filename)     # Full path
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')    # Save with high resolution
    print(f"Plot saved to: {plot_path}")
    plt.close()

    #plt.show()  # Display the plot (optional)



    # --- Pressure Field Visualization ---------------------------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, p[1:-1, 1:-1], 20, cmap='viridis')  # Exclude ghost cells
    plt.colorbar(label='Pressure (p)')
    plt.title(f'Pressure Field (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save the plot
    pressure_path = os.path.join(script_dir, f'O7_Pressure_Re{Re}.png')
    plt.savefig(pressure_path, dpi=300, bbox_inches='tight')
    print(f"Pressure field plot saved to: {pressure_path}")
    plt.close()
    #plt.show()

    #=== Pressure Gradient coutours countour plot
    # Calculate pressure gradients
    dpdx = (p[1:-1, 2:] - p[1:-1, 1:-1])/dx 
    dpdy = (p[2:, 1:-1] - p[1:-1, 1:-1])/dy

    # ===== Plot 7a: dp/dx Contour ======================================================================
    plt.figure(figsize=(5, 5))
    plt.contourf(xx_centers, yy_centers, dpdx, 20, cmap='rainbow')
    plt.colorbar(label='dp/dx')
    plt.title(f'Pressure Gradient X-Component (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save dpdx plot
    dpdx_path = os.path.join(script_dir, f'O7a_dpdx_Re{Re}.png')
    plt.savefig(dpdx_path, dpi=300, bbox_inches='tight')
    print(f"dp/dx plot saved to: {dpdx_path}")
    plt.close()

    # ===== Plot 7b: dp/dy Contour ======================================================================
    plt.figure(figsize=(5, 5))
    plt.contourf(xx_centers, yy_centers, dpdy, 20, cmap='rainbow')
    plt.colorbar(label='dp/dy')
    plt.title(f'Pressure Gradient Y-Component (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save dpdy plot
    dpdy_path = os.path.join(script_dir, f'O7b_dpdy_Re{Re}.png')
    plt.savefig(dpdy_path, dpi=300, bbox_inches='tight')
    print(f"dp/dy plot saved to: {dpdy_path}")
    plt.close()

    #===u-velocity Residual===========
    xx2_u, yy2_u = np.meshgrid(x_u[1:-1], y_u)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx2_u, yy2_u, abs(Ru), 20, cmap='plasma')  # Exclude ghost cells
    plt.colorbar(label='Ru')
    plt.title(f'Ru (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save the plot
    pressure_path = os.path.join(script_dir, f'AA_uResidual_Re{Re}.png')
    plt.savefig(pressure_path, dpi=300, bbox_inches='tight')
    print(f"Pressure field plot saved to: {pressure_path}")
    plt.close()
    #plt.show()

    #===v-velocity Residual===========
    xx3_v, yy3_v = np.meshgrid(x_v, y_v[1:-1])
    plt.figure(figsize=(5, 5))
    plt.contourf(xx3_v, yy3_v, abs(Rv), 20, cmap='plasma')  # Exclude ghost cells
    plt.colorbar(label='Rv')
    plt.title(f'Rv (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save the plot
    pressure_path = os.path.join(script_dir, f'AB_vResidual_Re{Re}.png')
    plt.savefig(pressure_path, dpi=300, bbox_inches='tight')
    print(f"Pressure field plot saved to: {pressure_path}")
    plt.close()
    #plt.show()

    #===p-Residual===========

    plt.figure(figsize=(5, 5))
    plt.contourf(xx_centers, yy_centers, abs(Rp), 20, cmap='jet')  # Exclude ghost cells
    plt.colorbar(label='Rp')
    plt.title(f'Rv (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save the plot
    pressure_path = os.path.join(script_dir, f'AC_pResidual_Re{Re}.png')
    plt.savefig(pressure_path, dpi=300, bbox_inches='tight')
    print(f"Pressure field plot saved to: {pressure_path}")
    plt.close()
    #plt.show()

    #===RelError Plot (Log-Log Scale)========================================================================= 
    plt.figure(figsize=(10, 6))

    # Use the same time points but skip first step (no residual yet)
    plot_times = t_sim[1:]

    plt.loglog(plot_times, residuals_u, 'r-', linewidth=2, label='u-velocity relError')
    plt.loglog(plot_times, residuals_v, 'b--', linewidth=2, label='v-velocity relError')
    plt.loglog(plot_times, residuals_p, 'g-.', linewidth=2, label='pressure relError')
    plt.loglog(plot_times, residuals_omega, 'y-', linewidth=2, label='omega relError')
    plt.loglog(plot_times, residuals_Ek, 'm:', linewidth=2, label='Ek relError')
    


    plt.xlabel('Time (log scale)')
    plt.ylabel('relError (log scale)')
    plt.title(f'relError Evolution (Re={Re}) - Log-Log Scale')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    # Add reference lines for visualization
    for exponent in [-2, -4, -6, -8]:
        plt.axhline(y=10**exponent, color='k', linestyle=':', alpha=0.2)

    # Save the plot
    residual_path = os.path.join(script_dir, f'O8_RelError_Re{Re}.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    print(f"RelError plot saved to: {residual_path}")
    plt.close()
    #plt.show()

    #-------------------------------------------------------------------------------------------------------------------
    storage_dir = os.path.join(script_dir, "Simulation_Data")
    os.makedirs(storage_dir, exist_ok=True)  # Create directory if it doesn't exist

    
    # Save each residual to its own file (only if we have data)
    if len(residuals_u) > 0:
        save_residual_data(
            f"u_relError_Re{Re}.txt",
            t_sim[1:],  # Skip first time point since residuals start at second step
            residuals_u,
            "u-velocity RelError", storage_dir
        )

    if len(residuals_v) > 0:
        save_residual_data(
            f"v_relError_Re{Re}.txt",
            t_sim[1:],
            residuals_v,
            "v-velocity relError", storage_dir
        )

    if len(residuals_p) > 0:
        save_residual_data(
            f"p_relError_Re{Re}.txt",
            t_sim[1:],
            residuals_p,
            "pressure relError", storage_dir
        )

    if len(residuals_omega) > 0:
        save_residual_data(
            f"omega_relError_Re{Re}.txt",
            t_sim[1:],
            residuals_omega,
            "Omega relError", storage_dir
        )

    if len(residuals_Ek) > 0:
        save_residual_data(
            f"Ek_relError_Re{Re}.txt",
            t_sim[1:],
            residuals_Ek,
            "Kinetic Energy relError", storage_dir
        )
    
    if len(dpdx) > 0:
        save_residual_data(
            f"dpdx_Re{Re}.txt",
            x_centers,
            dpdx.mean(axis=0),  # Average over y-direction
            "dp/dx", storage_dir
        )

    if len(dpdy) > 0:
        save_residual_data(
            f"dpdy_Re{Re}.txt",
            y_centers,
            dpdy.mean(axis=1),  # Average over x-direction
            "dp/dy", storage_dir
        )


    # Also save the main simulation data to the storage directory
    main_data_path = os.path.join(storage_dir, f"main_simulation_data_Re{Re}.txt")
    np.savetxt(
        main_data_path,
        np.column_stack((t_sim, Ek_store)),
        header=f"Time (t)\tKinetic Energy (Ek)\nRe = {Re}, nx = {nx}, ny = {ny}, dt = {dt:.18f}",
        delimiter='\t',
        fmt='%.18f',
        comments=''
    )
    print(f"Saved main simulation data to: {main_data_path}")

    # Save the final recommended time
    with open(os.path.join(storage_dir, f"simulation_info_Re{Re}.txt"), 'w') as f:
        f.write(f"Final simulation time: {t_sim[-1]:.6f}\n")
        f.write(f"Recommended end time (with 20% safety factor): {t_sim[-1] * 1.2:.6f}\n")
        f.write(f"Final kinetic energy: {Ek_store[-1]:.18f}\n")
        f.write(f"Maximum divergence: {np.max(div_mag):.6e}\n")
        f.write(f"Grid size: {nx}x{ny}\n")
        f.write(f"Time step: {dt:.6e}\n")
        f.write(f"Reynolds number: {Re}\n")

        # For Ru (u-velocity residual)
        max_Ru = np.max(Ru)
        Ru_indices = np.unravel_index(np.argmax(Ru), Ru.shape)
        f.write(f"Maximum Resu = {max_Ru:.6e} at position (i,j) = ({Ru_indices[1]}, {Ru_indices[0]})\n")
        
        # For Rv (v-velocity residual)
        max_Rv = np.max(Rv)
        Rv_indices = np.unravel_index(np.argmax(Rv), Rv.shape)
        f.write(f"Maximum Resv = {max_Rv:.6e} at position (i,j) = ({Rv_indices[1]}, {Rv_indices[0]})\n")
        



    #===ResidualsPlot (log-scale)===============================================================================================================
    plt.figure(figsize=(10, 6))

    # Use the same time points but skip first step (no residual yet)
    plot_times = t_sim[1:]

    plt.plot(plot_times, resNSu, 'r-', linewidth=2, label='u-velocity residual')
    plt.plot(plot_times, resNSv, 'b--', linewidth=2, label='v-velocity relError')
    
    plt.xlabel('Time')
    plt.ylabel('Absolute Residual')
    plt.title(f'Absolute Residual Evolution (Re={Re})')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    # Add reference lines for visualization
    for exponent in [-2, -4, -6, -8]:
        plt.axhline(y=10**exponent, color='k', linestyle=':', alpha=0.2)

    # Save the plot
    residual_path = os.path.join(script_dir, f'O8_Residuals_Re{Re}.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    print(f"RelError plot saved to: {residual_path}")
    plt.close()
    #plt.show()

    #-------------------------------------------------------------------------------------------------------------------
    storage_dir = os.path.join(script_dir, "Simulation_Data")
    os.makedirs(storage_dir, exist_ok=True)  # Create directory if it doesn't exist

    
    # Save each residual to its own file (only if we have data)
    if len(resNSu) > 0:
        save_residual_data(
            f"u_residual_Re{Re}.txt",
            t_sim[1:],  # Skip first time point since residuals start at second step
            resNSu,
            "u-velocity residual", storage_dir
        )

    if len(resNSv) > 0:
        save_residual_data(
            f"v_residual_Re{Re}.txt",
            t_sim[1:],  # Skip first time point since residuals start at second step
            resNSv,
            "v-velocity residual", storage_dir
        )


    #---MeanWallVelocityOverTime---------------------------------------------------------------------------------------
    # 1. Plot Mean Uwall Evolution
    plt.figure(figsize=(10,5))
    plt.plot(t_sim[1:], Uwall_mean_history, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Mean Wall Velocity')
    plt.title(f'Top Wall Velocity Evolution (Re={Re})')
    plt.grid(True)
    plt.tight_layout()

    Uwall_evo_path = os.path.join(script_dir, f'O9_UwallEvolution_Re{Re}.png')
    plt.savefig(Uwall_evo_path, dpi=300, bbox_inches='tight')
    print(f"Wall velocity evolution plot saved to: {Uwall_evo_path}")
    plt.close()

    # --- Vorticity Calculation and Plotting -------------------------------------------------------------------------
    # Calculate vorticity (dw/dx - du/dy) at cell centers
    
    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, omega, 20, cmap='jet')
    plt.colorbar(label='Vorticity')
    plt.title(f'Vorticity Field (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    # Save the plot
    vorticity_path = os.path.join(script_dir, f'O10_Vorticity_Re{Re}.png')
    plt.savefig(vorticity_path, dpi=300, bbox_inches='tight')
    print(f"Vorticity plot saved to: {vorticity_path}")
    plt.close()
    #plt.show()

    # --- Velocity Profiles ----------------------------------------------------------------------------------------
    # Find mid points (using integer division)
    mid_x_idx = nx // 2
    mid_y_idx = ny // 2

    mid_x = lx // 2 
    mid_y = ly // 2

    xd = np.linspace(0, lx, nx)
    yd = np.linspace(0, ly, ny+1)

    id_right = np.searchsorted(x_centers, mid_x)
    id_left = id_right - 1

    id_top = np.searchsorted(y_centers, mid_y)
    id_bottom = id_top - 1

    if nx % 2 == 0:
        even_idx = mid_x_idx + 1
        u_profile = u[1:-1, even_idx]
    else:
        u_profile = u_centers[:, mid_x_idx] 

    if ny % 2 == 0:
        even_idy = mid_y_idx + 1
        v_profile = v[even_idy, 1:-1] 
    else:
        v_profile = v_centers[mid_y_idx, :]
    


    # Create profile plots
    plt.figure(figsize=(12, 5))

    # U-velocity profile at mid-x (vertical line) - u is already at x faces
    plt.subplot(1, 2, 1)
    plt.plot(u_profile, y_centers, 'b-', linewidth=2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('u-velocity')
    plt.ylabel('y')
    plt.title(f'u-velocity profile at x={mid_x:.2f} (Re={Re})')
    plt.grid(True, linestyle='--', alpha=0.6)

    # V-velocity profile at mid-x (vertical line) - needs interpolation
    plt.subplot(1, 2, 2)
    # v is stored at [j, i] where j=1:ny+1, i=1:nx+1
    # For centerline (x=Lx/2), we need to average adjacent v points

    plt.plot(v_profile, x_centers, 'g-', linewidth=2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('v-velocity')
    plt.ylabel('y')
    plt.title(f'v-velocity profile at x={mid_x:.2f} (Re={Re})')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save the plot
    profiles_path = os.path.join(script_dir, f'O11_VelocityProfiles_Re{Re}.png')
    plt.savefig(profiles_path, dpi=300, bbox_inches='tight')
    print(f"Velocity profiles plot saved to: {profiles_path}")
    plt.close()
    #plt.show()

    # Also save the profile data to the storage directory
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Save u-profile data (vertical centerline)
    u_profile_path = os.path.join(storage_dir, f"u_profile_centerline_Re{Re}.txt")
    np.savetxt(
        u_profile_path,
        np.column_stack((y_centers, u_profile)),
        header=f"y\tu-velocity at x={mid_x:.8f}\nRe={Re}",
        delimiter='\t',
        fmt='%.12f'
    )
    print(f"Saved u-profile data to: {u_profile_path}")

    # Save v-profile data (vertical centerline)
    v_profile_path = os.path.join(storage_dir, f"v_profile_centerline_Re{Re}.txt")
    np.savetxt(
        v_profile_path,
        np.column_stack((x_centers, v_profile)),
        header=f"y\tv-velocity at x={mid_x:.8f}\nRe={Re}",
        delimiter='\t',
        fmt='%.12f'
    )
    print(f"Saved v-profile data to: {v_profile_path}")

    #---Comparison-(Re=100)----------------------------------------------------------------------------------------------
    # Compare with Ghia et al. (1982) benchmark data at Re=100
    if Re == 100:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.1015, -0.15662, 
                            -0.2109, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
                                0.78871, 0.84123, 1])
        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'Ou_Comparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()

        ghia_x = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                                0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])

        ghia_v = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 
                            0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
                                -0.07391, -0.05906, 0.00000])

        plt.figure(figsize=(8,6))
        plt.plot(x_centers, v_profile, 'g-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_x, ghia_v, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('x')
        plt.ylabel('v-velocity')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'Ov_Comparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()

    elif Re == 400:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
        
        ghia_u = np.array([0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, 
                            -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 
                                0.55892, 0.61756, 0.68439, 0.75837, 1.0000])
        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'OComparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()

    elif Re == 1000:
        # Ghia's u-velocity at vertical centerline (y-coordinates and values)
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.18109, -0.20196, -0.2222, -0.2973, -0.38289, -0.27805, 
                            -0.10648, -0.0608, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 
                                0.57492, 0.65928, 1])
        
        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'OComparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()
        #plt.show()
    elif Re == 3200:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323, -0.24427, 
                            -0.86636, -0.04272, 0.07156, 0.19791, 0.34682, 0.46101, 0.46547, 0.48296, 
                                0.53236, 1])
        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'OComparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()

    elif Re == 5000:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.41165, -0.42901, -0.43643, -0.40435, -0.3305, -0.22855, 
                            -0.07404, -0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 0.4612, 
                                0.48223, 1])

        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'OComparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()

    elif Re == 7500:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.43154, -0.4359, -0.43025, -0.38324, -0.32393, -0.23176, -0.07503,
                            -0.038, 0.08342, 0.20591, 0.34228, 0.47167, 0.47323, 0.47048, 0.47244, 1])

        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'OComparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()

    elif Re == 10000:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.42735, -0.42537, -0.41657, -0.38, -0.32709, -0.23186, 
                            -0.0754, 0.03111, 0.08344, 0.20673, 0.34635, 0.47804, 0.4807, 0.47783, 
                                0.47221, 1])

        plt.figure(figsize=(8,6))
        plt.plot(u_profile, y_centers, 'b-', linewidth=2, label='Current Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Centerline Velocity Comparison (Re={Re})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        benchmark_path = os.path.join(script_dir, f'OComparisin_Benchmark_Re{Re}.png')
        plt.savefig(benchmark_path, dpi=300)
        plt.close()


    # --- Shear Stress at Top Wall with Data Storage ---------------------------------------------------------------
    # Calculate wall shear stress (du/dy at top wall)
    du_dy_top = (u[-1,1:-1] - u[-2,1:-1])/dy  # Forward difference at top wall
    tau_top = nu * du_dy_top

    # Create and save the plot
    plt.figure(figsize=(8,4))
    plt.plot(x, tau_top, 'r-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Shear Stress')
    plt.title(f'Wall Shear Stress at Top Lid (Re={Re})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    shear_path = os.path.join(script_dir, f'O12_WallShearStress_Re{Re}.png')
    plt.savefig(shear_path, dpi=300, bbox_inches='tight')
    print(f"Shear stress plot saved to: {shear_path}")
    plt.close()

    # Prepare and save the shear stress data
    shear_data = np.column_stack((x, tau_top))

    # Create header with simulation parameters
    shear_header = (
        f"x [m]\tShear Stress [Pa]\n"
        f"Reynolds Number: {Re}\n"
        f"Grid Size: {nx}x{ny}\n"
        f"Time Step: {dt:.4e}\n"
        f"Kinematic Viscosity (nu): {nu:.4e}\n"
        f"Max Shear Stress: {np.max(tau_top):.6e} at x={x[np.argmax(tau_top)]:.4f}\n"
        f"Min Shear Stress: {np.min(tau_top):.6e} at x={x[np.argmin(tau_top)]:.4f}"
    )

    # Save to storage directory
    shear_filename = f"WallShearStress_Re{Re}.txt"
    shear_filepath = os.path.join(storage_dir, shear_filename)

    np.savetxt(
        shear_filepath,
        shear_data,
        header=shear_header,
        delimiter='\t',
        fmt='%.6e',
        comments='# '
    )
    print(f"Shear stress data saved to: {shear_filepath}")

    # Additional analysis: Save key statistics
    shear_stats = {
        'max_shear': np.max(tau_top),
        'min_shear': np.min(tau_top),
        'mean_shear': np.mean(tau_top),
        'x_max_shear': x[np.argmax(tau_top)],
        'x_min_shear': x[np.argmin(tau_top)],
        'reynolds': Re,
        'grid_size': f"{nx}x{ny}",
        'nu': nu
    }

    with open(os.path.join(storage_dir, f"ShearStressStats_Re{Re}.json"), 'w') as f:
        json.dump(shear_stats, f, indent=4)
    print(f"Shear stress statistics saved to: {os.path.join(storage_dir, f'ShearStressStats_Re{Re}.json')}")

    # Optional: Compare with theoretical prediction for fully developed flow
    if Re < 1000:  # Only for laminar regime
        theoretical_max = 2 * nu * Utop / ly  # Theoretical maximum for simple cases
        plt.figure(figsize=(8,4))
        plt.plot(x, tau_top, 'r-', label='Simulation')
        plt.axhline(y=theoretical_max, color='b', linestyle='--', label='Theoretical Max')
        plt.xlabel('x [m]')
        plt.ylabel('Shear Stress [Pa]')
        plt.title(f'Shear Stress Validation (Re={Re})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        validation_path = os.path.join(script_dir, f'O13_ShearValidation_Re{Re}.png')
        plt.savefig(validation_path, dpi=300, bbox_inches='tight')
        print(f"Shear validation plot saved to: {validation_path}")
        plt.close()

    #---STreamFuntionPLot------------------------------------------------------------------------------------------------
    # Calculate streamfunction
    psi = np.zeros((ny+2, nx+2))
    for j in range(1, ny+1):
        for i in range(1, nx+1):
            psi[j,i] = psi[j-1,i] - u[j,i]*dy  # Integration of u-velocity

    plt.figure(figsize=(6,6))
    plt.contour(xx, yy, psi[1:-1,1:-1], 20, colors='k')
    plt.contourf(xx, yy, psi[1:-1,1:-1], 20, cmap='viridis')
    plt.colorbar(label='Streamfunction')
    plt.title(f'Streamfunction Contours (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')

    stream_path = os.path.join(script_dir, f'O14_Streamfunction_Re{Re}.png')
    plt.savefig(stream_path, dpi=300)
    plt.close()
    #plt.show()

    # --- FFT Analysis with Data Storage --------------------------------------------------------------------------
    if len(Ek_store[1:]) > 10:
        dt_sampling = t_sim[1] - t_sim[0]  # Time between samples
        n = len(Ek_store[1:])
        fhat = np.fft.fft(Ek_store[1:] - np.mean(Ek_store[1:]))
        PSD = np.abs(fhat)**2 / n
        freq = (1/(dt_sampling*n)) * np.arange(n)
        L = np.arange(1, np.floor(n/2), dtype='int')

        # Create the plot
        plt.figure(figsize=(8,4))
        plt.loglog(freq[L], PSD[L], 'b-')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density')
        plt.title(f'Energy Spectrum Analysis (Re={Re})')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Save the plot
        spectrum_path = os.path.join(script_dir, f'O15_EnergySpectrum_Re{Re}.png')
        plt.savefig(spectrum_path, dpi=300, bbox_inches='tight')
        print(f"Energy spectrum plot saved to: {spectrum_path}")
        plt.close()

        # Prepare and save the FFT data
        fft_data = np.column_stack((freq[L], PSD[L]))
        
        # Create header with simulation parameters
        header = (
            f"Frequency [Hz]\tPower Spectral Density\n"
            f"Reynolds Number: {Re}\n"
            f"Grid Size: {nx}x{ny}\n"
            f"Time Step: {dt:.4e}\n"
            f"Sampling Interval: {dt_sampling:.4e}\n"
            f"Number of Samples: {n}\n"
            f"Mean Kinetic Energy: {np.mean(Ek_store):.4e}"
        )
        
        # Save to storage directory
        fft_filename = f"FFT_Analysis_Re{Re}.txt"
        fft_filepath = os.path.join(storage_dir, fft_filename)
        
        np.savetxt(
            fft_filepath,
            fft_data,
            header=header,
            delimiter='\t',
            fmt='%.6e',
            comments='# '
        )
        print(f"FFT data saved to: {fft_filepath}")

        # Additional diagnostic: Save dominant frequencies
        dominant_idx = np.argsort(PSD[L])[-3:]  # Top 3 dominant frequencies
        dominant_freq = freq[L][dominant_idx]
        dominant_amp = PSD[L][dominant_idx]
        
        with open(os.path.join(storage_dir, f"DominantFrequencies_Re{Re}.txt"), 'w') as f:
            f.write("# Dominant Frequencies Analysis\n")
            f.write(f"# Simulation Parameters (Re={Re}, nx={nx}, ny={ny}, dt={dt:.4e})\n\n")
            f.write("Rank\tFrequency [Hz]\tPower\n")
            for i, (freq, amp) in enumerate(zip(dominant_freq[::-1], dominant_amp[::-1])):
                f.write(f"{i+1}\t{freq:.6e}\t{amp:.6e}\n")
        
        print(f"Dominant frequencies saved to: {os.path.join(storage_dir, f'DominantFrequencies_Re{Re}.txt')}")
    else:
        print("Insufficient data points for FFT analysis (need >10 samples)")
        

    #---------------------------------------------------------------------------------------------------------------------
    # Save Ek and time data to a .txt file
    nsteps = len(Ek_store)
    step_numbers = np.arange(nsteps)  # Create array of time step numbers (0, 1, 2,...)

    # Combine step numbers, time, and Ek into three columns
    data = np.column_stack((step_numbers, t_sim, Ek_store))

    # Create filename with dt rounded to 2 decimal places
    data_filename = f"O4_Ek_Re{Re}_{nx}_{ny}_{dt:.4f}.txt"
    data_path = os.path.join(script_dir, data_filename)

    # Header for the file - updated to include step number
    header = f"Step\tTime (t)\tKinetic Energy (Ek)\nRe = {Re}, nx = {nx}, ny = {ny}, dt = {dt:.18f}"
        # Save using numpy (preferred for numerical data)
    # Save using numpy (preferred for numerical data)
    np.savetxt(
        data_path,
        data,
        header=header,
        delimiter='\t',
        fmt=['%d', '%.18f', '%.18f'],  # Different formats for step (integer) and others (float)
        comments=''  # Remove '#' from header
    )

    # Calculate recommended end time with 20% safety factor
    recommended_end_time = round(t_sim[-1] * 1.2)

    # Append the final Ek value and recommendation to the file
    with open(data_path, 'a') as f:  # 'a' mode for append
        f.write(f"\nEk_final = {Ek_store[-1]:.18f}")
        f.write(f"\nRecommended end time (with 20% safety factor): {recommended_end_time:.2f} (rounded from {t_sim[-1] * 1.2:.2f})")

    message = f'Post processing done at time: t = {t}'

    return message



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
    # This might be your original behavior for local testing
    output_base_dir = os.path.dirname(os.path.abspath(__file__))

# Modify where script_dir is defined globally or passed to postProcess
# If postProcess takes script_dir as an argument, you'll pass output_base_dir
# If script_dir is a global variable used by postProcess, then:
script_dir = output_base_dir # This will make all your outputs go to output_base_dir
print(f"Output files will be saved to: {script_dir}")

#=========================================================================================================================================
#script_dir = os.path.dirname(os.path.abspath(__file__))

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



def pressure_correction(u, v, Ap, p, dt, dx, dy, ml):
    div = np.zeros_like(p)
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1])/dx + (v[2:, 1:-1] - v[1:-1, 1:-1])/dy
    
    prhs = div / dt
    
    # Use AMG instead of SOR     p, S, Ap, nx, ny
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

def compute_explicit(u, v, dx, dy):
    RHSu_hat = np.zeros_like(u)
    RHSv_hat = np.zeros_like(v)
    
    # Compute u-component RHS
    for i in range(2, nx+1):
        for j in range(1, ny+1):
            ue = 0.5 * (u[j,i+1] + u[j,i])
            uw = 0.5 * (u[j,i] + u[j,i-1])
            un = 0.5 * (u[j+1,i] + u[j,i])
            us = 0.5 * (u[j,i] + u[j-1,i])

            vn = 0.5 * (v[j+1, i-1] + v[j+1, i])
            vs = 0.5 * (v[j, i-1] + v[j, i])

            RHSu_hat[j,i] = -(ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy
            
    # Compute v-component RHS
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            ve = 0.5 * (v[j,i+1] + v[j,i])
            vw = 0.5 * (v[j,i] + v[j,i-1])
            ue = 0.5 * (u[j,i+1] + u[j-1,i+1])
            uw = 0.5 * (u[j,i] + u[j-1,i])

            vn = 0.5 * (v[j+1, i] + v[j, i])
            vs = 0.5 * (v[j, i] + v[j-1, i])

            RHSv_hat[j,i] = -(ue*ve - uw*vw)/dx - (vn*vn - vs*vs)/dy
            
    return RHSu_hat, RHSv_hat

def compute_implicit(u, v, nu, dx, dy):
    RHSu = np.zeros_like(u)
    RHSv = np.zeros_like(v)
    
    # Compute u-component RHS
    for i in range(2, nx+1):
        for j in range(1, ny+1):
            RHSu[j,i] = nu * ((u[j,i-1] - 2.0 * u[j,i] + u[j,i+1])/dx/dx + 
                            (u[j-1,i] - 2.0 * u[j,i] + u[j+1,i])/dy/dy)
    
    # Compute v-component RHS
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            RHSv[j,i] = nu * ((v[j,i+1] - 2.0 * v[j,i] + v[j,i-1])/dx/dx + 
                            (v[j+1,i] - 2.0 * v[j,i] + v[j-1,i])/dy/dy)

    return RHSu, RHSv

def NS_operator(u, v, nu, dx, dy, dpdx, dpdy):
    diff_u = np.zeros_like(u)
    conv_u = np.zeros_like(u)

    diff_v = np.zeros_like(v)
    conv_v = np.zeros_like(v)


    RHS_u = np.zeros_like(u)
    RHS_v = np.zeros_like(v)

    NS_u = np.zeros_like(u)
    NS_v = np.zeros_like(v)
    NS_p = np.zeros_like(p)


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
    #print(f'NS_u(y,x)={np.shape(NS_u)}')
    return NS_u, NS_v



def generate_Lu(nx, ny, dx, dy, Utop):
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

def generate_Lv(nx, ny, dx, dy):
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
    

def do_1st_RK_Step(u, v, dx, dy, dt, nu, A, A_hat, Iu, Iv, Lu, Lv, Su, p, Utop, App):
    #Variable Initialization
    u0_int = u.copy()
    v0_int = v.copy()
    Hu0 = np.zeros_like(u0_int)
    Hv0 = np.zeros_like(v0_int)
    Ku_hat_int = np.zeros_like(u0_int)
    Kv_hat_int = np.zeros_like(v0_int)

    Uut = np.zeros_like(u)
    Uvt = np.zeros_like(v)
    
    p0 = p.copy()
    Ku_hat = np.zeros_like(u) #[ny+2, nx+2]
    Kv_hat = np.zeros_like(v) #[ny+2, nx+2]

    u0_int = u[1:-1, 2:-1]
    v0_int = v[2:-1, 1:-1]

    Ku_hat, Kv_hat = compute_explicit(u, v, dx, dy)

    Ku_hat_int = Ku_hat[1:-1, 2:-1] #[ny, nx-1]
    Kv_hat_int = Kv_hat[2:-1, 1:-1] #[ny-1, nx]

    Hu0 = u0_int + dt * A_hat[1,0] * Ku_hat_int #Only the interior points of u and Ku_hat [ny, nx-1]
    Hv0 = v0_int + dt * A_hat[1,0] * Kv_hat_int #Only the interior points of v and  Kv_hat [ny-1, nx]

    Mu = Iu - nu * dt * A[0,0] * Lu #[ny * (nx-1), ny * (nx-1)]
    Mv = Iv - nu * dt * A[0,0] * Lv #[(ny-1) * nx, (ny-1) * nx]

    b0_int = Hu0.ravel() + nu * dt * A[0,0] * Su

    Mpu = diags(1.0/Mu.diagonal())  # Jacobi preconditioner
    Uu_vec, info = cg(Mu, b0_int, x0=u0_int.ravel(), rtol=1e-06, M=Mpu)

    Mpv = diags(1.0/Mv.diagonal())  # Jacobi preconditioner
    Uv_vec, info = cg(Mv, Hv0.ravel(), x0=v0_int.ravel(), rtol=1e-06, M=Mpv)

    Uut_int = Uu_vec.reshape(u0_int.shape) #[ny, nx-1]
    Uvt_int = Uv_vec.reshape(v0_int.shape) #[ny-1, nx]

    Uut[1:-1, 2:-1] = Uut_int
    Uvt[2:-1, 1:-1] = Uvt_int
    
    Uut, Uvt = apply_BC(Uut, Uvt, Utop)

    #dt0 = (A[0,0]) * dt

    Uu, Uv, _, dpdx0, dpdy0, _ = pressure_correction(Uut, Uvt, App, p0, dt, dx, dy, ml)
    #print(f'dpdx0len = {np.shape(dpdx0)}')
    Uu, Uv = apply_BC(Uu, Uv, Utop)

    Ku, Kv = compute_implicit(Uu, Uv, nu, dx, dy)
    
    #Ku[1:-1, 2:-1] = Ku[1:-1, 2:-1] - dpdx0
    #Kv[2:-1, 1:-1] = Kv[2:-1, 1:-1] - dpdy0

    return Ku, Kv, Uu, Uv, Ku_hat, Kv_hat

#===Check Point setup===============================================================================================================
checkpoint_interval = 1000  # Save checkpoint every N steps
checkpoint_dir = os.path.join(script_dir, "checkpoints")
latest_checkpoint = None

#===Domain Setup====================================================================================================================
lx, ly = 1.0, 1.0
nx, ny = 16, 16
nelm = nx * ny
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
dx, dy = lx/nx, ly/ny
xx, yy = np.meshgrid(x, y)

print(f'dx = {dx}')

#===BC for Pressure=================================================================================================================
App, ml = generate_Ap(nx, ny, dx, dy)
print(f'ml = {ml}')

#===IMEX Paramethers================================================================================================================
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

dt = np.float64(t_end / nit)
print(f'dt = {dt}')
print(f'Nx=Ny = {nx}')

dt1 = 0.5/nu /(1.0/dx/dx + 1.0/dy/dy)
dt2 = 2.0 * nu / (Utop**2)

Lu, Su, _, _ = generate_Lu(nx, ny, dx, dy, Utop)
print(f'Lu = {Lu}')
Lv, _, _ = generate_Lv(nx, ny, dx, dy)
Iu = eye(ny * (nx-1))
Iv = eye((ny-1) * nx)
#print(f'Lu={Lu}')

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
s = A.shape[0]

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

nsave = 5 #Number of time steps to save

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


#---IMEX-------------------------------------------------------------------------------------------------------------------------------------------------
wall_times = [0.0]
cont_it = int(0)

start_time = time.time()
for cont_it in range(1, nit+1):
    
    u, v = apply_BC(u, v, Utop)
    K_hat = []
    U = []
    K = []
    
    p0 = p.copy()
    
    Ku0, Kv0, Uu0, Uv0, Ku_hat0, Kv_hat0 = do_1st_RK_Step(u, v, dx, dy, dt, nu, A, A_hat, Iu, Iv, Lu, Lv, Su, p0, Utop, App)

    K_hat.append((Ku_hat0, Kv_hat0))
    U.append((Uu0, Uv0))
    K.append((Ku0, Kv0))

    for i in range(1, s):
        pi = p.copy()
        Mui = Iu - nu * dt * A[i,i] * Lu
        Mvi = Iv - nu * dt * A[i,i] * Lv

        Kui_hat, Kvi_hat = compute_explicit(U[i-1][0], U[i-1][1], dx, dy)
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

        bi_int = Zui_int.ravel() + nu * dt * A[i,i] *Su

        ilui = spilu(Mui.tocsc())
        mui = LinearOperator(Mui.shape, ilui.solve)
        Uui_vec, flag1 = bicgstab(Mui, bi_int, x0=None, rtol=1e-06, atol=0.0, maxiter=None, M=mui, callback=None)

        ilvi = spilu(Mvi.tocsc())
        mvi = LinearOperator(Mvi.shape, ilvi.solve)
        Uvi_vec, flag2 = bicgstab(Mvi, Zvi_int.ravel(), x0=None, rtol=1e-06, atol=0.0, maxiter=None, M=mvi, callback=None)
  
        Uuit_int = Uui_vec.reshape(Zui_int.shape) #[ny, nx-1]
        Uvit_int = Uvi_vec.reshape(Zvi_int.shape) #[ny-1, nx]

        Uuit = np.zeros_like(u)
        Uvit = np.zeros_like(v)
        Uuit[1:-1, 2:-1] = Uuit_int
        Uvit[2:-1, 1:-1] = Uvit_int
        Uuit, Uvit = apply_BC(Uuit, Uvit, Utop)
        
        #dti = (A[i,i]) * dt

        Uui, Uvi, _, dpdxi, dpdyi, _ = pressure_correction(Uuit, Uvit, App, pi, dt, dx, dy, ml)

        Uui, Uvi = apply_BC(Uui, Uvi, Utop)
        U.append((Uui, Uvi))

        Kui, Kvi = compute_implicit(Uui, Uvi, nu, dx, dy)
    
        #Kui[1:-1, 2:-1] = Kui[1:-1, 2:-1] - dpdxi
        #Kvi[2:-1, 1:-1] = Kvi[2:-1, 1:-1] - dpdyi

        K.append((Kui, Kvi))

    #Compute the last K_hat
    Kus_hat, Kvs_hat = compute_explicit(U[-1][0], U[-1][1], dx, dy)
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

    u, v = apply_BC(u, v, Utop)
    u, v, p, dpdxn, dpdyn, Rp = pressure_correction(uhast, vhast, App, p.copy(), dt, dx, dy, ml)

    u, v = apply_BC(u, v, Utop)

    NS_u, NS_v = NS_operator(u_old, v_old, nu, dx, dy, dpdxn, dpdyn)

    #NS_u, NS_v = compute_implicit(u, v, nu, dx, dy)
    #NS_u[1:-1, 2:-1] = NS_u[1:-1, 2:-1] - dpdxn
    #NS_v[2:-1, 1:-1] = NS_v[2:-1, 1:-1] - dpdyn
    Ru = np.zeros([ny, nx+1])
    Ru = np.zeros([ny+1, nx])
    
    Ru = abs((u[1:-1, 2:-1]-u_old[1:-1, 2:-1])) / np.max(u_old[1:-1, 2:-1])
    Rv = abs((v[2:-1, 1:-1]-v_old[2:-1, 1:-1])) / np.max(v_old[2:-1, 1:-1])

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

    current_wall_time = time.time() - start_time
    wall_times.append(current_wall_time)

    t = np.float64(cont_it * dt) 
    t_sim.append(t)
#Perform th final Post Processing

end_time = time.time()
elapsed_time = end_time - start_time

simtime_filename = f"simulation_time_Re{Re}_{nx}_{ny}_{dt:.4f}.txt"
simtime_path = os.path.join(script_dir, simtime_filename)

# Write the elapsed time to this new file
with open(simtime_path, 'w') as f:
    f.write(f"Simulation elapsed time (seconds): {elapsed_time:.2f}\n")
    f.write(f"Re = {Re}, nx = {nx}, ny = {ny}, dt = {dt:.18f}\n")

print(f"Simulation time saved to {simtime_path}")
print(f"Total simulation time: {elapsed_time:.2f} seconds")

final_message = postProcess(u, v, dx, dy, nx, ny, lx, ly, x, y, xx, yy, t,  Re, nu, Utop, p, omega, Ek_store, residuals_u, residuals_v, residuals_p, residuals_Ek, t_sim, Uwall_mean_history, script_dir, resNSu,resNSv, resNSp, Ru, Rv, Rp)

plt.figure()
plt.plot(t_sim, wall_times, label="Wall time vs Simulation time")
plt.xlabel("Simulation time [s]")
plt.ylabel("Wall time [s]")
plt.title("Simulation Time vs Wall Clock Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f"wall_time_vs_sim_time_Re{Re}_{nx}_{ny}.png"), dpi=300)


# Save simulation time vs wall time vs Ek
sim_vs_wall_file = f"sim_vs_wall_Re{Re}_{nx}_{ny}_dt{dt:.4f}.txt"
sim_vs_wall_path = os.path.join(script_dir, sim_vs_wall_file)

with open(sim_vs_wall_path, 'w') as f:
    f.write("SimulationTime,WallTime,Ek\n")
    for sim_t, wall_t, ek in zip(t_sim, wall_times, Ek_store):
        f.write(f"{sim_t},{wall_t},{ek}\n")

print(f"Simulation time vs wall time and Ek saved to {sim_vs_wall_path}")
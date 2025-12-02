import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def postProcessing(u, v, p, nx, ny, lx, ly, dx, dy, dt, t, Re, frames_dir, frames_dir2, Ek_store, t_sim):
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

    # 2. Calculate velocity magnitude at cell centers (properly interpolated)
    u_centers = 0.5*(u[1:-1,1:-1] + u[1:-1,2:])  # Average u to cell centers
    v_centers = 0.5*(v[1:-1,1:-1] + v[2:,1:-1])  # Average v to cell centers
    velocity_magnitude = np.sqrt(u_centers**2 + v_centers**2)

    # ===== Plot 1: Velocity Magnitude =================================================================================
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
    cbar.set_label(r"$\left|\mathbf{V^{*}}\right|\,[-]$", fontsize=18)

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

    # Save velocity magnitude plot with descriptive name
    velocity_filename = os.path.join(frames_dir, f'O1_velMag_Re{Re}_t{t:.1f}.png')
    plt.savefig(velocity_filename, dpi=150)
    plt.close()

    # --- Absolute Divergence -----------------------------------------------------------------------------------------
    divu = np.zeros_like(p)
    divu[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx + (v[2:, 1:-1] - v[1:-1, 1:-1]) / dy
    div_mag = np.abs(divu[1:-1, 1:-1])

    # Compute max absolute divergence and its location
    max_val = np.max(div_mag)
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

    # Save divergence magnitude plot with descriptive name
    divergence_filename = os.path.join(frames_dir, f'O2_divV_Re{Re}_t{t:.1f}.png')
    plt.savefig(divergence_filename, dpi=150)
    plt.close()

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

    # Create profile plots with square subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # U-velocity profile at mid-x (vertical line) - u is already at x faces
    ax1.plot(u_profile, y_centers, 'b-', linewidth=2)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel(r'$u^*\,[-]$', fontsize=14)
    ax1.set_ylabel(r'$y^*\,[-]$', fontsize=14)
    ax1.set_title(rf'$u^*$ at $x^* = {mid_x:.2f}$ ($Re = {Re}$)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('auto')  # Square aspect ratio

    # V-velocity profile at mid-y (horizontal line)
    ax2.plot(v_profile, x_centers, 'g-', linewidth=2)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$v^*\,[-]$', fontsize=14)
    ax2.set_ylabel(r'$x^*\,[-]$', fontsize=14)
    ax2.set_title(rf'$v^*$ at $y^* = {mid_y:.2f}$ ($Re = {Re}$)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_aspect('auto')  # Square aspect ratio

    plt.tight_layout()

    

    # Also save the profile data to the storage directory
    if not os.path.exists(frames_dir2):
        os.makedirs(frames_dir2)

    # Save u-profile data (vertical centerline)
    u_profile_path = os.path.join(frames_dir2, f"u_profile_centerline_Re{Re}.txt")
    np.savetxt(
        u_profile_path,
        np.column_stack((y_centers, u_profile)),
        header=f"y\tu-velocity at x={mid_x:.8f}\nRe={Re}",
        delimiter='\t',
        fmt='%.12f'
    )
    print(f"Saved u-profile data to: {u_profile_path}")

    # Save v-profile data (vertical centerline)
    v_profile_path = os.path.join(frames_dir2, f"v_profile_centerline_Re{Re}.txt")
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
    elif Re == 400:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
        
        ghia_u = np.array([0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, 
                            -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 
                                0.55892, 0.61756, 0.68439, 0.75837, 1.0000])

    elif Re == 1000:
        # Ghia's u-velocity at vertical centerline (y-coordinates and values)
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.18109, -0.20196, -0.2222, -0.2973, -0.38289, -0.27805, 
                            -0.10648, -0.0608, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 
                                0.57492, 0.65928, 1])
        
    elif Re == 3200:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323, -0.24427, 
                            -0.86636, -0.04272, 0.07156, 0.19791, 0.34682, 0.46101, 0.46547, 0.48296, 
                                0.53236, 1])

    elif Re == 5000:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.41165, -0.42901, -0.43643, -0.40435, -0.3305, -0.22855, 
                            -0.07404, -0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 0.4612, 
                                0.48223, 1])

    elif Re == 7500:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.43154, -0.4359, -0.43025, -0.38324, -0.32393, -0.23176, -0.07503,
                            -0.038, 0.08342, 0.20591, 0.34228, 0.47167, 0.47323, 0.47048, 0.47244, 1])


    elif Re == 10000:
        ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                            0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

        ghia_u = np.array([0.0000, -0.42735, -0.42537, -0.41657, -0.38, -0.32709, -0.23186, 
                            -0.0754, 0.03111, 0.08344, 0.20673, 0.34635, 0.47804, 0.4807, 0.47783, 
                                0.47221, 1])
        
    
    ghia_x = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                            0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])

    ghia_v = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 
                        0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
                            -0.07391, -0.05906, 0.00000])

    fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
    ax.plot(u_profile, y_centers, 'b-', linewidth=2, label=rf'(${nx} \times {ny}$)')
    ax.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax.set_xlabel(r'$u^*\,[-]$', fontsize=14)
    ax.set_ylabel(r'$y^*\,[-]$', fontsize=14)
    ax.set_title(rf'$Re={Re}$', fontsize=14)
    ax.grid(True)
    ax.legend()
    ax.set_aspect('auto')  # Square aspect ratio
    # Add LaTeX formatting for axis numbers
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'$%g$'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$%g$'))
    plt.tight_layout()
    
    benchmark_path = os.path.join(frames_dir, f'O3_Comparisin_Benchmark_Re{Re}.png')
    plt.savefig(benchmark_path, dpi=300)
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
    ax.plot(x_centers, v_profile, 'g-', linewidth=2, label=rf'$\left({nx} \times {ny}\right)$')
    ax.plot(ghia_x, ghia_v, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax.set_xlabel(r'$x^*\,[-]$', fontsize=14)
    ax.set_ylabel(r'$v^*\,[-]$', fontsize=14)
    ax.set_title(rf'$Re={Re}$', fontsize=14)
    ax.grid(True)
    ax.legend()
    ax.set_aspect('auto')  # Square aspect ratio
    # Add LaTeX formatting for axis numbers
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'$%g$'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$%g$'))
    plt.tight_layout()
    
    benchmark_path = os.path.join(frames_dir, f'O4_Comparisin_Benchmark_Re{Re}.png')
    plt.savefig(benchmark_path, dpi=300)
    plt.close()
    # Continue with other Re values...

    # --- Kinetic Energy Over Time ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_sim, Ek_store, 'magenta', linewidth=2, label=rf'$({nx} \times {ny})$')
    ax.set_xlabel(r'$t^*\,[-]$', fontsize=14)
    ax.set_ylabel(r'$\overline{E}_k^*\,[-]$', fontsize=14)
    ax.set_title(rf'Mean Kinetic Energy Evolution ($Re = {Re}$)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    
    # Add LaTeX formatting for axis numbers
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'$%g$'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$%g$'))
    
    plt.tight_layout()

    # Save the plot to the frames directory
    plot_filename = f"O5_Ek_Re{Re}.png"
    plot_path = os.path.join(frames_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Kinetic energy plot saved to: {plot_path}")
    plt.close()

    print(f'Post Processing is Done! Files saved in: {frames_dir}')
    return
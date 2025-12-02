import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def postProcessing(u, v, p, nx, ny, lx, ly, dx, dy, dt, t, Re, frames_dir, frames_dir2):
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

    # Make the tick numbers bigger
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
    frame_filename = os.path.join(frames_dir, f'frame_{int(t / dt):08d}.png')
    plt.savefig(frame_filename, dpi=150)
    plt.close()

    # --- Absolute Divergence -----------------------------------------------------------------------------------------
    divu = np.zeros_like(p)
    divu[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx + (v[2:, 1:-1] - v[1:-1, 1:-1]) / dy
    div_mag = np.abs(divu[1:-1, 1:-1])

    # Compute max absolute divergence and its location
    max_val = np.max(div_mag)  # This is the ACTUAL maximum value (e.g., 93.3)
    #print(f'MAXVAL = {max_val}')
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
    frame_filename2 = os.path.join(frames_dir2, f'frame_{int(t / dt):08d}.png')
    plt.savefig(frame_filename2, dpi=150)
    plt.close()

    
    return print(f'Post Processing is Done !')
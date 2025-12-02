import numpy as np

def NS_operator(u, v, nu, dx, dy, nx, ny):
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


def compute_explicit(u, v, dx, dy, nx, ny):
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

def compute_implicit(u, v, nu, dx, dy, nx, ny):
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
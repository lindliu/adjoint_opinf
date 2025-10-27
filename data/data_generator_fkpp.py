#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:43:17 2025

@author: dliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve

# # Enable LaTeX rendering
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Computer Modern"]

# Parameters
Lx = 10.0        # Domain length in x-direction
Ly = 10.0        # Domain length in y-direction
Nx = 500         # Number of spatial points in x-direction
Ny = 500         # Number of spatial points in y-direction
dx = Lx / (Nx - 1)  # Spatial step in x
dy = Ly / (Ny - 1)  # Spatial step in y
D = 0.1          # Diffusion coefficient
r = 1.0          # Growth rate
dt = 0.0025       # Time step 0.005
T = 5.0          # Total simulation time
Nt = int(T / dt)+1 # Number of time steps

# Spatial grids
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition: 2D Gaussian at the center
u0 = np.exp(-10 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
u = u0.flatten()  # Flatten to 1D vector

# Function to construct 1D Laplacian matrix with Neumann BCs
def construct_1d_laplacian(N, d):
    main_diag = -2 * np.ones(N) / d**2
    super_diag = np.ones(N-1) / d**2
    sub_diag = np.ones(N-1) / d**2
    # Neumann BC adjustments
    super_diag[0] = 2 / d**2
    sub_diag[-1] = 2 / d**2
    return diags([sub_diag, main_diag, super_diag], [-1, 0, 1], format="csc")

# Construct 1D Laplacians
Lx = construct_1d_laplacian(Nx, dx)
Ly = construct_1d_laplacian(Ny, dy)

# Construct 2D Laplacian using Kronecker products
Ix = eye(Nx, format="csc")
Iy = eye(Ny, format="csc")
L = kron(Iy, Lx) + kron(Ly, Ix)

# Crank-Nicolson matrix
alpha = D * dt / 2
A = eye(Nx * Ny) - alpha * L
A = A.tocsc()

# Store solutions at certain time steps
snapshot_interval = 1 # Desired number of interval snapshot to store
snapshots = []

# Time stepping
for tt in range(1, Nt+1):
    u_prev = u.copy()
    reaction = r * u_prev * (1 - u_prev)
    b = u_prev + dt * reaction + alpha * L.dot(u_prev)
    u = spsolve(A, b)
    
    if tt % snapshot_interval == 0:
        snapshots.append(u.reshape(Ny, Nx))

# Convert snapshots to numpy array
snapshots = np.array(snapshots)

t = np.linspace(0, T, Nt)
# t = np.arange(0,T,dt)

print("Solution shape:", snapshots.shape)  # (100, 100, 1000)
print("time shape:", t.shape)
print("time:", t)
print("x space shape:", x.shape)
print("y space shape:", y.shape)

# Save results

np.save("./fkpp/total_fkpp_t.npy", t)
np.save('./fkpp/total_fkpp_x.npy', x[::4])
np.save('./fkpp/total_fkpp_y.npy', y[::4])
np.save('./fkpp/total_fkpp_1.npy', snapshots.T[::4,::4,:400])
np.save('./fkpp/total_fkpp_2.npy', snapshots.T[::4,::4,400:800])
np.save('./fkpp/total_fkpp_3.npy', snapshots.T[::4,::4,800:1200])
np.save('./fkpp/total_fkpp_4.npy', snapshots.T[::4,::4,1200:1600])
np.save('./fkpp/total_fkpp_5.npy', snapshots.T[::4,::4,1600:])

        
# np.savez("./total_fkpp.npz", Q=snapshots.T, t=t, x=x, y=y)







import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# Load the stored solution and spatial grid
t = np.load("./fkpp/total_fkpp_t.npy")
x = np.load('./fkpp/total_fkpp_x.npy')
y = np.load('./fkpp/total_fkpp_y.npy')
data_files = [glob.glob(os.path.join(os.getcwd(), f'./fkpp/total_fkpp_{i}.npy'))[0] for i in range(1,6)]
snapshots = [np.load(data_files[i]) for i in range(5)]
snapshots = np.concatenate(snapshots,axis=2)


U = snapshots

# Plot snapshots
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

axes[0].contourf(x, y, U[:,:,0], levels=20, cmap='viridis')
axes[0].set_title(f't = {t[0]:.2f}')
axes[0].axis('scaled')

axes[1].contourf(x, y, U[:,:,500], levels=20, cmap='viridis')
axes[1].set_title(f't = {t[500]:.2f}')
axes[1].axis('scaled')

axes[2].contourf(x, y, U[:,:,1000], levels=20, cmap='viridis')
axes[2].set_title(f't = {t[1000]:.2f}')
axes[2].axis('scaled')

axes[3].contourf(x, y, U[:,:,1500], levels=20, cmap='viridis')
axes[3].set_title(f't = {t[1500]:.2f}')
axes[3].axis('scaled')

axes[4].contourf(x, y, U[:,:,2000], levels=20, cmap='viridis')
axes[4].set_title(f't = {t[2000]:.2f}')
axes[4].axis('scaled')

plt.savefig('./fkpp/fkpp_contour.png')
plt.show()




# Define the mesh grid for plotting
X, Y = np.meshgrid(x, y, indexing='ij')
Nt = t.shape[0]

# Function to plot the 3D surface of the solution at a given time step
def plot_3d_surface(U_snapshot, time_step, ax):
    surf = ax.plot_surface(X, Y, U_snapshot, cmap='plasma', edgecolor='none')
    ax.set_title(r"$t = {:.1f}$".format(t[time_step]), fontsize=30)
    ax.set_xlabel(r"$x$", fontsize=25, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"$y$", fontsize=25, fontweight='bold', labelpad=10)
    ax.set_zlabel(r"$q(x,y,t)$",fontsize=25, fontweight='bold', labelpad=10)
    # Tick parameters for axes
    ax.tick_params(axis='both', which='major', labelsize=22, length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=22, length=4, width=1.5)
    ax.view_init(elev=30, azim=135)  # Adjust view angle for better visualization
    return surf

# Create figure for 3D plots with constrained layout for better positioning
fig = plt.figure(figsize=(20, 6), constrained_layout=True)

# Plot initial condition (t = 0)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf1 = plot_3d_surface(U[:, :, 0], 0, ax1)

# Plot intermediate time step (t = 2.5)
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf2 = plot_3d_surface(U[:, :, Nt//2], Nt//2, ax2)

# Plot final time step (t = T-dt)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf3 = plot_3d_surface(U[:, :, -1], -1, ax3)

# Set global title with LaTeX and manually adjust vertical position
#fig.suptitle(r"Fisher-KPP Solution", fontsize=16, y=1.0001)  

plt.tight_layout(rect=[0, 0, 0.95, 1])

plt.savefig('./fkpp/fkpp_3d.png', bbox_inches='tight')
# Show plot
plt.show()


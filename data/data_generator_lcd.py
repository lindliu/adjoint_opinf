#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:43:17 2025

@author: dliu
"""

# linear convection–diffusion equation
# Hybrid numerical methods for convection–diffusion problems in arbitrary geometries
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt

# ---- 参数定义 ----
Nx, Ny = 201, 201           # 网格点数
Lx, Ly = 2.0, 2.0           # 域大小 [-1,1]x[-1,1]
nu = 5e-3                   # 扩散系数
omega = np.pi               # 旋转角速度
dt = 1e-3                   # 时间步长
t_end = 0.5                 # 终止时间
sigma = 0.1
x0, y0 = -.5, -.5

assert dt<(Lx/Nx)**2/(4*nu)
assert dt<(Lx/Nx)/Lx

# ---- 网格与初值 ----
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij")
u = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2))

# 周期边界辅助函数
def periodic(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]
    return arr

## u_t + (vx*u_x + vy*u_y) + nu (u_xx+u_yy) = 0
U = []
# ---- 时间推进 ----
t = np.arange(0, t_end+dt, dt)
nt = t.shape[0]
for n in range(nt):
    # 周期边界
    u = periodic(u)

    # 对流速度场
    vx = 1 #-omega * Y
    vy =  1.5 #omega * X

    # 空间导数 (中心差分)
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)  ## 二阶中心差分
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
    lap  = ((np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2 +
            (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2)

    # 显式更新
    u = u + dt * (-vx*dudx - vy*dudy + nu*lap)
    
    U.append(u)

U = np.array(U)

np.save("./lcd/total_lcd_t.npy", t)
np.save('./lcd/total_lcd_x.npy', x[::2])
np.save('./lcd/total_lcd_y.npy', y[::2])
np.save('./lcd/total_lcd_1.npy', U.T[::2,::2,:400])
np.save('./lcd/total_lcd_2.npy', U.T[::2,::2,400:800])
np.save('./lcd/total_lcd_3.npy', U.T[::2,::2,800:1200])
np.save('./lcd/total_lcd_4.npy', U.T[::2,::2,1200:1600])
np.save('./lcd/total_lcd_5.npy', U.T[::2,::2,1600:])







import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# Load the stored solution and spatial grid
t = np.load("./lcd/total_lcd_t.npy")
x = np.load('./lcd/total_lcd_x.npy')
y = np.load('./lcd/total_lcd_y.npy')
nt = t.shape[0]

data_files = [glob.glob(os.path.join(os.getcwd(), f'./lcd/total_lcd_{i}.npy'))[0] for i in range(1,6)]
U = [np.load(data_files[i]) for i in range(5)]
U = np.concatenate(U, axis=2)
U = np.transpose(U, axes=(2,0,1))



import matplotlib.ticker as tkr
import matplotlib
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

# U = U.T
X, Y = np.meshgrid(x, y, indexing="ij")
step = nt//4
idxs = [0, step, 2*step, 3*step, 4*step]


# Common levels (or use vmin/vmax)
vmin = np.min([U[i].min() for i in idxs])
vmax = np.max([U[i].max() for i in idxs])
levels = np.linspace(vmin, vmax, 21)

fig, axes = plt.subplots(1, 5, sharey=True, figsize=(15, 5))

mappable = None
for ax, i in zip(axes, idxs):
    im = ax.contourf(x, y, U[i], levels=levels, cmap='plasma', extend='both')
    ax.set_title(f't = {t[i]:.2f}', fontsize=18)
    ax.set_xlabel('x', fontsize=15)
    if i==0:
        ax.set_ylabel('y', fontsize=15)
    ax.set_aspect('equal', adjustable='box')  # same as axis('scaled')
    mappable = im  # keep last (all share same levels/cmap)

# fig.suptitle('Fisher-KPP')

# Leave room on the right for a single colorbar (won't shrink subplots)
fig.subplots_adjust(right=0.9, wspace=0.1)
cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.5])  # [left, bottom, width, height] in fig coords
cbar = fig.colorbar(mappable, cax=cbar_ax,  format=tkr.FormatStrFormatter('%.1f'))
# cbar.ax.set_ylabel('u', rotation=90, va='center')

# plt.tight_layout()
plt.savefig('./lcd/lcd_2d.png', dpi=650, bbox_inches='tight')
plt.show()








# # 可视化（每隔一定步数显示
# # U = U.T
# X, Y = np.meshgrid(x, y, indexing="ij")
# step = nt//4

# fig, axes = plt.subplots(1,5,figsize=[15,3])
# axes[0].pcolormesh(X, Y, U[0], shading='auto', cmap='turbo')
# axes[0].set_title(f"t = {0*dt:.2f}")
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('y')
# # axes[0].legend()

# axes[1].pcolormesh(X, Y, U[step], shading='auto', cmap='turbo')
# axes[1].set_title(f"t = {step*dt:.2f}")
# axes[1].set_xlabel('x')
# axes[1].set_ylabel('y')
# # axes[1].legend()

# axes[2].pcolormesh(X, Y, U[2*step], shading='auto', cmap='turbo')
# axes[2].set_title(f"t = {2*step*dt:.2f}")
# axes[2].set_xlabel('x')
# axes[2].set_ylabel('y')
# # axes[2].legend()

# axes[3].pcolormesh(X, Y, U[3*step], shading='auto', cmap='turbo')
# axes[3].set_title(f"t = {3*step*dt:.2f}")
# axes[3].set_xlabel('x')
# axes[3].set_ylabel('y')
# # axes[3].legend()

# axes[4].pcolormesh(X, Y, U[4*step], shading='auto', cmap='turbo')
# axes[4].set_title(f"t = {4*step*dt:.2f}")
# axes[4].set_xlabel('x')
# axes[4].set_ylabel('y')
# # axes[4].legend()

# fig.savefig('./lcd/lcd_2d.png')














# Define the mesh grid for plotting

# Load the stored solution and spatial grid
t = np.load("./lcd/total_lcd_t.npy")
x = np.load('./lcd/total_lcd_x.npy')
y = np.load('./lcd/total_lcd_y.npy')
data_files = [glob.glob(os.path.join(os.getcwd(), f'./lcd/total_lcd_{i}.npy'))[0] for i in range(1,6)]
U = [np.load(data_files[i]) for i in range(5)]
U = np.concatenate(U, axis=2)

U = np.transpose(U, (1,0,2))
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
fig = plt.figure(figsize=(25, 6), constrained_layout=True)
# fig, axes = plt.subplots(1,3, figsize=[20,6], constrained_layout=True)

# Plot initial condition (t = 0)
ax1 = fig.add_subplot(1, 4, 1, projection='3d')
surf1 = plot_3d_surface(U[:, :, 0], 0, ax1)

# Plot intermediate time step (t = 2.5)
ax2 = fig.add_subplot(1, 4, 2, projection='3d')
surf2 = plot_3d_surface(U[:, :, Nt//2], Nt//2, ax2)

# Plot final time step (t = T-dt)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
surf3 = plot_3d_surface(U[:, :, -1], -1, ax3)

# ax3 = fig.add_subplot(1, 4, 4, projection='3d')
# surf3 = plot_3d_surface(U[:, :, -1], -1, ax3)
# # Set global title with LaTeX and manually adjust vertical position

# fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.06, hspace=0.06)
plt.savefig('./lcd/lcd_3d.png', bbox_inches='tight',
            bbox_extra_artists=[ax3.zaxis.label])
# Show plot
plt.show()


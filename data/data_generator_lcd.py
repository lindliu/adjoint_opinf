#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:43:17 2025

@author: dliu
"""

# linear convection–diffusion equation
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
x0, y0 = 0.5, 0.0

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


U = []
# ---- 时间推进 ----
t = np.arange(0, t_end+dt, dt)
nt = t.shape[0]
for n in range(nt):
    # 周期边界
    u = periodic(u)

    # 对流速度场
    vx = -omega * Y
    vy =  omega * X

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
data_files = [glob.glob(os.path.join(os.getcwd(), f'./lcd/total_lcd_{i}.npy'))[0] for i in range(1,6)]
U = [np.load(data_files[i]) for i in range(5)]
U = np.concatenate(U, axis=2)


# 可视化（每隔一定步数显示
U = U.T
X, Y = np.meshgrid(x, y, indexing="ij")
step = nt//4

fig, axes = plt.subplots(1,5,figsize=[15,3])
axes[0].pcolormesh(X, Y, U[0], shading='auto', cmap='turbo')
axes[0].set_title(f"t = {0*dt:.2f}")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
# axes[0].legend()

axes[1].pcolormesh(X, Y, U[step], shading='auto', cmap='turbo')
axes[1].set_title(f"t = {step*dt:.2f}")
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
# axes[1].legend()

axes[2].pcolormesh(X, Y, U[2*step], shading='auto', cmap='turbo')
axes[2].set_title(f"t = {2*step*dt:.2f}")
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
# axes[2].legend()

axes[3].pcolormesh(X, Y, U[3*step], shading='auto', cmap='turbo')
axes[3].set_title(f"t = {3*step*dt:.2f}")
axes[3].set_xlabel('x')
axes[3].set_ylabel('y')
# axes[3].legend()

axes[4].pcolormesh(X, Y, U[4*step], shading='auto', cmap='turbo')
axes[4].set_title(f"t = {4*step*dt:.2f}")
axes[4].set_xlabel('x')
axes[4].set_ylabel('y')
# axes[4].legend()

fig.savefig('./lcd/2d_lcd.png')
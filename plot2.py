#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:46:02 2025

@author: dliu
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import opinf
from train_adjoint import data_loader
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from utils import model_reducer


# import matplotlib
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20)


data_name = 'burgers'
split_ratio = .5
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
split_ratio_validation = .1
step = 10
r = 5
num_samples = 10000//step



def relative_error(A,B):
    error = np.mean((A-B)**2)/np.mean(B**2)
    return error**.5


# Heat map
fig, axes = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(16, 10))
pad = 5 # in points

cols = ['FOM', 'POD recon.', 'OpInf-ord2', 'OpInf-ord6', 'Adjoint']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=22, ha='center', fontweight='bold', va='baseline')

rows = ['NL = 0%', 'NL = 80%', 'NL = 160%']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=22,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向



noise_level = 0
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

assert (data['Q_train_']==Q_train_).all()
assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 



error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')


Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 0

cc = 0
# True state evolution
im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
axes[rr,cc].set_ylabel(r"$x$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)


cc = 1
# True state evolution
im0 = axes[rr,cc].imshow(Q_test_re, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)



cc = 3
# Predicted state evolution
im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)



cc = 4
# Predicted state evolution
im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)
axes[rr,cc].set_xticks([0,.5,1.], [0.6,0.8,1.0])


divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)
        
# cbar = fig.colorbar(im3, ax=axes[rr,cc], fraction=0.025, pad=.02, shrink=1, aspect=40)
# cbar.ax.tick_params(labelsize=13)

# # 每列加一个 colorbar
# cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
#                     fraction=0.02, pad=0.04)
# cbar.ax.tick_params(labelsize=13)
# cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)











noise_level = 80
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

assert (data['Q_train_']==Q_train_).all()
assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 


error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e}, \
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')


Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 1

cc = 0
# True state evolution
im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
axes[rr,cc].set_ylabel(r"$x$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)



cc = 1
# True state evolution
im0 = axes[rr,cc].imshow(Q_test_re, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 3
# Predicted state evolution
im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 4
# Predicted state evolution
im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)
axes[rr,cc].set_xticks([0,.5,1.], [0.6,0.8,1.0])


divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)
        
# cbar = fig.colorbar(im3, ax=axes[rr,cc], fraction=0.025, pad=.02, shrink=1, aspect=40)
# cbar.ax.tick_params(labelsize=13)

# # 每列加一个 colorbar
# cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
#                     fraction=0.02, pad=0.04)
# cbar.ax.tick_params(labelsize=13)
# cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)











noise_level = 160
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

assert (data['Q_train_']==Q_train_).all()
assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 


error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e}, \
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')



Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 2

cc = 0
# True state evolution
im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
axes[rr,cc].set_ylabel(r"$x$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)





cc = 1
# True state evolution
im0 = axes[rr,cc].imshow(Q_test_re, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)



# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)



cc = 3
# Predicted state evolution
im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)



cc = 4
# Predicted state evolution
im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)
axes[rr,cc].set_xticks([0,.5,1.], [0.6,0.8,1.0])

divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)


# axes[rr, cc].text(r'$t$', transform=axes[rr,cc].transAxes)
# cbar = fig.colorbar(im3, ax=axes[rr,cc], fraction=0.025, pad=.02, shrink=1, aspect=40)
# cbar.ax.tick_params(labelsize=13)

# # 每列加一个 colorbar
# cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
#                     fraction=0.02, pad=0.04)
# cbar.ax.tick_params(labelsize=13)
# cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


for cc in range(5):
    axes[-1, cc].text(
        0.5, -0.32,
        r'$t$',
        transform=axes[-1, cc].transAxes,
        ha='center',
        va='top',
        fontsize=20,
        fontweight='bold'
    )



plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Illustrative Predition of Evolution under Different Noise Levels (Burgers' Equation)",
#              fontsize=23, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig(f'./figures_analysis/plot_examples_evolution_burgers_sample{num_samples}.png',bbox_inches='tight',dpi=300)
plt.show()


























only_end=True



data_name = 'fkpp'  ##  
split_ratio = .75
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
split_ratio_validation = .1
step = 2
r = 5
num_samples = 2001//step




# Heat map
fig, axes = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(16, 10))
pad = 5 # in points

cols = ['FOM', 'POD recon.', 'OpInf-ord2', 'OpInf-ord6', 'Adjoint']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=22, ha='center', fontweight='bold', va='baseline')

rows = ['NL = 0%', 'NL = 80%', 'NL = 160%']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=22,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向

for ax in axes.ravel():
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])

noise_level = 0
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

assert (data['Q_train_']==Q_train_).all()
assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 


error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')
      
      
Q = np.c_[Q_train,Q_valid,Q_original_test]



vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 0


cc = 0
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_original_test.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    axes[rr,cc].set_ylabel(r"$y$", fontsize=20, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)



cc = 1
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_test_re.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_ylabel(r"$y$", fontsize=15, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
if only_end:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_ylabel(r"$y$", fontsize=15, fontweight='bold')
else:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 3
# Predicted state evolution
if only_end:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_ylabel(r"$y$", fontsize=15, fontweight='bold')
else:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 4
# Predicted state evolution
if only_end:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$x$", fontsize=15, fontweight='bold')
else:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)



divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)

# cbar = fig.colorbar(im3, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # 每列加一个 colorbar
# cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
#                     fraction=0.02, pad=0.04)
# cbar.ax.tick_params(labelsize=13)
# cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)










noise_level = 80
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

assert (data['Q_train_']==Q_train_).all()
assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 


error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')
      

Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 1

cc = 0
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_original_test.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
axes[rr,cc].set_ylabel(r"$y$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)





cc = 1
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_test_re.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
if only_end:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
else:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 3
# Predicted state evolution
if only_end:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
else:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 4
# Predicted state evolution
if only_end:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$x$", fontsize=15, fontweight='bold')
else:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)


divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)
# cbar = fig.colorbar(im3, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # # 每列加一个 colorbar
# # cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
# #                     fraction=0.02, pad=0.04)
# # cbar.ax.tick_params(labelsize=13)
# # cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)










noise_level = 160
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

assert (data['Q_train_']==Q_train_).all()
assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 


error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')


Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 2

cc = 0
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_original_test.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
axes[rr,cc].set_ylabel(r"$y$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)




cc = 1
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_test_re.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
if only_end:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 3
# Predicted state evolution
if only_end:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)

cc = 4
# Predicted state evolution
if only_end:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint.reshape([125,125,151])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 10, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)


# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)


divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)

# cbar = fig.colorbar(im3, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # # 每列加一个 colorbar
# # cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
# #                     fraction=0.02, pad=0.04)
# # cbar.ax.tick_params(labelsize=13)
# # cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


for cc in range(5):
    axes[-1, cc].text(
        0.5, -0.32,
        r'$x$',
        transform=axes[-1, cc].transAxes,
        ha='center',
        va='top',
        fontsize=20,
        fontweight='bold'
    )



plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Illustrative Predition of Evolution under Different Noise Levels (FKPP Equation)",
#              fontsize=23, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig(f'./figures_analysis/plot_examples_evolution_FKPP_sample{num_samples}_end{only_end}.png',bbox_inches='tight',dpi=300)
plt.show()




















only_end=True



data_name = 'lcd'  ##  
split_ratio = .75
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
split_ratio_validation = .1
step = 2
r = 15
num_samples = 2001//step




# Heat map
fig, axes = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(16, 10))
pad = 5 # in points

cols = ['FOM', 'POD recon.', 'OpInf-ord2', 'OpInf-ord6', 'Adjoint']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=22, ha='center', fontweight='bold', va='baseline')

rows = ['NL = 0%', 'NL = 80%', 'NL = 160%']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=22,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向
    

noise_level = 0
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

# assert (data['Q_train_']==Q_train_).all()
# assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 


error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')


Q = np.c_[Q_train,Q_valid,Q_original_test]



vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 0


cc = 0
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_original_test.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    axes[rr,cc].set_ylabel(r"$y$", fontsize=20, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)



cc = 1
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_test_re.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    # axes[rr,cc].set_ylabel(r"$y$", fontsize=15, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
if only_end:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    # axes[rr,cc].set_ylabel(r"$y$", fontsize=15, fontweight='bold')
else:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)

cc = 3
# Predicted state evolution
if only_end:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    # axes[rr,cc].set_ylabel(r"$y$", fontsize=15, fontweight='bold')
else:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 4
# Predicted state evolution
if only_end:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$x$", fontsize=15, fontweight='bold')
else:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)



divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)

# cbar = fig.colorbar(im3, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # 每列加一个 colorbar
# cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
#                     fraction=0.02, pad=0.04)
# cbar.ax.tick_params(labelsize=13)
# cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)










noise_level = 80
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

# assert (data['Q_train_']==Q_train_).all()
# assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 

error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')

Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 1

cc = 0
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_original_test.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
axes[rr,cc].set_ylabel(r"$y$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)





cc = 1
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_test_re.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
if only_end:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
else:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 3
# Predicted state evolution
if only_end:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
else:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 4
# Predicted state evolution
if only_end:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$x$", fontsize=15, fontweight='bold')
else:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
    # axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)


divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)
# cbar = fig.colorbar(im3, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # # 每列加一个 colorbar
# # cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
# #                     fraction=0.02, pad=0.04)
# # cbar.ax.tick_params(labelsize=13)
# # cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)











noise_level = 160
   
### get data ###
Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, _ = \
                                    data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

Vr = opinf.basis.PODBasis(num_vectors=r)
# Fit the basis (compute Vr) using the snapshot data.
Vr.fit(Q_train)
# Compress the state snapshots to the reduced space defined by the basis.
Q_train_ = Vr.compress(Q_train)
Q_valid_ = Vr.compress(Q_valid)
Q_test_ = Vr.compress(Q_test)


name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])

# assert (data['Q_train_']==Q_train_).all()
# assert (data['Q_valid_']==Q_valid_).all()

# plt.plot(Q_train_.T)
# plt.plot(data['Q_train_'].T)


Q_test_re = Vr.decompress(Q_test_) 
Q_pred_opinf2 = Vr.decompress(data['Q_opinf_2']) 
Q_pred_opinf6 = Vr.decompress(data['Q_opinf_6']) 
Q_pred_adjoint = Vr.decompress(data['Q_adjoint']) 

error_to_FOM = [relative_error(Q_test_re, Q_original_test),
                relative_error(Q_pred_opinf2, Q_original_test),
                relative_error(Q_pred_opinf6, Q_original_test),
                relative_error(Q_pred_adjoint, Q_original_test)]
print(f'noise level: {noise_level}')
print(f'rec. to true: {error_to_FOM[0]:.4e},\
      opinf2 to true: {error_to_FOM[1]:.4e},\
      opinf6 to true: {error_to_FOM[2]:.4e},\
      adjoint to true: {error_to_FOM[3]:.4e}')

Q = np.c_[Q_train,Q_valid,Q_original_test]


    
vmin = min(Q_original_test.min(), Q_pred_opinf2.min(),
           Q_pred_opinf6.min(), Q_pred_adjoint.min())
vmax = max(Q_original_test.max(), Q_pred_opinf2.max(),
           Q_pred_opinf6.max(), Q_pred_adjoint.max())
rr = 2

cc = 0
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_original_test.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
axes[rr,cc].set_ylabel(r"$y$", fontsize=20, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)




cc = 1
# True state evolution
if only_end:
    im0 = axes[rr,cc].imshow(Q_test_re.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im0 = axes[rr,cc].imshow(Q_original_test, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im0, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[0]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


# np.c_[Q_train,Q_valid,Q_pred_adjoint]
cc = 2
# Predicted state evolution
if only_end:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im1 = axes[rr,cc].imshow(Q_pred_opinf2, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im1, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # plt.suptitle("Space-Time Evolution of Burgers' Equation")
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[1]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)



cc = 3
# Predicted state evolution
if only_end:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im2 = axes[rr,cc].imshow(Q_pred_opinf6, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
# axes[rr,cc].set_xlabel(r"$t$", fontsize=15, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# cbar = fig.colorbar(im2, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[2]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


cc = 4
# Predicted state evolution
if only_end:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint.reshape([101,101,38])[:,:,-1], vmin=vmin, vmax=vmax, aspect='auto', extent=[-1,1,-1,1], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$x$", fontsize=20, fontweight='bold')
else:
    im3 = axes[rr,cc].imshow(Q_pred_adjoint, vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 5, 0, 10], origin='lower', cmap='plasma')
    axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$x$", fontsize=15, fontweight='bold')

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_linewidth(1.7)  
axes[rr,cc].spines['right'].set_linewidth(1.7)

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)


divider = make_axes_locatable(axes[rr, cc])
cax = divider.append_axes("right", size="3%", pad=0.06)  # size 改小=更细
cbar = fig.colorbar(im3, cax=cax)
cbar.ax.tick_params(labelsize=15)

# cbar = fig.colorbar(im3, ax=axes[rr,cc])
# cbar.ax.tick_params(labelsize=13)
# # # 每列加一个 colorbar
# # cbar = fig.colorbar(im3, ax=axes[:, cc], orientation='vertical',
# #                     fraction=0.02, pad=0.04)
# # cbar.ax.tick_params(labelsize=13)
# # cbar.set_label("Value", fontsize=15, fontweight='bold')
axes[rr, cc].set_xlabel(
    rf'RSE = {error_to_FOM[3]:.2e}',
    fontsize=15,
    fontweight='bold',
    labelpad=5
)


for cc in range(5):
    axes[-1, cc].text(
        0.5, -0.32,
        r'$x$',
        transform=axes[-1, cc].transAxes,
        ha='center',
        va='top',
        fontsize=20,
        fontweight='bold'
    )





plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Illustrative Predition of Evolution under Different Noise Levels (CDE)",
#              fontsize=23, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig(f'./figures_analysis/plot_examples_evolution_ADE_sample{num_samples}_end{only_end}.png',bbox_inches='tight',dpi=300)
plt.show()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:43:17 2025

@author: dliu
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from train_adjoint import data_loader


import random
random.seed(10)
np.random.seed(10)    # for numpy random

def get_min_from_two_list(error_train_1, error_train_2, error_valid_1, error_valid_2):
    assert error_valid_1.shape == error_valid_2.shape
    
    error = np.zeros_like(error_valid_1)
    mask = np.argmin(np.c_[error_valid_1, error_valid_2],axis=1)

    error[mask==0] = error_train_1[mask==0]
    error[mask==1] = error_train_2[mask==1]
    
    return error

def get_errors(data_name, num_samples, noise_level, ratio, max_iter=10, smoother=True):
    opinf_use_val = True
    name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
    path = glob.glob(f'./results/error_{name_suffix}*')
    assert len(path)==1
    error_true = np.load(path[0])
    # error = np.load(path[0])

    opinf_use_val = False
    name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
    path = glob.glob(f'./results/error_{name_suffix}*')
    assert len(path)==1
    error_false = np.load(path[0])

    error = dict()
    error['error_opinf_2_list'] = get_min_from_two_list(error_true['error_opinf_2_list'], error_false['error_opinf_2_list'], \
                                                        error_true['error_opinf_2_valid_list'], error_false['error_opinf_2_valid_list'])
    error['error_opinf_6_list'] = get_min_from_two_list(error_true['error_opinf_6_list'], error_false['error_opinf_6_list'], \
                                                        error_true['error_opinf_6_valid_list'], error_false['error_opinf_6_valid_list'])
    error['error_adjoint_list'] = get_min_from_two_list(error_true['error_adjoint_list'], error_false['error_adjoint_list'], \
                                                        error_true['error_adjoint_valid_list'], error_false['error_adjoint_valid_list'])

    return error

data_name = 'burgers'
split_ratio = .5
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10

############ PLOTS errors #################
fig, axes = plt.subplots(4, 6, sharex=True, figsize=[15,10])
pad = 5 # in points

# cols = ['Noise Level: 0', 'Noise Level: 40', 'Noise Level: 80', 'Noise Level: 120', 'Noise Level: 200']
# for ax, col in zip(axes[0], cols):
#     ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
#                 xycoords='axes fraction', textcoords='offset points',
#                 size='large', ha='center', va='baseline')

rows = ['Samples: 20', 'Samples: 100', 'Samples: 1000', 'Samples: 10000']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size='x-large',
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向
    

num_samples = 100
noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
rr, cc = 0,0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_title(f'Noise Level: {noise_level}%', fontweight='bold', fontsize='x-large')
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
rr, cc = 0,1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_title(f'Noise Level: {noise_level}%', fontweight='bold', fontsize='x-large')
# axes[rr,cc].legend()
# axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
rr, cc = 0,2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_title(f'Noise Level: {noise_level}%', fontweight='bold', fontsize='x-large')
# axes[rr,cc].legend()
# axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
rr, cc = 0,3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_title(f'Noise Level: {noise_level}%', fontweight='bold', fontsize='x-large')
# axes[rr,cc].legend()
# axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
rr, cc = 0,4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_title(f'Noise Level: {noise_level}%', fontweight='bold', fontsize='x-large')
# axes[rr,cc].legend()
# axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
rr, cc = 0,5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_title(f'Noise Level: {noise_level}%', fontweight='bold', fontsize='x-large')
# axes[rr,cc].legend()
# axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')






rr, cc = 1,0
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large')



rr, cc = 2,0
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large')



rr, cc = 3,0
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large')


plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.2)
fig.suptitle("Model Performance under Different Noise and Sample Conditions (Burgers Equation)",
             fontsize=18, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.show()

# Q_train, t_train, Q_test, t_test, Q_original_train, Q_original_test, num_samples = data_loader(data_name, step, noise_level, split_ratio)








############ PLOTS examples #################
data_name = 'burgers'
split_ratio = .5
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
num_samples = 100
noise_level = 0

name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim5*')
assert len(path)==1

data = np.load(path[0])

fig, axes = plt.subplots(2,2, sharex=True, figsize=[15,10])

axes[0,0].plot(data['t_train'], data['Q_train_'].T)
axes[0,0].plot(data['t_valid'], data['Q_valid_'].T)
axes[0,0].plot(data['t_test'], data['Q_test_'].T)

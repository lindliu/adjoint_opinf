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



data_name = 'burgers'
num_samples = 10000
split_ratio = .5
ratio = str(split_ratio).replace('.','p')

opinf_use_val = True
smoother = True
max_iter = 10

############ PLOTS #################
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=[10,7])

noise_level = 0
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/error_{name_suffix}*')
assert len(path)==1
error = np.load(path[0])

axes[0,0].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[0,0].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[0,0].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[0,0].legend()
axes[0,0].set_title(f'(a) {noise_level}% of noise level', fontsize='x-large')
# axes[0,0].set_xlabel('Model Dimension (r)', fontsize='large')
axes[0,0].set_ylabel(r'Relative State Error ($log_{10}$)', fontsize='large')

noise_level = 120
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/error_{name_suffix}*')
assert len(path)==1
error = np.load(path[0])

axes[1,0].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='ord2-OpInf', markersize=7, linewidth=3)
axes[1,0].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='ord6-OpInf', markersize=7, linewidth=3)
axes[1,0].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[1,0].legend()
axes[1,0].set_title(f'(d) {noise_level}% of noise level', fontsize='x-large')
axes[1,0].set_xlabel('Model Dimension (r)', fontsize='large')
axes[1,0].set_ylabel(r'Relative State Error ($log_{10}$)', fontsize='large')


plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)
plt.show()
# Q_train, t_train, Q_test, t_test, Q_original_train, Q_original_test, num_samples = data_loader(data_name, step, noise_level, split_ratio)

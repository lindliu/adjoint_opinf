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

import matplotlib
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 

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


# def plot_errors():
    
data_name = 'burgers'
split_ratio = .5
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10

############ PLOTS errors #################
fig, axes = plt.subplots(4, 6, sharex=True, figsize=[16,8])
pad = 5 # in points

cols = ['NL = 0%', 'NL = 40%', 'NL = 80%', 'NL = 120%', 'NL = 160%', 'NL = 200%']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, 
                ha='center', fontweight='bold', va='baseline')

rows = ['Samples: 20', 'Samples: 100', 'Samples: 1000', 'Samples: 10000']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=14,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向
    


num_samples = 20
rr = 0

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])




num_samples = 100
rr = 1

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])




num_samples = 1000
rr = 2

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])





num_samples = 10000
rr = 3

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')



plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Model Performance under Different Noise and Sample Conditions (Burgers' Equation)",
#              fontsize=18, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig('./figures_analysis/plot_error_burgers.png',bbox_inches='tight',dpi=300)
plt.show()

# Q_train, t_train, Q_test, t_test, Q_original_train, Q_original_test, num_samples = data_loader(data_name, step, noise_level, split_ratio)







data_name = 'fkpp'
split_ratio = .75
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10

############ PLOTS errors #################
fig, axes = plt.subplots(4, 6, sharex=True, figsize=[16,8])
pad = 5 # in points

cols = ['NL = 0%', 'NL = 40%', 'NL = 80%', 'NL = 120%', 'NL = 160%', 'NL = 200%']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, ha='center', fontweight='bold', va='baseline')

rows = ['Samples: 200', 'Samples: 500', 'Samples: 1000', 'Samples: 2001']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=14,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向
    


num_samples = 200
rr = 0

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])



num_samples = 500
rr = 1

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])




num_samples = 1000
rr = 2

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])





num_samples = 2001
rr = 3

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')



plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Model Performance under Different Noise and Sample Conditions (FKPP Equation)",
#              fontsize=18, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig('./figures_analysis/plot_error_fkpp.png',bbox_inches='tight',dpi=300)
plt.show()




















data_name = 'lcd'
split_ratio = .75
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10

############ PLOTS errors #################
fig, axes = plt.subplots(4, 6, sharex=True, figsize=[16,8])
pad = 5 # in points

cols = ['NL = 0%', 'NL = 40%', 'NL = 80%', 'NL = 120%', 'NL = 160%', 'NL = 200%']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, ha='center', fontweight='bold', va='baseline')

rows = ['Samples: 200', 'Samples: 500', 'Samples: 1000', 'Samples: 2001']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=14,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向
    


num_samples = 200
rr = 0

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])



num_samples = 500
rr = 1

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])




num_samples = 1000
rr = 2

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,1,2,3,4], [1,2,3,4,5])





num_samples = 2001
rr = 3

noise_level = 0
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 0
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,4,9,14], [1,5,10,15])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')
axes[rr,cc].set_ylabel(r'RSE ($log_{10}$)', fontsize='large') #Relative State Error
axes[rr,cc].legend()

noise_level = 40
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 1
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,4,9,14], [1,5,10,15])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 80
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 2
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,4,9,14], [1,5,10,15])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 120
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 3
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,4,9,14], [1,5,10,15])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 160
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 4
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,4,9,14], [1,5,10,15])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')

noise_level = 200
error = get_errors(data_name, num_samples, noise_level, ratio)
cc = 5
axes[rr,cc].plot(np.log10(error['error_opinf_2_list']), marker='*',  linestyle='-', color='#21918C', label='OpInf-ord2', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_opinf_6_list']), marker='o',  linestyle='--', color='#FCA636', label='OpInf-ord6', markersize=7, linewidth=3)
axes[rr,cc].plot(np.log10(error['error_adjoint_list']), marker='*', linestyle='--', color='#6A00A8', label='Adjoint', markersize=8, linewidth=3)
axes[rr,cc].set_xticks([0,4,9,14], [1,5,10,15])
axes[rr,cc].set_xlabel('Model Dimension (r)', fontsize='large')



plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Model Performance under Different Noise and Sample Conditions (CDE)",
#              fontsize=18, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig('./figures_analysis/plot_error_CDE.png',bbox_inches='tight',dpi=300)
plt.show()













############ PLOTS examples #################
from matplotlib.lines import Line2D

data_name = 'burgers'
split_ratio = .5
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
r = 5

num_samples = 1000

x_indices = range(0,r)
colors = plt.cm.plasma(np.linspace(0, 1, len(x_indices)))


fig, axes = plt.subplots(3,3, sharex=True, sharey=True, figsize=[16,10])
pad = 5 # in points

cols = ['OpInf-ord2', 'OpInf-ord6', 'Adjoint']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=25, ha='center', fontweight='bold', va='baseline')

rows = ['NL = 0%', 'NL = 80%', 'NL = 160%']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=25,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向



fontsize_text = 18
T = 1
for ax in axes.ravel():
    # x in data coordinates, y in axes coordinates (0–1)
    trans = ax.get_xaxis_transform()  # x=data, y=axes

    ax.text(0.4 * T, 0.05, "Train",
            ha="right", color="black", fontsize=fontsize_text,
            transform=trans)
    ax.text(0.59 * T, 0.05, "Val",
            ha="right", color="black", fontsize=fontsize_text,
            transform=trans)
    ax.text(0.7 * T, 0.05, "Test",
            ha="left", color="black", fontsize=fontsize_text,
            transform=trans)
    
    ax.axvline(x=.5*T, ymin=0, ymax=0.96, color="gray", linewidth=1.3, linestyle="--")
    ax.axvline(x=.6*T, ymin=0, ymax=0.96, color="gray", linewidth=1.3, linestyle="--")
    ax.set_xlim(0, T)



noise_level = 0
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 0

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$q_{{x_indices[i]},true}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$q_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)





cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)






noise_level = 80
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 1

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
# axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)





cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
# axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
# axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)







noise_level = 160
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 2

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
# axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)





cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
# axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
# axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="center left", fontsize=15)



plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Illustrative Predictions under Different Noise Levels (Burgers' Equation)",
#              fontsize=26, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig(f'./figures_analysis/plot_examples_burgers_sample{num_samples}.png',bbox_inches='tight',dpi=300)
plt.show()

# Illustration of Predicted Trajectories from OpInf (order 2, 6) and Adjoint Methods



























############ PLOTS examples fpkk #################
from matplotlib.lines import Line2D

data_name = 'fkpp'
split_ratio = .75
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
r = 5

num_samples = 1000

x_indices = range(0,r)
colors = plt.cm.plasma(np.linspace(0, 1, len(x_indices)))


fig, axes = plt.subplots(3,3, sharex=True, sharey=True, figsize=[16,10])
pad = 5 # in points

cols = ['OpInf-ord2', 'OpInf-ord6', 'Adjoint']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=25, ha='center', fontweight='bold', va='baseline')

rows = ['NL = 0%', 'NL = 80%', 'NL = 160%']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=25,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向

fontsize_text = 18
T = 5
for ax in axes.ravel():
    # x in data coordinates, y in axes coordinates (0–1)
    trans = ax.get_xaxis_transform()  # x=data, y=axes

    ax.text(0.65 * T, 0.05, "Train",
            ha="right", color="black", fontsize=fontsize_text,
            transform=trans)
    ax.text(0.84 * T, 0.05, "Val",
            ha="right", color="black", fontsize=fontsize_text,
            transform=trans)
    ax.text(0.87 * T, 0.05, "Test",
            ha="left", color="black", fontsize=fontsize_text,
            transform=trans)
    
    ax.axvline(x=.75*T, ymin=0, ymax=0.96, color="gray", linewidth=1.3, linestyle="--")
    ax.axvline(x=.85*T, ymin=0, ymax=0.96, color="gray", linewidth=1.3, linestyle="--")
    ax.set_xlim(0, T)


noise_level = 0
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 0

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_6'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)










noise_level = 80
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 1

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_6'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)









noise_level = 160
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 2

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_6'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Illustrative Predictions under Different Noise Levels (FKPP Equation)",
#              fontsize=26, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig(f'./figures_analysis/plot_examples_FKPP_sample{num_samples}.png',bbox_inches='tight',dpi=300)
plt.show()






























# import matplotlib
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 

############ PLOTS examples CDE #################
from matplotlib.lines import Line2D

data_name = 'lcd'  # linear convection–diffusion equation
split_ratio = .75
ratio = str(split_ratio).replace('.','p')

smoother = True
max_iter = 10
opinf_use_val = True
r = 15

num_samples = 1000

x_indices = range(0,r)
colors = plt.cm.plasma(np.linspace(0, 1, len(x_indices)))


fig, axes = plt.subplots(3,3, sharex=True, sharey=True, figsize=[16,10])
pad = 5 # in points

cols = ['OpInf-ord2', 'OpInf-ord6', 'Adjoint']
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=25, ha='center', fontweight='bold', va='baseline')

rows = ['NL = 0%', 'NL = 80%', 'NL = 160%']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                fontsize=25,
                fontweight='bold',  # ✅ 加粗
                ha='right', va='center',
                rotation=ax.yaxis.label.get_rotation())  # ✅ 与 ylabel 同方向


fontsize_text = 18
T=.5
for ax in axes.ravel():
    # x in data coordinates, y in axes coordinates (0–1)
    trans = ax.get_xaxis_transform()  # x=data, y=axes

    ax.text(0.65 * T, 0.05, "Train",
            ha="right", color="black", fontsize=fontsize_text,
            transform=trans)
    ax.text(0.84 * T, 0.05, "Val",
            ha="right", color="black", fontsize=fontsize_text,
            transform=trans)
    ax.text(0.87 * T, 0.05, "Test",
            ha="left", color="black", fontsize=fontsize_text,
            transform=trans)
    
    ax.axvline(x=.75*T, ymin=0, ymax=0.96, color="gray", linewidth=1.3, linestyle="--")
    ax.axvline(x=.85*T, ymin=0, ymax=0.96, color="gray", linewidth=1.3, linestyle="--")
    ax.set_xlim(0, T)

noise_level = 0
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 0

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_6'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)










noise_level = 80
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 1

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_6'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
# axes[rr,cc].set_xlabel(r"$t$", fontsize=20, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)









noise_level = 160
name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'
path = glob.glob(f'./results/Predictions_{name_suffix}_best_dim{r}*')
assert len(path)==1
data = np.load(path[0])
rr = 2

cc = 0
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_2'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
axes[rr,cc].set_ylabel(r"$q(t)$", fontsize=22, fontweight='bold')
axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




cc = 1
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_opinf_6'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)





cc = 2
for i, color in zip(x_indices, colors):
    axes[rr,cc].plot(data['t_train'], data['Q_train_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_valid'], data['Q_valid_'][i,:], marker='+', color=color, linestyle='None')
    axes[rr,cc].plot(data['t_test'], data['Q_test_'][i,:], color=color, linestyle="-", linewidth=4)
    
    axes[rr,cc].plot(data['t_test'], data['Q_adjoint'][i,:], color="#00FFFF", linestyle='--', linewidth=2)

# Customize axes thickness
axes[rr,cc].spines['left'].set_linewidth(1.7)
axes[rr,cc].spines['bottom'].set_linewidth(1.7)
axes[rr,cc].spines['top'].set_visible(False)  # Hide top axis
axes[rr,cc].spines['right'].set_visible(False)  # Hide right axis

# Adjust tick parameters (size of ticks and labels)
axes[rr,cc].tick_params(axis='both', which='major', labelsize=15, length=6, width=2)
axes[rr,cc].tick_params(axis='both', which='minor', labelsize=15, length=4, width=1.5)

# Ensure only left and bottom ticks are visible
axes[rr,cc].yaxis.set_ticks_position('left')
# axes[rr,cc].xaxis.set_ticks_position('bottom')

# # Create legend
# legend_elements = []
# for i in range(r):
#     legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3, label=rf"$\hat{{q}}_{{{x_indices[i]},true}}(t)$"))
# legend_elements.append(Line2D([0], [0], color="#00FFFF", linestyle="--", linewidth=2, label=r"$\hat{q}_{i,pred}(t)$"))

# Create legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='True data'),
    Line2D([0], [0], color='#00FFFF', linestyle='--', label='Prediction')
]

# # axes[rr,cc].set_title(rf"{method_names[method]} -- $\mathbf{{{noise} \%}}$", fontsize=33)
axes[rr,cc].set_xlabel(r"$t$", fontsize=22, fontweight='bold')
# axes[rr,cc].set_ylabel(r"$\hat{q}(t)$", fontsize=20, fontweight='bold')
# axes[rr,cc].legend(handles=legend_elements, loc="lower left", fontsize=15)




plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
# fig.suptitle("Illustrative Predictions under Different Noise Levels (CDE)",
#              fontsize=26, fontweight='bold', y=1.02)
# Effect of Noise and Sample Size on Model Accuracy for the Burgers Equation System
plt.savefig(f'./figures_analysis/plot_examples_CDE_sample{num_samples}.png',bbox_inches='tight',dpi=300)
plt.show()


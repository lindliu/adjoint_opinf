#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:43:17 2025

@author: dliu
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import opinf
from train_adjoint import ode_solver, func_surrogate, model_reducer
from train_opinf import add_noise, get_train_test_data
import glob 
from functools import partial


import random
random.seed(10)
np.random.seed(10)    # for numpy random



###### config #####
data_name = 'burgers' 
order = 'ord6'  # 'ord2' # 
noise_level = 20 ## 0 is no noise on train data
num_samples = 10000


if data_name == 'burgers':
    split_ratio = .5   
if data_name == 'fkpp':
    split_ratio = .75   
    
    

if data_name == 'burgers':
    data_file = "./data/total_burgers_snapshots_nu_01.npz"  ###  
if data_name == 'fkpp':
    data_file = "./data/total_fkpp.npz"     ## 
data = np.load(data_file)

Q_original, t = data['Q'], data['t']
Q_original = Q_original.reshape(-1, Q_original.shape[-1])

Q_original_train, t_train, Q_original_test, t_test = get_train_test_data(Q_original, t, num_samples=num_samples, split_ratio=split_ratio)
k_samples = Q_original_train.shape[1]

Q = add_noise(Q_original, percentage=noise_level, method="std")
Q_train, t_train, Q_test, t_test = get_train_test_data(Q, t, num_samples=num_samples, split_ratio=split_ratio)




error = np.load(f"./data/error_{data_name}_{order}_noise{noise_level}_sam{num_samples}_.npz")
error_opinf_list, error_adjoint_list = error['error_opinf_list'], error['error_adjoint_list']
    

# error_opinf_list, error_adjoint_list = [], []
# for r in range(1,8):
# # for r in [5]:

#     theta_opinf_path = glob.glob(f'./data/theta_opinf_{data_name}_{r}_{order}_*_noise{noise_level}_sam{num_samples}*.npz')[0]
#     theta_adjoint_path = glob.glob(f'./data/theta_adjoint_{data_name}_{r}_{order}_*_noise{noise_level}_sam{num_samples}*.npz')[0]

#     theta_opinf = np.load(theta_opinf_path)
#     A_opinf, H_opinf = theta_opinf['A_opinf'], theta_opinf['H_opinf']
#     theta_opt = np.load(theta_adjoint_path)
#     A_opt, H_opt = theta_opt['A_opt'], theta_opt['H_opt']

    
#     ### reduce data order to r
#     Q_, _ = model_reducer(Q_train, Q_test, r)
#     Q_, Q_test_ = model_reducer(Q_train, Q_original_test, r)
#     Q_all_ = np.c_[Q_,Q_test_]
    
#     Q_opinf = ode_solver(func_surrogate, Q_[:,0], t, par=(A_opinf, H_opinf))
#     Q_adjoint = ode_solver(func_surrogate, Q_[:,0], t, par=(A_opt, H_opt))

#     error_opinf = Q_all_ - Q_opinf
#     error_adjoint = Q_all_ - Q_adjoint
        
#     error_opinf_list.append(np.mean(error_opinf[:,k_samples:]**2))
#     error_adjoint_list.append(np.mean(error_adjoint[:,k_samples:]**2))




fig, axes = plt.subplots(1,2,figsize=(12,8))
axes[0].plot(error_opinf_list, marker='+', label='opinf')
axes[0].plot(error_adjoint_list, marker='o', label='adjoint')
axes[0].set_xlabel('Model Dimension(r)')
axes[0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
axes[0].legend()









# r = 5
# theta_opinf_path = glob.glob(f'./data/theta_opinf_{data_name}_{r}_{order}_*_noise{noise_level}_sam{num_samples}*.npz')[0]
# theta_adjoint_path = glob.glob(f'./data/theta_adjoint_{data_name}_{r}_{order}_*_noise{noise_level}_sam{num_samples}*.npz')[0]

# theta_opinf = np.load(theta_opinf_path)
# A_opinf, H_opinf = theta_opinf['A_opinf'], theta_opinf['H_opinf']
# theta_opt = np.load(theta_adjoint_path)
# A_opt, H_opt = theta_opt['A_opt'], theta_opt['H_opt']


# ### reduce data order to r
# Q_, _ = model_reducer(Q_train, Q_test, r)
# Q_, Q_test_ = model_reducer(Q_train, Q_original_test, r)
# Q_all_ = np.c_[Q_,Q_test_]

# Q_opinf = ode_solver(func_surrogate, Q_[:,0], t, par=(A_opinf, H_opinf))
# Q_adjoint = ode_solver(func_surrogate, Q_[:,0], t, par=(A_opt, H_opt))

    


# fig, axes = plt.subplots(1,2,figsize=[12,5])
# axes[0].plot(t, Q_all_.T, label='true')
# axes[0].plot(t, Q_opinf.T, '--')
# axes[0].axvline(x=t_train[-1], ls='--')
# axes[0].title.set_text('opinf vs true')
# axes[1].plot(t, Q_all_.T, label='true')
# axes[1].plot(t, Q_adjoint.T, '--')
# axes[1].axvline(x=t_train[-1], ls='--')
# axes[1].title.set_text('adjoint vs true')




# error_adjoint = Q_all_ - Q_adjoint
# error_opinf = Q_all_ - Q_opinf

# print(f'adjoint train error: {np.mean(error_adjoint[:,:k_samples]**2)}')
# print(f'opinf train error: {np.mean(error_opinf[:,:k_samples]**2)}')

# print(f'adjoint test error: {np.mean(error_adjoint[:,k_samples:]**2)}')
# print(f'opinf test error: {np.mean(error_opinf[:,k_samples:]**2)}')


# print(f'train error (opinf-adjoint): {np.mean(error_opinf[:,:k_samples]**2)- np.mean(error_adjoint[:,:k_samples]**2)}')
# print(f'test error (opinf-adjoint): {np.mean(error_opinf[:,k_samples:]**2)- np.mean(error_adjoint[:,k_samples:]**2)}')



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:59:23 2026

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def ralative_error(A, B):
    return np.sum((A-B)**2)**.5/np.sum(B**2)**.5
    
#table 1 and 2
pertube = 1 #5,4,3,2,1
noise = 0

# table 3 and 4
pertube = 0 
noise = 20#1,5,10,20
for r in range(1,6):

    theta = np.load(glob.glob(f'./results_exact/theta_*noise{noise}_*best_dim{r}_*_per{pertube}.npz')[0])
    A_adjoint, H_adjoint, A_opinf, H_opinf = theta['A_adjoint'], theta['H_adjoint'], theta['A_opinf'], theta['H_opinf']
    
    data = np.load(glob.glob(f"./data/burgers/exact_intrusive_operator/exact_operator_reproj_noise{noise}_{r}.npz")[0])
    A_r_true, H_r_true, Q_reproj, t = data['A_r_true'], data['H_r_true'], data['Q_reproj'], data['t']
    
    theta_opinf = np.r_[A_opinf.flatten(), H_opinf.flatten()]
    theta_adjoint = np.r_[A_adjoint.flatten(), H_adjoint.flatten()]
    theta_true = np.r_[A_r_true.flatten(), H_r_true.flatten()]
    
    operator_opinf_error = ralative_error(theta_opinf, theta_true)
    operator_adjoint_error = ralative_error(theta_adjoint, theta_true)
    
    
    print(f'dim{r} adjoint operator error: {operator_adjoint_error: .4e}, opinf: {operator_opinf_error:.4e}')
    
                
    
for r in range(1,6):

    prediction = np.load(glob.glob(f'./results_exact/Predictions_*noise{noise}_*best_dim{r}_*_per{pertube}.npz')[0])
    
    
    error = np.load(glob.glob(f'./results_exact/error_*noise{noise}_*_per{pertube}.npz')[0])
    error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = \
            error['error_opinf_6_train_list'], error['error_adjoint_train_list'], error['error_opinf_2_train_list']
    
    
    # print(f'dim{r} adjoint trajectory error: {error_adjoint_train_list[r-1]: .4e}, opinf: {error_opinf_6_train_list[r-1]:.4e}')
    print(f'dim{r} adjoint trajectory error: {error_adjoint_train_list[r-1]**.5: .4e}, opinf: {error_opinf_6_train_list[r-1]**.5:.4e}')















# ################ rank problem
# for r in range(1,6):

#     theta = np.load(glob.glob(f'./results_exact_rank/theta_*noise0*_best_dim{r}_*_per0.npz')[0])
#     A_adjoint, H_adjoint, A_opinf, H_opinf = theta['A_adjoint'], theta['H_adjoint'], theta['A_opinf'], theta['H_opinf']
    
#     data = np.load(glob.glob(f"./data/burgers/exact_intrusive_operator/exact_operator_reproj_noise0_{r}.npz")[0])
#     A_r_true, H_r_true, Q_reproj, t = data['A_r_true'], data['H_r_true'], data['Q_reproj'], data['t']
    
#     theta_opinf = np.r_[A_opinf.flatten(), H_opinf.flatten()]
#     theta_adjoint = np.r_[A_adjoint.flatten(), H_adjoint.flatten()]
#     theta_true = np.r_[A_r_true.flatten(), H_r_true.flatten()]
    
#     operator_opinf_error = ralative_error(theta_opinf, theta_true)
#     operator_adjoint_error = ralative_error(theta_adjoint, theta_true)
    
    
#     print(f'dim{r} adjoint operator error: {operator_adjoint_error: .4e}, opinf: {operator_opinf_error:.4e}')
    
                
    
# # for r in range(3,6):
# r=3
# prediction = np.load(glob.glob(f'./results_exact_rank/Predictions_*noise0*_best_dim{r}_*_per0.npz')[0])
    

# error = np.load(glob.glob(f'./results_exact_rank/error_*noise0*_per0.npz')[0])
# error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = \
#         error['error_opinf_6_train_list'], error['error_adjoint_train_list'], error['error_opinf_2_train_list']
    
# for i, r in enumerate([1,2,3,4,5]):
    
#     print(f'dim{r} adjoint trajectory error: {error_adjoint_train_list[i]: .4e}, opinf: {error_opinf_6_train_list[i]:.4e}')



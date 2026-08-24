#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 09:50:38 2025

@author: dliu
"""


import matplotlib.pyplot as plt
import numpy as np
import opinf
from scipy.interpolate import interp1d
from utils import get_train_test_data, add_noise, smooth, \
                  get_theta_by_opinf, model_reducer, optimal_opinf, \
                  integrate, ode_solver, func_surrogate, func_lambda
from scipy.integrate import solve_ivp
import glob
import os

import random
random.seed(10)
np.random.seed(10)    # for numpy random
################################ OpInf ###############################
######################################################################

# opinf.utils.mpl_config()

def main(data_name, r, noise_level, step, smoother=False, pieces=[2], reg_Frobenius=0, weighted=False, max_iter=10, split_ratio=.75, split_ratio_validation=.1):
    
    import random
    random.seed(10)
    np.random.seed(10)    # for numpy random
    
    if data_name == 'burgers':
        data_file = glob.glob(os.path.join(os.getcwd(),"./data/burgers/total_burgers_snapshots_nu_01.npz"))[0]  ###  
        num_samples = 10000//step ## 2000 ##
        # split_ratio = .5   
        
        data = np.load(data_file)

    if data_name == 'fkpp':        
        data_files = [glob.glob(os.path.join(os.getcwd(), f'./data/fkpp/total_fkpp_{i}.npy'))[0] for i in range(1,6)]
        data_Q = [np.load(data_files[i]) for i in range(5)]
        data_Q = np.concatenate(data_Q,axis=2)
        
        data = {}
        data['Q'] = data_Q
        data['t'] = np.load(glob.glob(os.path.join(os.getcwd(), "./data/fkpp/total_fkpp_t.npy"))[0])
        data['x'] = np.load(glob.glob(os.path.join(os.getcwd(), './data/fkpp/total_fkpp_x.npy'))[0])
        data['y'] = np.load(glob.glob(os.path.join(os.getcwd(), './data/fkpp/total_fkpp_y.npy'))[0])
        
        num_samples = 2001//step ## 2000 ##
        # split_ratio = .75   


    Q_original, t = data['Q'], data['t']
    
    # if data_name == 'fkpp':
    #     Q_original = Q_original[::4,::4,:]
    Q_original = Q_original.reshape(-1, Q_original.shape[-1])

    Q_original_train, t_train, Q_original_test, t_test = \
        get_train_test_data(Q_original, t, step=step, split_ratio=split_ratio)

    Q = add_noise(Q_original, percentage=noise_level, method="std")
    Q_train, t_train, Q_test, t_test = get_train_test_data(Q, t, step=step, split_ratio=split_ratio)

    
    # # Snapshot data Q = [q(t_0) q(t_1) ... q(t_k)], size=(r,k)
    dt = t_train[1] - t_train[0]
    
    ### reduce data order to r
    Q__, Q_test_, svdvals = model_reducer(Q_train, Q_test, r)
    k_samples = Q__.shape[1]  # number of samples for training(snapshot data)
    
    ### smoother
    # smoother = False
    if smoother and noise_level>0:
        Q_s, _ = smooth(Q__, t_train, window_size=None, poly_order=3)
        # Q__, _ = smooth(Q__, t_train, window_size=None, poly_order=3)
    
        resid = Q_s - Q__
        var = np.var(resid, axis=1) + 1e-10*noise_level
    else:
        var = 1

        
    if weighted:
        ### 权重，特征值越小噪音越大，则其权重越小
        # weights = svdvals[:r]
        weights = svdvals[:r]/(var+1e-8)
        # weights = svdvals[:r]**2/var

        weights = weights/weights.sum()
    else:
        weights = np.ones(r)
    
    
    ############################################
    t_all = np.r_[t_train,t_test]
    Q_, Q_test_, svdvals = model_reducer(Q_train, Q_original_test, r)
    Q_all_ = np.c_[Q_,Q_test_]
    
    
    
    ### select best A_opinf, H_opinf by grid search ####
    A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
        optimal_opinf(Q__, t_train, t_test, 'ord6', M=np.max(np.abs(Q__))*10, T=t[-1], valid_ratio=split_ratio_validation, Q_test_=Q_test_)
    
    A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
        optimal_opinf(Q__, t_train, t_test, 'ord2', M=np.max(np.abs(Q__))*10, T=t[-1], valid_ratio=split_ratio_validation, Q_test_=Q_test_)
    
    
    reg = str(reg_Frobenius).replace('.','p')
    ### save result (theta)
    if save_results:
        file_opinf = f"./results_opinf/theta_opinf_{data_name}_sam{num_samples}_ratio{ratio}_validrat{valid_ratio_str}_noise{noise_level}_dim{r}_{weighted}_reg{reg}_iter{max_iter}_weighted{weighted}.npz"
        
        np.savez(file_opinf, A_opinf_6=A_opinf_6, H_opinf_6=H_opinf_6, A_opinf_2=A_opinf_2, H_opinf_2=H_opinf_2)
    
        # theta_opinf = np.load(file_opinf)
        # A_opinf_6, H_opinf_6 = theta_opinf['A_opinf_6'], theta_opinf['H_opinf_6']
    
    
    
    
    
    # ############### opinf vs adjoint #############    
    t_all = np.r_[t_train,t_test]
    Q_, Q_test_, svdvals = model_reducer(Q_train, Q_original_test, r)
    Q_all_ = np.c_[Q_,Q_test_]
    
    
    split_ratio_validation = .1
    train_idx = Q_.shape[1]
    valid_idx = int(split_ratio_validation * Q_all_.shape[1]) + train_idx
    
    
    t_val = t_all[train_idx:valid_idx]
    t_test = t_all[valid_idx:]
    Q_val_ = Q_all_[:, train_idx:valid_idx]
    Q_test_ = Q_all_[:, valid_idx:]
    
    
    
    fig, axes = plt.subplots(3,3,figsize=[16,10])
    
    Q_0 = Q_all_[:,0]
    Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_all, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_all, par=(A_opinf_2, H_opinf_2), rescale=True)

    error_opinf_6_init_valid = np.mean((Q_val_.T - Q_opinf_6[:,train_idx:valid_idx].T)**2)/np.mean(Q_val_.T**2)
    error_opinf_2_init_valid = np.mean((Q_val_.T - Q_opinf_2[:,train_idx:valid_idx].T)**2)/np.mean(Q_val_.T**2)
    
    error_opinf_6_init_test = np.mean((Q_test_.T - Q_opinf_6[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
    error_opinf_2_init_test = np.mean((Q_test_.T - Q_opinf_2[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
    
    axes[0,0].plot(t_all, Q_all_.T, label='true')
    axes[0,0].plot(t_all, Q_opinf_6.T, '--')
    axes[0,0].axvline(x=t_all[train_idx], ls='--')
    axes[0,0].axvline(x=t_all[valid_idx], ls='--')
    axes[0,0].title.set_text(f'opinf_6 vs true: {np.log10(error_opinf_6_init_test):.3} val: {np.log10(error_opinf_6_init_valid):.3}')
    axes[0,1].plot(t_all, Q_all_.T, label='true')
    axes[0,1].plot(t_all, Q_opinf_2.T, '--')
    axes[0,1].axvline(x=t_all[train_idx], ls='--')
    axes[0,1].axvline(x=t_all[valid_idx], ls='--')
    axes[0,1].title.set_text(f'opinf_2 vs true: {np.log10(error_opinf_2_init_test):.3} val: {np.log10(error_opinf_2_init_valid):.3}')
    
    
    
    Q_0 = Q_val_[:,0]
    Q_opinf_6_val = ode_solver(func_surrogate, Q_0, t_val, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_opinf_2_val = ode_solver(func_surrogate, Q_0, t_val, par=(A_opinf_2, H_opinf_2), rescale=True)
    error_opinf_6_valid = np.mean((Q_val_.T - Q_opinf_6_val.T)**2)/np.mean(Q_val_.T**2)
    error_opinf_2_valid = np.mean((Q_val_.T - Q_opinf_2_val.T)**2)/np.mean(Q_val_.T**2)
    
    Q_0 = Q_[:,0]
    Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2), rescale=True)

    error_opinf_6_train = np.mean((Q_.T - Q_opinf_6.T)**2)/np.mean(Q_.T**2)
    error_opinf_2_train = np.mean((Q_.T - Q_opinf_2.T)**2)/np.mean(Q_.T**2)
    
    axes[1,0].plot(t_all, Q_all_.T, label='true')
    axes[1,0].plot(t_train, Q_opinf_6.T, '--')
    axes[1,0].axvline(x=t_all[train_idx], ls='--')
    axes[1,0].title.set_text(f'opinf_6 vs true, train: {np.log10(error_opinf_6_train):.3} val: {np.log10(error_opinf_6_valid):.3}')
    axes[1,1].plot(t_all, Q_all_.T, label='true')
    axes[1,1].axvline(x=t_all[train_idx], ls='--')
    axes[1,2].plot(t_all, Q_all_.T, label='true')
    axes[1,2].plot(t_train, Q_opinf_2.T, '--')
    axes[1,2].axvline(x=t_all[train_idx], ls='--')
    axes[1,2].title.set_text(f'opinf_2 vs true, train: {np.log10(error_opinf_2_train):.3} val: {np.log10(error_opinf_2_valid):.3}')
    
    
    axes[1,0].plot(t_val, Q_opinf_6_val.T, '--')
    axes[1,0].axvline(x=t_all[valid_idx], ls='--')
    axes[1,1].axvline(x=t_all[valid_idx], ls='--')
    axes[1,2].plot(t_val, Q_opinf_2_val.T, '--')
    axes[1,2].axvline(x=t_all[valid_idx], ls='--')


    
    Q_0 = Q_test_[:,0]
    Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_test, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_test, par=(A_opinf_2, H_opinf_2), rescale=True)
    error_opinf_6 = np.mean((Q_test_.T - Q_opinf_6.T)**2)/np.mean(Q_test_.T**2)
    error_opinf_2 = np.mean((Q_test_.T - Q_opinf_2.T)**2)/np.mean(Q_test_.T**2)
    
    axes[2,0].plot(t_all, Q_all_.T, label='true')
    axes[2,0].plot(t_test, Q_opinf_6.T, '--')
    axes[2,0].axvline(x=t_all[valid_idx], ls='--')
    axes[2,0].title.set_text(f'opinf_6 vs true: {np.log10(error_opinf_6):.3}')
    axes[2,1].plot(t_all, Q_all_.T, label='true')
    axes[2,1].axvline(x=t_all[valid_idx], ls='--')
    axes[2,2].plot(t_all, Q_all_.T, label='true')
    axes[2,2].plot(t_test, Q_opinf_2.T, '--')
    axes[2,2].axvline(x=t_all[valid_idx], ls='--')
    axes[2,2].title.set_text(f'opinf_2 vs true: {np.log10(error_opinf_2):.3}')
    
    fig.suptitle(f'reg_Frobenius: {reg_Frobenius}, pieces: {pieces}, weighted: {weighted}')
    
    if save_results:
        fig.savefig(f'./figures_opinf/Results_{data_name}_sam{num_samples}_ratio{ratio}_validrat{valid_ratio_str}_noise{noise_level}_dim{r}_{weighted}_reg{reg}_iter{max_iter}_weighted{weighted}.png')
        plt.close()

    print(f'opinf 6 test error: {error_opinf_6:.6}, val error: {error_opinf_6_valid}')

    
    return Q_, Q_test_, Q_opinf_6, Q_opinf_2, t_all, valid_idx, \
        error_opinf_6_init_test, error_opinf_2_init_test, \
        error_opinf_6_train, error_opinf_2_train, \
        error_opinf_6_init_valid, error_opinf_2_init_valid, \
        error_opinf_6_valid, error_opinf_2_valid    
    
if __name__ == "__main__": 
    
    ###### config #####
    smoother = True #  False # 
    max_iter = 10
    # ###Perform piecewise integration and optimization; if it is a list, then divide it into segments in order and optimize accordingly.
    pieces = [3,1,3]  # [5,1,5] # [1]
    
    save_results = True # False # 
    
    # data_name = 'fkpp'  ###   'burgers'  ##  
    # ## if no regularizer on adjoint, let reg_Frobenius=0
    # reg_Frobenius = 1e1 # 0 # 
    # weighted = False # True # 

    fig, axes = plt.subplots(4,3,figsize=(16,16))

    split_ratio_validation = .1 # 0 # 
    for split_ratio_validation in [0,.1]:
        valid_ratio_str = str(split_ratio_validation).replace('.','p')
    
        # for data_name in ['burgers', 'fkpp']:
        for data_name in ['fkpp']:
            if data_name=='fkpp':
                step = 1  ## 1, 2, 4, 5, 10, 20, 40
                num_samples = 2001//step ## 2000 ##
                split_ratio = .75
                
            if data_name=='burgers':
                step = 1 # 1 # 10 # 20 #  # 500 # 100
                num_samples = 10000//step # 10000
                split_ratio = .5
            
            ratio = str(split_ratio).replace('.','p')
            
            noise_level_list = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
            # for noise_level in noise_level_list:
            for noise_level in [60]:
                error_opinf_6_init_list, error_adjoint_init_list, error_opinf_2_init_list = [], [], []
                error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = [], [], []
                error_opinf_6_list, error_adjoint_list, error_opinf_2_list = [], [], []
                error_opinf_6_valid_list, error_adjoint_valid_list, error_opinf_2_valid_list = [], [], []
                
                for r in range(1,6):
                # for r in [4]:
                    print(f'dimension: {r}')
                    ##########################################
                    Q_, Q_test_, Q_opinf_6, Q_opinf_2, t_all, valid_idx, \
                    error_opinf_6_init_test, error_opinf_2_init_test, \
                    error_opinf_6_train, error_opinf_2_train, \
                    error_opinf_6_init_valid, error_opinf_2_init_valid, \
                    error_opinf_6_valid, error_opinf_2_valid   \
                        = main(data_name, r, noise_level, step, \
                               smoother, pieces, 0, False, max_iter, split_ratio, split_ratio_validation=split_ratio_validation)
                    
                    
                    error_state_opinf_6 = Q_test_.T - Q_opinf_6.T
                    error_state_opinf_2 = Q_test_.T - Q_opinf_2.T
                    
                    error_opinf_6 = np.mean(error_state_opinf_6**2)/np.mean(Q_test_.T**2)
                    error_opinf_2= np.mean(error_state_opinf_2**2)/np.mean(Q_test_.T**2)
            
                    error_opinf_6_list.append(error_opinf_6)
                    error_opinf_2_list.append(error_opinf_2)
                    
                    error_opinf_6_init_list.append(error_opinf_6_init_test)
                    error_opinf_2_init_list.append(error_opinf_2_init_test)
                    
                    error_opinf_6_train_list.append(error_opinf_6_train)
                    error_opinf_2_train_list.append(error_opinf_2_train)
    
                    error_opinf_6_valid_list.append(error_opinf_6_valid)
                    error_opinf_2_valid_list.append(error_opinf_2_valid) 
                    
                    
                error_opinf_6_list = np.array(error_opinf_6_list)
                error_opinf_2_list = np.array(error_opinf_2_list)
            
                error_opinf_6_init_list = np.array(error_opinf_6_init_list)
                error_opinf_2_init_list = np.array(error_opinf_2_init_list)
                
                error_opinf_6_train_list = np.array(error_opinf_6_train_list)
                error_opinf_2_train_list = np.array(error_opinf_2_train_list)
    
                error_opinf_6_valid_list = np.array(error_opinf_6_valid_list)
                error_opinf_2_valid_list = np.array(error_opinf_2_valid_list)
                
                
                if save_results:
                    np.savez(f"./results_opinf/error_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_iter{max_iter}_smooth{smoother}_validrat{valid_ratio_str}.npz", 
                            error_opinf_6_list=error_opinf_6_list, error_opinf_2_list=error_opinf_2_list,
                            error_opinf_6_init_list=error_opinf_6_init_list, error_opinf_2_init_list=error_opinf_2_init_list,
                            error_opinf_6_train_list=error_opinf_6_train_list, error_opinf_2_train_list=error_opinf_2_train_list,
                            error_opinf_6_valid_list=error_opinf_6_valid_list, error_opinf_2_valid_list=error_opinf_2_valid_list)
                
                
                
                axes[0,0].plot(np.log10(error_opinf_6_init_list), marker='+', label='opinf_6')
                # axes[0,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[0,0].set_ylabel(r'test(init) relative error ($log_{10}$)', fontsize='x-large')
                axes[0,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[0,0].legend()
                
                axes[0,1].plot(np.log10(error_opinf_2_init_list), marker='+', label='opinf_2')
                # axes[0,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[0,1].set_ylabel(r'test(init) relative error ($log_{10}$)', fontsize='x-large')
                axes[0,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[0,1].legend()
                
                
                axes[1,0].plot(np.log10(error_opinf_6_train_list), marker='+', label='opinf_6')
                # axes[1,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[1,0].set_ylabel(r'train relative error ($log_{10}$)', fontsize='x-large')
                # axes[1,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[1,0].legend()
                
                axes[1,1].plot(np.log10(error_opinf_2_train_list), marker='+', label='opinf_2')
                # axes[1,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[1,1].set_ylabel(r'train relative error ($log_{10}$)', fontsize='x-large')
                # axes[1,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[1,1].legend()
                
                
                
                axes[2,0].plot(np.log10(error_opinf_6_valid_list), marker='+', label='opinf_6')
                # axes[2,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[2,0].set_ylabel(r'valid relative error ($log_{10}$)', fontsize='x-large')
                # axes[2,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[2,0].legend()
                
                axes[2,1].plot(np.log10(error_opinf_2_valid_list), marker='+', label='opinf_2')
                # axes[2,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[2,1].set_ylabel(r'valid relative error ($log_{10}$)', fontsize='x-large')
                # axes[2,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[2,1].legend()
                
    
                
                axes[3,0].plot(np.log10(error_opinf_6_list), marker='+', label=f'{split_ratio_validation}')
                axes[3,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[3,0].set_ylabel(r'test relative error ($log_{10}$)', fontsize='x-large')
                # axes[3,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[3,0].legend()
                
                axes[3,1].plot(np.log10(error_opinf_2_list), marker='+', label=f'{split_ratio_validation}')
                axes[3,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[3,1].set_ylabel(r'test relative error ($log_{10}$)', fontsize='x-large')
                # axes[3,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
                axes[3,1].legend()
                
        
    if save_results:
        fig.savefig(f'./figures_opinf/plot_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_iter{max_iter}_smooth{smoother}_validrat{valid_ratio_str}.png')
        plt.close()
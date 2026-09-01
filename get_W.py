
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:33:53 2025

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

def data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation=.1):
    
    #### the seed should be fixed in function where random values are used ###
    random.seed(10)
    np.random.seed(10)    
    ##########################################################################
    
    if data_name == 'burgers':
        data_file = glob.glob(os.path.join(os.getcwd(),"./data/burgers/total_burgers_snapshots_nu_01.npz"))[0]  ###   
        data = np.load(data_file)
        # num_samples = 10000//step ## 2000 ##
        # split_ratio = .5  
        
    if data_name == 'fkpp':        
        data_files = [glob.glob(os.path.join(os.getcwd(), f'./data/fkpp/total_fkpp_{i}.npy'))[0] for i in range(1,6)]
        data_Q = [np.load(data_files[i]) for i in range(5)]
        data_Q = np.concatenate(data_Q,axis=2)
        
        data = {}
        data['Q'] = data_Q
        data['t'] = np.load(glob.glob(os.path.join(os.getcwd(), "./data/fkpp/total_fkpp_t.npy"))[0])
        data['x'] = np.load(glob.glob(os.path.join(os.getcwd(), './data/fkpp/total_fkpp_x.npy'))[0])
        data['y'] = np.load(glob.glob(os.path.join(os.getcwd(), './data/fkpp/total_fkpp_y.npy'))[0])
        
        # num_samples = 2001//step ## 2000 ##
        # split_ratio = .75   
    
    if data_name == 'lcd':        
        data_files = [glob.glob(os.path.join(os.getcwd(), f'./data/lcd/total_lcd_{i}.npy'))[0] for i in range(1,6)]
        data_Q = [np.load(data_files[i]) for i in range(5)]
        data_Q = np.concatenate(data_Q,axis=2)
        
        data = {}
        data['Q'] = data_Q
        data['t'] = np.load(glob.glob(os.path.join(os.getcwd(), "./data/lcd/total_lcd_t.npy"))[0])
        data['x'] = np.load(glob.glob(os.path.join(os.getcwd(), './data/lcd/total_lcd_x.npy'))[0])
        data['y'] = np.load(glob.glob(os.path.join(os.getcwd(), './data/lcd/total_lcd_y.npy'))[0])
        
        # num_samples = 2001//step ## 2000 ##
        # split_ratio = .75   
        

    Q_original, t = data['Q'], data['t']
    Q_original = Q_original.reshape(-1, Q_original.shape[-1])
    Q_original, t = Q_original[:, ::step], t[::step] #subsample snapshots
    num_samples = Q_original.shape[1]

    Q_original_train, t_train, Q_original_valid, t_valid, Q_original_test, t_test = \
        get_train_test_data(Q_original, t, split_ratio=split_ratio, split_ratio_validation=split_ratio_validation)

    Q_original_noised = add_noise(Q_original, percentage=noise_level, method="std")
    Q_train, t_train, Q_valid, t_valid, Q_test, t_test = \
        get_train_test_data(Q_original_noised, t, split_ratio=split_ratio, split_ratio_validation=split_ratio_validation)
    # dt = t_train[1] - t_train[0]

    return Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, num_samples

def get_smoothed(Q_train_, t_train):
    ### smoother
    # smoother = False
    # if smoother:
    Q_s, _, smoothed = smooth(Q_train_, t_train, window_size=None, poly_order=3)
    # Q_train_, _ = smooth(Q_train_, t_train, window_size=None, poly_order=3)
    
    if smoothed:
        resid = Q_s - Q_train_
        var = np.var(resid, axis=1) + 1e-8
    else:
        var = 1
    # else:
    #     var = 1
    return Q_s, var

def get_weights(r, svdvals, var):
    ### 权重，特征值越小噪音越大，则其权重越小
    # weights = svdvals[:r]
    weights = svdvals[:r]/(var+1e-8)
    # weights = svdvals[:r]**2/var

    weights = weights/weights.sum()
    return weights
    
def operator_inference(Q_train, t_train, Q_valid, t_valid, Q_original_test, t_test, r, opinf_use_val=True, smoother=True, weighted=True):
    ### Snapshot data Q = [q(t_0) q(t_1) ... q(t_k)], size=(r,k)
    ### reduce data order to r
    Q_train_, Q_valid_, Q_test_, svdvals = model_reducer(Q_train, Q_valid, Q_original_test, r)
    
    if smoother:
        Q_s, var = get_smoothed(Q_train_, t_train)
    else:
        Q_s = Q_train_
        var = 1
    
    print('var: ', var)
    print('svdvals:', svdvals[:r])
    if weighted:
        weights = get_weights(r, svdvals, var)
    else:
        weights = np.ones(r)
        
    # split_ratio_validation = .1  ## if this is 0, then it means opinf choose model based on train dataset
    ### select best A_opinf, H_opinf by grid search ####
    A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
        optimal_opinf(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
    
    ##### result by order='ord2'
    A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
        optimal_opinf(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)

    #### ord2/6
    if loss_min_6<=loss_min_2:
        A_opinf = A_opinf_6
        H_opinf = H_opinf_6
        
        regularizer = regularizer_6
        par_tsvd = par_tsvd_6
        order = 'ord6'
    
    else:
        A_opinf = A_opinf_2
        H_opinf = H_opinf_2
        
        regularizer = regularizer_2
        par_tsvd = par_tsvd_2
        order = 'ord2'
    
        
    return A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
        Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order, var, svdvals[:r]


def main(data_name, r, noise_level, step, smoother, pieces, reg_Frobenius=0, \
          weighted=False, max_iter=10, split_ratio=.75, split_ratio_validation=.1, opinf_use_val=True, name_suffix=None, save_results=True):
    ### get data ###
    Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, num_samples = \
                                        data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

    ### A and H by operator inference under reduced order dataset
    A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
        Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order, var, svdvals = \
        operator_inference(Q_train, t_train, Q_valid, t_valid, Q_original_test, t_test, r, opinf_use_val, smoother=smoother, weighted=weighted)


        
    return weights, var, svdvals

    
    
if __name__ == "__main__": 
    
    ###### config #####
    max_iter = 10
    # ###Perform piecewise integration and optimization; if it is a list, then divide it into segments in order and optimize accordingly.
    pieces = [3,2,3]  # [1] #  [5,1,5] # 
    split_ratio_validation = .1
    smoother = True # False
    
    save_results = False # False #
    

    for opinf_use_val in [True]:#[True, False]:
    # for opinf_use_val in [False]:

        for data_name in ['burgers', 'fkpp', 'lcd']:
        # for data_name in ['lcd']:
            weights_l, var_l, svdvals_l = [],[],[]
            
            if data_name=='fkpp':
                step = 1 ## 1, 2, 4, 10
                num_samples = 2001//step ## 2000 ##
                split_ratio = .75
                r_list = range(1,6)
                
            if data_name=='burgers':
                step = 1 # 1 # 10 # 100 # 500 # 
                num_samples = 10000//step # 10000
                split_ratio = .5
                r_list = range(1,6)
                
            if data_name=='lcd':
                step = 1 ## 1, 2, 4, 10
                num_samples = 2001//step ## 2000 ##
                split_ratio = .75
                r_list = range(1,16)
                
            assert split_ratio + split_ratio_validation < 1, 'percentage of train and validation data is more than 1!!'
            
            ratio = str(split_ratio).replace('.','p')
            
            noise_level_list = [0, 40, 80, 120, 160, 200]
            for noise_level in noise_level_list:
            # for noise_level in [40,200]:
                
                name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'

                # ### get data ###
                # Q_train, t_train, Q_test, t_test, Q_original_train, Q_original_test, num_samples = data_loader(data_name, step, split_ratio)
                
                error_opinf_6_init_list, error_adjoint_init_list, error_opinf_2_init_list = [], [], []
                error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = [], [], []
                error_opinf_6_list, error_adjoint_list, error_opinf_2_list = [], [], []
                error_opinf_6_valid_list, error_adjoint_valid_list, error_opinf_2_valid_list = [], [], []
                
                reg_best, weighted_best = [], []
                # for r in r_list:
                for r in [5]:
                    print(f'dimension: {r}')
                    
                    # ### A and H by operator inference under reduced order dataset
                    # A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, Q_train_, svdvals, regularizer, par_tsvd, order = \
                    #     operator_inference(Q_train, t_train, Q_test, t_test, r, split_ratio_validation, opinf_use_val)
                    
                    #### find the best reg_Frobenius value ###
                    reg_Frobenius_list = [0]
                    weighted_list = [True]
                    
                    # reg_Frobenius_list, weighted_list = [0,0], [True,False]
                    choose_reg = []
                    for reg_Frobenius, weighted in zip(reg_Frobenius_list, weighted_list):
                        print(f'noise: {noise_level}, dimension: {r}')
                        print(f'reg_Frobenius: {reg_Frobenius}, weighted: {weighted}')
                        
                        weights,var, svdvals= main(data_name, r, noise_level, step, smoother, pieces, reg_Frobenius, \
                                weighted, max_iter, split_ratio, split_ratio_validation, opinf_use_val, name_suffix, save_results=False)
                        
                        weights_l.append(weights)
                        var_l.append(var)
                        svdvals_l.append(svdvals)
                        
    
            fig, axes = plt.subplots(1,3,figsize=[12,4])         
            axes[0].plot(var_l[1],marker='*',linestyle='--', label='NL=40%')
            axes[0].plot(var_l[2],marker='*',linestyle='--', label='NL=80%')
            axes[0].plot(var_l[3],marker='*',linestyle='--', label='NL=120%')
            axes[0].plot(var_l[4],marker='*',linestyle='--', label='NL=160%')
            axes[0].plot(var_l[5],marker='*',linestyle='--', label='NL=200%')
            # axes[0].set_xticks(np.arange(0,15,2), np.arange(0,15,2)+1)
            axes[0].set_xticks(np.arange(0,5,1), np.arange(0,5,1)+1)
            axes[0].set_xlabel(r'POD mode index $i$', fontsize='large')
            axes[0].set_ylabel(r'Estimated noise variance $\nu_i^2$', fontsize='large')
            axes[0].legend()
            
            axes[1].plot(svdvals_l[1],marker='*',linestyle='--', label='NL=40%')
            axes[1].plot(svdvals_l[2],marker='*',linestyle='--', label='NL=80%')
            axes[1].plot(svdvals_l[3],marker='*',linestyle='--', label='NL=120%')
            axes[1].plot(svdvals_l[4],marker='*',linestyle='--', label='NL=160%')
            axes[1].plot(svdvals_l[5],marker='*',linestyle='--', label='NL=200%')
            # axes[1].set_xticks(np.arange(0,15,2), np.arange(0,15,2)+1)
            axes[1].set_xticks(np.arange(0,5,1), np.arange(0,5,1)+1)
            axes[1].set_xlabel(r'POD mode index $i$', fontsize='large')
            axes[1].set_ylabel(r'Normalized singular value $\sigma_i/\sigma_1$', fontsize='large')
            axes[1].set_yscale('log')
            axes[1].legend()

            axes[2].plot(weights_l[1],marker='*',linestyle='--', label='NL=40%')
            axes[2].plot(weights_l[2],marker='*',linestyle='--', label='NL=80%')
            axes[2].plot(weights_l[3],marker='*',linestyle='--', label='NL=120%')
            axes[2].plot(weights_l[4],marker='*',linestyle='--', label='NL=160%')
            axes[2].plot(weights_l[5],marker='*',linestyle='--', label='NL=200%')
            # axes[2].set_xticks(np.arange(0,15,2), np.arange(0,15,2)+1)
            axes[2].set_xticks(np.arange(0,5,1), np.arange(0,5,1)+1)
            axes[2].set_xlabel(r'POD mode index $i$', fontsize='large')
            axes[2].set_ylabel(r'Normalized weight $\omega_i$', fontsize='large')
            axes[2].set_yscale('log')
            axes[2].legend()

            # plt.subplots_adjust(hspace=0.1, wspace=0.1)
            plt.tight_layout()
            plt.savefig(f'./figures_analysis/plot_W_{data_name}.png',bbox_inches='tight',dpi=650)
            plt.show()
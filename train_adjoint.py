#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:33:53 2025

@author: dliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:30:17 2025

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

def main(data_name, r, noise_level, step, smoother=False, pieces=[2], reg_Frobenius=0, weighted=False, max_iter=10, split_ratio=.75):
    
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
        
    
    ### select best A_opinf, H_opinf by grid search ####
    ### TruncatedSVDSolver for L2T is critical  ########
    # A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
    #     optimal_opinf(weights[:,None]*Q__, t_train, t_test, 'ord6', M=np.max(np.abs(Q__))*10, T=t[-1])
    A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
        optimal_opinf(Q__, t_train, t_test, 'ord6', M=np.max(np.abs(Q__))*10, T=t[-1])
    
    ##### result by order='ord2'
    A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
        optimal_opinf(Q__, t_train, t_test, 'ord2', M=np.max(np.abs(Q__))*10, T=t[-1])
    
    # A_opinf = A_opinf_6
    # H_opinf = H_opinf_6
    
    # regularizer = regularizer_6
    # par_tsvd = par_tsvd_6
    # order = 'ord6'
    
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
    
    ### initial guess for A and H from operator inference
    theta = np.concatenate([A_opinf.ravel(), H_opinf.ravel()])
    # theta = np.random.rand(r**2+r**3)*.1
    
        
    loss_boundary = 20000 # np.inf # 
    for piece in pieces: # [750]:#
    # for piece in reversed(range(5,6)): # [750]:#
        # piece = 5
        split_a = [int(k_samples//piece)*i for i in range(piece)]
        split_b = [int(k_samples//piece)*(i+1) for i in range(piece)]
        split_b.pop()
        split_b.append(k_samples)
        
        for l in range(piece):
            Q_ = Q__[:, split_a[l]:split_b[l]]
            t = t_train[split_a[l]:split_b[l]]
        
            k_samples_ = Q_.shape[1]
            ############################ GD Parameters ###########################
            ######################################################################
            
            # max_iter = 10
            epsilon = 1e-8   # stopping threshold for gradient norm
            
            # Armijo parameters:
            eta = 1e-3       # initial learning rate 
            alpha = 1e-4
            beta = 0.5
            d = r**2 + r**3
            
            ############################### GD Loop ##############################
            ######################################################################
            
            if smoother and noise_level>0:
                Q_s_ = Q_s[:, split_a[l]:split_b[l]]
                q0 = Q_s_[:, 0]
            else:
                q0 = Q_[:, 0]
            
            # loss_new = -np.inf
            for j in range(max_iter):
                theta_old = theta.copy()
                
                A = theta[:r**2].reshape(r, r)
                H = theta[r**2:].reshape(r, r**2)
                
                ### Forward, Compute predicted states
                # tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
                tilde_Q = ode_solver(func_surrogate, q0, t, par=(A, H), method='BDF', rescale=True)
                
                # Loss computation: mean squared error.
                loss = np.mean(np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0))
                print(f"Iteration {j}, Loss: {loss:.6f}")
                
                if loss>loss_boundary:
                    print("Loss too large, reverting theta to previous value and breaking loop.")
                    theta = theta_old
                    break
                
                #### Frobenius 正则 约束A不要太大，间接约束其特征值 ####
                if reg_Frobenius>0:
                    # reg_Frobenius = 1e1
                    reg_loss = reg_Frobenius*np.linalg.norm(A, 'fro')**2 + reg_Frobenius*np.linalg.norm(H, 'fro')**2
                    loss = loss + reg_loss
                ########################################################
                
            
                
                
                ##### Solve adjoint ODE backwards #####
                error = 2*weights[:, None]*(tilde_Q-Q_)  ## Q_ - tilde_Q ## 
                error = error[:,::-1]
                
                s = t[-1]-t
                s = s[::-1]
                error_interp = interp1d(s, error.T, axis=0, kind="linear", fill_value="extrapolate")
                # Initial condition
                lambda_T = np.zeros(r)
                lambda_values = ode_solver(func_lambda, lambda_T, s, par=(A, H, error_interp), method='BDF', rescale=True)
                lambda_values = lambda_values[:,::-1]
                
                ##### Gradient computation. #####
                grad_A = np.zeros(r**2)
                grad_H = np.zeros(r**3)
                
                for k in range(k_samples_):
                    lambda_k = lambda_values[:, k]
                    q_k = Q_[:, k]
                    
                    # Gradient parts for A.
                    outer_A = np.outer(lambda_k, q_k).flatten()
                    grad_A += outer_A * dt
                    
                    # Gradient parts for H.
                    q_outer = np.outer(q_k, q_k).flatten()
                    outer_H = np.outer(lambda_k, q_outer).flatten()
                    grad_H += outer_H * dt
                    
                if reg_Frobenius>0:
                    grad_A += 2 * reg_Frobenius * A.flatten()
                    grad_H += 2 * reg_Frobenius * H.flatten()
                    
                gradient = np.concatenate([grad_A, grad_H])
                grad_norm = np.linalg.norm(gradient)
                
                if grad_norm < epsilon:
                    print("Gradient norm below tolerance; stopping descent.")
                    break
                
                ##### Armijo backtracking line search #####
                eta_current = eta * 1.05  # Initial trial step size
                ls_success = False
                theta_ls_old = theta.copy()   # line search 前的备份
                
                for _ in range(50):  # Max line search iterations
                    theta_new = theta - eta_current * gradient
                    A = theta_new[:r**2].reshape(r, r)
                    H = theta_new[r**2:].reshape(r, r**2)
                    # tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
                    tilde_Q = ode_solver(func_surrogate, q0, t, par=(A, H), method='BDF', rescale=True)
                    loss_new = np.mean(np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0))
                    
                    # --- 检查是否爆炸 ---
                    if loss_new > loss_boundary:
                        theta = theta_ls_old  # 回退
                        print("Line search loss too large, reverting theta and breaking.")
                        break

                    if loss_new <= loss - alpha * eta_current * (grad_norm ** 2):
                        eta = eta_current
                        theta = theta_new
                        ls_success = True
                        break
                    else:
                        eta_current *= beta
                    
                if not ls_success and loss_new <= loss_boundary:
                    theta_new = theta - eta * gradient  # Fallback to previous eta
                    A = theta_new[:r**2].reshape(r, r)
                    H = theta_new[r**2:].reshape(r, r**2)
                    # tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
                    tilde_Q = ode_solver(func_surrogate, q0, t, par=(A, H), method='BDF', rescale=True)
                    loss_new = np.mean(np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0))
                    
                    theta = theta_new
                    if loss_new > loss:
                        eta *= beta  # Force reduce eta if line search failed
                    
                    # if loss_new > loss:
                    #     eta *= beta # Force reduce eta if line search failed
                    #     print("Fallback step failed, reverting theta and reducing eta.")
                    # else:
                    #     theta = theta_new
                        
                        
                # if loss_new >= loss - 1e-6:
                #     break
    
    # Detach optimal A and H from theta.
    A_opt = theta[:r**2].reshape(r, r)
    H_opt = theta[r**2:].reshape(r, r**2)
    
    
    reg = str(reg_Frobenius).replace('.','p')
    ### save result (theta)
    if save_results:
        file_opinf = f"./results/theta_opinf_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_dim{r}_{weighted}_{order}_{regularizer}_{abs(par_tsvd)}_reg{reg}_iter{max_iter}_weighted{weighted}.npz"
        file_adjoint = f"./results/theta_adjoint_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_dim{r}_{weighted}_{order}_{regularizer}_{abs(par_tsvd)}_reg{reg}_iter{max_iter}_weighted{weighted}.npz"
        
        np.savez(file_opinf, A_opinf=A_opinf, H_opinf=H_opinf)
        np.savez(file_adjoint, A_opt=A_opt, H_opt=H_opt)
    
        # theta_opinf = np.load(file_opinf)
        # A_opinf, H_opinf = theta_opinf['A_opinf'], theta_opinf['H_opinf']
        # theta_opt = np.load(file_adjoint)
        # A_opt, H_opt = theta_opt['A_opt'], theta_opt['H_opt']
    
    
    
    
    
    
    
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
    Q_adjoint = ode_solver(func_surrogate, Q_0, t_all, par=(A_opt, H_opt), rescale=True)
    Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_all, par=(A_opinf_2, H_opinf_2), rescale=True)

    error_opinf_6_init_valid = np.mean((Q_val_.T - Q_opinf_6[:,train_idx:valid_idx].T)**2)/np.mean(Q_val_.T**2)
    error_adjoint_init_valid = np.mean((Q_val_.T - Q_adjoint[:,train_idx:valid_idx].T)**2)/np.mean(Q_val_.T**2)
    error_opinf_2_init_valid = np.mean((Q_val_.T - Q_opinf_2[:,train_idx:valid_idx].T)**2)/np.mean(Q_val_.T**2)
    
    error_opinf_6_init_test = np.mean((Q_test_.T - Q_opinf_6[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
    error_adjoint_init_test = np.mean((Q_test_.T - Q_adjoint[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
    error_opinf_2_init_test = np.mean((Q_test_.T - Q_opinf_2[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
    
    axes[0,0].plot(t_all, Q_all_.T, label='true')
    axes[0,0].plot(t_all, Q_opinf_6.T, '--')
    axes[0,0].axvline(x=t_all[train_idx], ls='--')
    axes[0,0].axvline(x=t_all[valid_idx], ls='--')
    axes[0,0].title.set_text(f'opinf_6 vs true: {np.log10(error_opinf_6_init_test):.3} val: {np.log10(error_opinf_6_init_valid):.3}')
    axes[0,1].plot(t_all, Q_all_.T, label='true')
    axes[0,1].plot(t_all, Q_adjoint.T, '--')
    axes[0,1].axvline(x=t_all[train_idx], ls='--')
    axes[0,1].axvline(x=t_all[valid_idx], ls='--')
    axes[0,1].title.set_text(f'adjoint vs true: {np.log10(error_adjoint_init_test):.3} val: {np.log10(error_adjoint_init_valid):.3}')
    axes[0,2].plot(t_all, Q_all_.T, label='true')
    axes[0,2].plot(t_all, Q_opinf_2.T, '--')
    axes[0,2].axvline(x=t_all[train_idx], ls='--')
    axes[0,2].axvline(x=t_all[valid_idx], ls='--')
    axes[0,2].title.set_text(f'opinf_2 vs true: {np.log10(error_opinf_2_init_test):.3} val: {np.log10(error_opinf_2_init_valid):.3}')
    
    
    
    Q_0 = Q_val_[:,0]
    Q_opinf_6_val = ode_solver(func_surrogate, Q_0, t_val, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_adjoint_val = ode_solver(func_surrogate, Q_0, t_val, par=(A_opt, H_opt), rescale=True)
    Q_opinf_2_val = ode_solver(func_surrogate, Q_0, t_val, par=(A_opinf_2, H_opinf_2), rescale=True)
    error_opinf_6_valid = np.mean((Q_val_.T - Q_opinf_6_val.T)**2)/np.mean(Q_val_.T**2)
    error_adjoint_valid = np.mean((Q_val_.T - Q_adjoint_val.T)**2)/np.mean(Q_val_.T**2)
    error_opinf_2_valid = np.mean((Q_val_.T - Q_opinf_2_val.T)**2)/np.mean(Q_val_.T**2)
    
    Q_0 = Q_[:,0]
    Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_adjoint = ode_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt), rescale=True)
    Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2), rescale=True)

    error_opinf_6_train = np.mean((Q_.T - Q_opinf_6.T)**2)/np.mean(Q_.T**2)
    error_adjoint_train = np.mean((Q_.T - Q_adjoint.T)**2)/np.mean(Q_.T**2)
    error_opinf_2_train = np.mean((Q_.T - Q_opinf_2.T)**2)/np.mean(Q_.T**2)
    
    axes[1,0].plot(t_all, Q_all_.T, label='true')
    axes[1,0].plot(t_train, Q_opinf_6.T, '--')
    axes[1,0].axvline(x=t_all[train_idx], ls='--')
    axes[1,0].title.set_text(f'opinf_6 vs true, train: {np.log10(error_opinf_6_train):.3} val: {np.log10(error_opinf_6_valid):.3}')
    axes[1,1].plot(t_all, Q_all_.T, label='true')
    axes[1,1].plot(t_train, Q_adjoint.T, '--')
    axes[1,1].axvline(x=t_all[train_idx], ls='--')
    axes[1,1].title.set_text(f'adjoint vs true, train: {np.log10(error_adjoint_train):.3} val: {np.log10(error_adjoint_valid):.3}')
    axes[1,2].plot(t_all, Q_all_.T, label='true')
    axes[1,2].plot(t_train, Q_opinf_2.T, '--')
    axes[1,2].axvline(x=t_all[train_idx], ls='--')
    axes[1,2].title.set_text(f'opinf_2 vs true, train: {np.log10(error_opinf_2_train):.3} val: {np.log10(error_opinf_2_valid):.3}')
    
    
    axes[1,0].plot(t_val, Q_opinf_6_val.T, '--')
    axes[1,0].axvline(x=t_all[valid_idx], ls='--')
    axes[1,1].plot(t_val, Q_adjoint_val.T, '--')
    axes[1,1].axvline(x=t_all[valid_idx], ls='--')
    axes[1,2].plot(t_val, Q_opinf_2_val.T, '--')
    axes[1,2].axvline(x=t_all[valid_idx], ls='--')


    
    Q_0 = Q_test_[:,0]
    Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_test, par=(A_opinf_6, H_opinf_6), rescale=True)
    Q_adjoint = ode_solver(func_surrogate, Q_0, t_test, par=(A_opt, H_opt), rescale=True)
    Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_test, par=(A_opinf_2, H_opinf_2), rescale=True)
    error_opinf_6 = np.mean((Q_test_.T - Q_opinf_6.T)**2)/np.mean(Q_test_.T**2)
    error_adjoint = np.mean((Q_test_.T - Q_adjoint.T)**2)/np.mean(Q_test_.T**2)
    error_opinf_2 = np.mean((Q_test_.T - Q_opinf_2.T)**2)/np.mean(Q_test_.T**2)
    
    axes[2,0].plot(t_all, Q_all_.T, label='true')
    axes[2,0].plot(t_test, Q_opinf_6.T, '--')
    axes[2,0].axvline(x=t_all[valid_idx], ls='--')
    axes[2,0].title.set_text(f'opinf_6 vs true: {np.log10(error_opinf_6):.3}')
    axes[2,1].plot(t_all, Q_all_.T, label='true')
    axes[2,1].plot(t_test, Q_adjoint.T, '--')
    axes[2,1].axvline(x=t_all[valid_idx], ls='--')
    axes[2,1].title.set_text(f'adjoint vs true: {np.log10(error_adjoint):.3}')
    axes[2,2].plot(t_all, Q_all_.T, label='true')
    axes[2,2].plot(t_test, Q_opinf_2.T, '--')
    axes[2,2].axvline(x=t_all[valid_idx], ls='--')
    axes[2,2].title.set_text(f'opinf_2 vs true: {np.log10(error_opinf_2):.3}')
    
    fig.suptitle(f'reg_Frobenius: {reg_Frobenius}, pieces: {pieces}, weighted: {weighted}')
    
    if save_results:
        fig.savefig(f'./figures/Results_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_dim{r}_{weighted}_{order}_reg{reg}_iter{max_iter}_weighted{weighted}.png')
        plt.close()

    print(f'opinf 6 test error: {error_opinf_6:.6}, val error: {error_opinf_6_valid}')
    print(f'opinf 2 test error: {error_opinf_2:.6}, val error: {error_adjoint_valid}')
    print(f'adjoint test error: {error_adjoint:.6}, val error: {error_opinf_2_valid}')

    print(f'order: {order}')
    print(f'test error (opinf-adjoint): {error_opinf_6 - error_adjoint}')
    

    sol_ode = solve_ivp(func_surrogate, (t_all[0], t_all[-1]), Q_[:,0], t_eval=t_all, args=(A_opt, H_opt), method='BDF') # or "BDF", "RK45", "Radau", "LSODA"

    return Q_, Q_test_, Q_opinf_6, Q_adjoint, Q_opinf_2, t_all, valid_idx, \
        error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
        error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
        error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
        error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
        sol_ode.success
    
    
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

    
    for data_name in ['burgers', 'fkpp']:
    # for data_name in ['fkpp']:
        if data_name=='fkpp':
            step = 1  ## 1, 2, 4, 5, 10, 20, 40
            num_samples = 2001//step ## 2000 ##
            split_ratio = .75
            
        if data_name=='burgers':
            step = 10 # 10 # 20 #  # 500 # 100
            num_samples = 10000//step # 10000
            split_ratio = .5
        
        ratio = str(split_ratio).replace('.','p')
        
        noise_level_list = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        for noise_level in noise_level_list:
        # for noise_level in [0]:
            error_opinf_6_init_list, error_adjoint_init_list, error_opinf_2_init_list = [], [], []
            error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = [], [], []
            error_opinf_6_list, error_adjoint_list, error_opinf_2_list = [], [], []
            error_opinf_6_valid_list, error_adjoint_valid_list, error_opinf_2_valid_list = [], [], []
            
            reg_best, weighted_best = [], []
            for r in range(1,6):
            # for r in [4]:
                print(f'dimension: {r}')
                # ###Perform piecewise integration and optimization; if it is a list, then divide it into segments in order and optimize accordingly.
                # if r == 1:
                #     pieces = [5,1,5]
                # else:
                #     pieces = [3,1,3]
                    
                #### find the best reg_Frobenius value ###
                reg_Frobenius_list = [0, 1e-2, 1e-1, 1e0, 1e1]*2
                weighted_list = [True]*int(len(reg_Frobenius_list)//2) + \
                                [False]*int(len(reg_Frobenius_list)//2)
                
                # reg_Frobenius_list, weighted_list = [5e0], [True]
                choose_reg = []
                for reg_Frobenius, weighted in zip(reg_Frobenius_list, weighted_list):
                    print(f'noise: {noise_level}, dimension: {r}')
                    print(f'reg_Frobenius: {reg_Frobenius}, weighted: {weighted}')
                    
                    Q_, Q_test_, Q_opinf_6, Q_adjoint, Q_opinf_2, t, k_samples, \
                    error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
                    error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
                    error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
                    error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
                    success \
                        = main(data_name, r, noise_level, step, \
                               smoother, pieces, reg_Frobenius, weighted, max_iter, split_ratio)
                    
                    # 判断是否快速下降或上升（积分爆炸）
                    if not success or np.abs(Q_adjoint[:,1:]-Q_adjoint[:,:-1]).max()>10:
                        choose_reg.append(np.inf)
                    else:
                        # choose_reg.append(error_adjoint_train)
                        choose_reg.append(error_adjoint_valid)
                        
                idx_ = np.argmin(choose_reg)
                reg_Frobenius = reg_Frobenius_list[idx_]
                weighted = weighted_list[idx_]
                
                reg_best.append(reg_Frobenius)
                weighted_best.append(weighted)
                ##########################################
                
                Q_, Q_test_, Q_opinf_6, Q_adjoint, Q_opinf_2, t, k_samples, \
                error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
                error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
                error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
                error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
                success \
                    = main(data_name, r, noise_level, step, \
                           smoother, pieces, reg_Frobenius, weighted, max_iter, split_ratio)
                
                
                error_state_opinf_6 = Q_test_.T - Q_opinf_6.T
                error_state_adjoint = Q_test_.T - Q_adjoint.T
                error_state_opinf_2 = Q_test_.T - Q_opinf_2.T
                
                error_opinf_6 = np.mean(error_state_opinf_6**2)/np.mean(Q_test_.T**2)
                error_adjoint = np.mean(error_state_adjoint**2)/np.mean(Q_test_.T**2)
                error_opinf_2= np.mean(error_state_opinf_2**2)/np.mean(Q_test_.T**2)
        
                error_opinf_6_list.append(error_opinf_6)
                error_adjoint_list.append(error_adjoint)
                error_opinf_2_list.append(error_opinf_2)
                
                error_opinf_6_init_list.append(error_opinf_6_init_test)
                error_adjoint_init_list.append(error_adjoint_init_test)
                error_opinf_2_init_list.append(error_opinf_2_init_test)
                
                error_opinf_6_train_list.append(error_opinf_6_train)
                error_adjoint_train_list.append(error_adjoint_train)
                error_opinf_2_train_list.append(error_opinf_2_train)

                error_opinf_6_valid_list.append(error_opinf_6_valid)
                error_adjoint_valid_list.append(error_adjoint_valid)
                error_opinf_2_valid_list.append(error_opinf_2_valid) 
                
                
            error_opinf_6_list = np.array(error_opinf_6_list)
            error_adjoint_list = np.array(error_adjoint_list)
            error_opinf_2_list = np.array(error_opinf_2_list)
        
            error_opinf_6_init_list = np.array(error_opinf_6_init_list)
            error_adjoint_init_list = np.array(error_adjoint_init_list)
            error_opinf_2_init_list = np.array(error_opinf_2_init_list)
            
            error_opinf_6_train_list = np.array(error_opinf_6_train_list)
            error_adjoint_train_list = np.array(error_adjoint_train_list)
            error_opinf_2_train_list = np.array(error_opinf_2_train_list)

            error_opinf_6_valid_list = np.array(error_opinf_6_valid_list)
            error_adjoint_valid_list = np.array(error_adjoint_valid_list)
            error_opinf_2_valid_list = np.array(error_opinf_2_valid_list)
            
            reg_best = np.array(reg_best)
            weighted_best = np.array(weighted_best)
            
            if save_results:
                np.savez(f"./results/error_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_iter{max_iter}_smooth{smoother}.npz", 
                        error_opinf_6_list=error_opinf_6_list, error_adjoint_list=error_adjoint_list, error_opinf_2_list=error_opinf_2_list,
                        error_opinf_6_init_list=error_opinf_6_init_list, error_adjoint_init_list=error_adjoint_init_list, error_opinf_2_init_list=error_opinf_2_init_list,
                        error_opinf_6_train_list=error_opinf_6_train_list, error_adjoint_train_list=error_adjoint_train_list, error_opinf_2_train_list=error_opinf_2_train_list,
                        error_opinf_6_valid_list=error_opinf_6_valid_list, error_adjoint_valid_list=error_adjoint_valid_list, error_opinf_2_valid_list=error_opinf_2_valid_list,
                        reg_best=reg_best, weighted_best=weighted_best)
            
            
            
            fig, axes = plt.subplots(4,3,figsize=(16,16))
            axes[0,0].plot(np.log10(error_opinf_6_init_list), marker='+', label='opinf_6')
            axes[0,0].plot(np.log10(error_adjoint_init_list), marker='o', label='adjoint')
            # axes[0,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[0,0].set_ylabel(r'test(init) relative error ($log_{10}$)', fontsize='x-large')
            axes[0,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[0,0].legend()
            
            axes[0,1].plot(np.log10(error_opinf_2_init_list), marker='+', label='opinf_2')
            axes[0,1].plot(np.log10(error_adjoint_init_list), marker='o', label='adjoint')
            # axes[0,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[0,1].set_ylabel(r'test(init) relative error ($log_{10}$)', fontsize='x-large')
            axes[0,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[0,1].legend()
            
            axes[0,2].plot(reg_best, 'o', label='reg_best')
            # axes[0,2].set_xlabel('Model Dimension(r)', fontsize='x-large')
            # axes[0,2].set_ylabel(r'regularizer value', fontsize='x-large')
            axes[0,2].set_title('regularizer value')
            
            
            axes[1,0].plot(np.log10(error_opinf_6_train_list), marker='+', label='opinf_6')
            axes[1,0].plot(np.log10(error_adjoint_train_list), marker='o', label='adjoint')
            # axes[1,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[1,0].set_ylabel(r'train relative error ($log_{10}$)', fontsize='x-large')
            # axes[1,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[1,0].legend()
            
            axes[1,1].plot(np.log10(error_opinf_2_train_list), marker='+', label='opinf_2')
            axes[1,1].plot(np.log10(error_adjoint_train_list), marker='o', label='adjoint')
            # axes[1,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[1,1].set_ylabel(r'train relative error ($log_{10}$)', fontsize='x-large')
            # axes[1,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[1,1].legend()
            
            axes[1,2].plot(weighted_best, 'o', label='weighted')
            axes[1,2].set_title('if weighted loss')
            axes[1,2].set_title('weighted best')

            
            
            axes[2,0].plot(np.log10(error_opinf_6_valid_list), marker='+', label='opinf_6')
            axes[2,0].plot(np.log10(error_adjoint_valid_list), marker='o', label='adjoint')
            # axes[2,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[2,0].set_ylabel(r'valid relative error ($log_{10}$)', fontsize='x-large')
            # axes[2,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[2,0].legend()
            
            axes[2,1].plot(np.log10(error_opinf_2_valid_list), marker='+', label='opinf_2')
            axes[2,1].plot(np.log10(error_adjoint_valid_list), marker='o', label='adjoint')
            # axes[2,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[2,1].set_ylabel(r'valid relative error ($log_{10}$)', fontsize='x-large')
            # axes[2,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[2,1].legend()
            

            
            axes[3,0].plot(np.log10(error_opinf_6_list), marker='+', label='opinf_6')
            axes[3,0].plot(np.log10(error_adjoint_list), marker='o', label='adjoint')
            axes[3,0].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[3,0].set_ylabel(r'test relative error ($log_{10}$)', fontsize='x-large')
            # axes[3,0].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[3,0].legend()
            
            axes[3,1].plot(np.log10(error_opinf_2_list), marker='+', label='opinf_2')
            axes[3,1].plot(np.log10(error_adjoint_list), marker='o', label='adjoint')
            axes[3,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
            axes[3,1].set_ylabel(r'test relative error ($log_{10}$)', fontsize='x-large')
            # axes[3,1].set_title(f'{data_name} noise {noise_level} samples {num_samples}')
            axes[3,1].legend()
            
            axes[3,2].plot(t[k_samples:], np.mean(abs(error_state_opinf_6),axis=1), marker='+', label='opinf_6')
            axes[3,2].plot(t[k_samples:], np.mean(abs(error_state_adjoint),axis=1), marker='o', label='adjoint')
            axes[3,2].plot(t[k_samples:], np.mean(abs(error_state_opinf_2),axis=1), marker='x', label='opinf_2')
            axes[3,2].legend()
            
            if save_results:
                fig.savefig(f'./figures/plot_{data_name}_sam{num_samples}_ratio{ratio}_noise{noise_level}_iter{max_iter}_smooth{smoother}.png')
                plt.close()
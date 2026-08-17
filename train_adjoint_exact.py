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
                  get_theta_by_opinf, model_reducer, optimal_opinf, optimal_opinf_rk4, optimal_opinf_euler, \
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
        data_file = glob.glob(os.path.join(os.getcwd(),"./data/burgers/exact_intrusive_operator/exact_operator_reproj_1.npz"))[0]  ###   
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
    if smoother:
        Q_s, _, smoothed = smooth(Q_train_, t_train, window_size=None, poly_order=3)
        # Q_train_, _ = smooth(Q_train_, t_train, window_size=None, poly_order=3)
        
        if smoothed:
            resid = Q_s - Q_train_
            var = np.var(resid, axis=1) + 1e-8
        else:
            var = 1
    else:
        var = 1
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
    # Q_train_, Q_valid_, Q_test_, svdvals = model_reducer(Q_train, Q_valid, Q_original_test, r)
    
    Q_train_, Q_valid_, Q_test_ = Q_train, Q_valid, Q_original_test
    Vr = opinf.basis.PODBasis(num_vectors=r)
    Vr.fit(Q_train_)
    svdvals = Vr.svdvals
    
    if smoother:
        Q_s, var = get_smoothed(Q_train_, t_train)
    else:
        Q_s = Q_train_
        var = 1
        
    if weighted:
        weights = get_weights(r, svdvals, var)
    else:
        weights = np.ones(r)
        
    # split_ratio_validation = .1  ## if this is 0, then it means opinf choose model based on train dataset
    ### select best A_opinf, H_opinf by grid search ####
    if meth == 'continue':
        A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
            optimal_opinf(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
    # A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
    #     optimal_opinf_rk4(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
    if meth == 'euler':
        A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
            optimal_opinf_euler(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
            
        
    ##### result by order='ord2'
    if meth == 'continue':
        A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
            optimal_opinf(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
    # A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
    #     optimal_opinf_rk4(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
    if meth == 'euler':
        A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
            optimal_opinf_euler(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)

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
        Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order


def euler_solver(func, y0, t, par=()):
    """
    Fixed-step explicit Euler solver for RHS with signature

        func(t, y, *par)

    Parameters
    ----------
    func : callable
        Right-hand side function.
    y0 : ndarray, shape (r,)
        Initial condition.
    t : ndarray, shape (K,)
        Prescribed time grid.
    par : tuple
        Additional parameters passed to func.

    Returns
    -------
    Y : ndarray, shape (r, K)
        State trajectory.
    """

    y0 = np.asarray(y0, dtype=float).reshape(-1)

    r = y0.size
    K = len(t)

    if K < 1:
        raise ValueError("The time grid must contain at least one point.")

    if K > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("The time grid must be strictly increasing.")

    Y = np.zeros((r, K), dtype=float)
    Y[:, 0] = y0

    y = y0.copy()

    for k in range(K - 1):
        tk = t[k]
        dt_k = t[k + 1] - t[k]

        dydt = np.asarray(
            func(tk, y, *par),
            dtype=float,
        ).reshape(-1)

        if dydt.shape != y.shape:
            raise ValueError(
                f"RHS shape {dydt.shape} does not match "
                f"state shape {y.shape} at step {k}."
            )

        y = y + dt_k * dydt

        if not np.all(np.isfinite(y)):
            raise RuntimeError(
                f"Euler solution became non-finite at step {k}."
            )

        Y[:, k + 1] = y

    return Y


def rk4_solver(func, y0, t, par=()):
    """
    Fixed-step RK4 solver for RHS with signature

        func(t, y, *args)

    Returns
    -------
    Y : ndarray, shape (r, K)
    """

    y0 = np.asarray(y0, dtype=float)
    r = y0.size
    K = len(t)

    Y = np.zeros((r, K))
    Y[:, 0] = y0

    y = y0.copy()

    for k in range(K - 1):
        tk = t[k]
        dt_k = t[k + 1] - t[k]

        k1 = func(tk, y, *par)
        k2 = func(tk + 0.5 * dt_k, y + 0.5 * dt_k * k1, *par)
        k3 = func(tk + 0.5 * dt_k, y + 0.5 * dt_k * k2, *par)
        k4 = func(tk + dt_k, y + dt_k * k3, *par)

        y = y + dt_k * (k1 + 2*k2 + 2*k3 + k4) / 6.0

        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"RK4 solution blew up at step {k}")

        Y[:, k + 1] = y

    return Y

def optimize_by_adjoint(A_opinf, H_opinf, Q_train_, t_train, Q_s, weights, pieces=[2], reg_Frobenius=0, max_iter=10):
    k_samples = Q_train_.shape[1]  # number of samples for training(snapshot data)
    r = Q_train_.shape[0]
    
    ### initial guess for A and H from operator inference
    theta = np.concatenate([A_opinf.ravel(), H_opinf.ravel()])
    # theta = np.random.rand(r**2+r**3)*.1

    dt = t_train[1]-t_train[0]

    # loss_boundary = np.inf # 30000 # 
    for piece in pieces: # [750]:#
    # for piece in reversed(range(5,6)): # [750]:#
        # piece = 5
        split_a = [int(k_samples//piece)*i for i in range(piece)]
        split_b = [int(k_samples//piece)*(i+1) for i in range(piece)]
        split_b.pop()
        split_b.append(k_samples)
        
        for l in range(piece):
            Q_ = Q_train_[:, split_a[l]:split_b[l]]
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
            # d = r**2 + r**3
            
            
            ############################### GD Loop ##############################
            ######################################################################
            # if smoothed:
            Q_s_ = Q_s[:, split_a[l]:split_b[l]]
            q0 = Q_s_[:, 0]
            # else:
            #     q0 = Q_[:, 0]
            
            # loss_new = -np.inf
            for j in range(max_iter):
                theta_old = theta.copy()
                
                A = theta[:r**2].reshape(r, r)
                H = theta[r**2:].reshape(r, r**2)
                
                ### Forward, Compute predicted states
                # tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
                if meth == 'continue':
                    tilde_Q = ode_solver(func_surrogate, q0, t, par=(A, H), method='BDF', rescale=True)
                # tilde_Q = rk4_solver(func_surrogate, q0, t, par=(A, H))
                if meth == 'euler':
                    tilde_Q = euler_solver(func_surrogate, q0, t, par=(A, H))

                # Loss computation: mean squared error.
                # loss = np.mean(np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0))
                pointwise_loss = np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0)
                loss = np.sum(pointwise_loss) * dt

                print(f"Iteration {j}, Loss: {loss:.6f}")
                
                # if loss>loss_boundary:
                #     print("Loss too large, reverting theta to previous value and breaking loop.")
                #     theta = theta_old
                #     break
                
                #### Frobenius 正则 约束A不要太大，间接约束其特征值 ####
                if reg_Frobenius>0:
                    # reg_Frobenius = 1e1
                    reg_loss = reg_Frobenius*np.linalg.norm(A, 'fro')**2 + reg_Frobenius*np.linalg.norm(H, 'fro')**2
                    loss = loss + reg_loss
                ########################################################
                
            
                
                
                ##### Solve adjoint ODE backwards #####
                s = t[-1]-t
                s = s[::-1]
                
                error = 2*weights[:, None]*(tilde_Q-Q_)  
                
                error_rev = error[:, ::-1]
                error_interp = interp1d(s, error_rev.T, axis=0, kind='linear', fill_value='extrapolate')
                
                # Forward trajectory expressed as a function of reversed time s
                q_rev = tilde_Q[:, ::-1]
                q_interp = interp1d(s, q_rev.T, axis=0, kind='linear', fill_value='extrapolate')
                
                # Initial condition
                lambda_T = np.zeros(r)
                if meth == 'continue':
                    # lambda_values = ode_solver(func_lambda, lambda_T, s, \
                    #                             par=(A, H, error_interp), method='BDF', rescale=True)
                    lambda_values = ode_solver(func_lambda, lambda_T, s, \
                                                par=(A, H, [error_interp, q_interp]), method='BDF', rescale=True)
                # lambda_values = rk4_solver(func_lambda, lambda_T, s, par=(A, H, error_interp))
                if meth == 'euler':
                    # lambda_values = euler_solver(func_lambda, lambda_T, s, par=(A, H, error_interp))
                    lambda_values = euler_solver(func_lambda, lambda_T, s, \
                                               par=(A, H, [error_interp, q_interp]))
                lambda_values = lambda_values[:, ::-1]

                ##### Gradient computation. #####
                grad_A = np.zeros(r**2)
                grad_H = np.zeros(r**3)
                
                for k in range(k_samples_):
                    lambda_k = lambda_values[:, k]
                    # q_k = Q_[:, k]
                    q_k = tilde_Q[:, k]
                    
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
                # theta_ls_old = theta.copy()   # line search 前的备份
                
                for _ in range(50):  # Max line search iterations
                    theta_new = theta - eta_current * gradient
                    A = theta_new[:r**2].reshape(r, r)
                    H = theta_new[r**2:].reshape(r, r**2)
                    
                    # tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
                    if meth == 'continue':
                        tilde_Q = ode_solver(func_surrogate, q0, t, par=(A, H), method='BDF', rescale=True)
                    # tilde_Q = rk4_solver(func_surrogate, q0, t, par=(A, H))
                    if meth == 'euler':
                        tilde_Q = euler_solver(func_surrogate, q0, t, par=(A, H))
                    
                    # loss_new = np.mean(np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0))
                    pointwise_loss = np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0)
                    loss_new = np.sum(pointwise_loss) * dt
                    
                    if reg_Frobenius>0:
                        loss_new += (reg_Frobenius * np.linalg.norm(A, 'fro')**2
                                     + reg_Frobenius * np.linalg.norm(H, 'fro')**2)
                        
                    # # --- 检查是否爆炸 ---
                    # if loss_new > loss_boundary:
                    #     theta = theta_ls_old  # 回退
                    #     print("Line search loss too large, reverting theta and breaking.")
                    #     break

                    if loss_new <= loss - alpha * eta_current * (grad_norm ** 2):
                        eta = eta_current
                        theta = theta_new
                        ls_success = True
                        break
                    else:
                        eta_current *= beta
                
                # Do not accept a step that fails Armijo
                if not ls_success:
                    eta *= beta
                    print("Armijo line search failed; keeping current parameters.")
                    break

                # if not ls_success:
                #     theta_new = theta - eta * gradient  # Fallback to previous eta
                #     A = theta_new[:r**2].reshape(r, r)
                #     H = theta_new[r**2:].reshape(r, r**2)
                    
                #     # tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
                #     if meth == 'continue':
                #         tilde_Q = ode_solver(func_surrogate, q0, t, par=(A, H), method='BDF', rescale=True)
                #     # tilde_Q = rk4_solver(func_surrogate, q0, t, par=(A, H))
                #     if meth == 'euler':
                #         tilde_Q = euler_solver(func_surrogate, q0, t, par=(A, H))
                    
                #     # loss_new = np.mean(np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0))
                #     pointwise_loss = np.sum(weights[:, None]*(Q_ - tilde_Q)**2, axis=0)
                #     loss_new = np.sum(pointwise_loss) * dt
                    
                #     if reg_Frobenius>0:
                #         loss_new += (reg_Frobenius * np.linalg.norm(A, 'fro')**2
                #                      + reg_Frobenius * np.linalg.norm(H, 'fro')**2)
                        
                #     theta = theta_new
                #     if loss_new > loss:
                #         eta *= beta  # Force reduce eta if line search failed
                    
                #     # if loss_new > loss:
                #     #     eta *= beta # Force reduce eta if line search failed
                #     #     print("Fallback step failed, reverting theta and reducing eta.")
                #     # else:
                #     #     theta = theta_new
                        
                        
                # if loss_new >= loss - 1e-6:
                #     break
    
    # Detach optimal A and H from theta.
    A_opt = theta[:r**2].reshape(r, r)
    H_opt = theta[r**2:].reshape(r, r**2)
    
    return A_opt, H_opt

# def save_theta(A_opt, H_opt, A_opinf, H_opinf, name_suffix):

#         # np.savez(file_adjoint, A_opt=A_opt, H_opt=H_opt)

#         # theta_opinf = np.load(file_opinf)
#         # A_opinf, H_opinf = theta_opinf['A_opinf'], theta_opinf['H_opinf']
#         # theta_opt = np.load(file_adjoint)
#         # A_opt, H_opt = theta_opt['A_opt'], theta_opt['H_opt']



def predict_and_plot(A_opt, H_opt, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
                     Q_train_, Q_valid_, Q_test_, Q_s, t_train, t_valid, t_test, name_suffix=None, save_results=True):
    
    
    # ############### opinf vs adjoint #############    
    t_all = np.r_[t_train,t_valid,t_test]
    Q_all_ = np.c_[Q_train_, Q_valid_, Q_test_]
    
    
    train_idx = Q_train_.shape[1]
    valid_idx = train_idx + Q_valid_.shape[1]
    
    
    fig, axes = plt.subplots(3,3,figsize=[16,10])
    
    ### all time prediction
    Q_0 = Q_s[:,0]
    if meth == 'continue':
        Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_all, par=(A_opinf_6, H_opinf_6), rescale=True)
        Q_adjoint = ode_solver(func_surrogate, Q_0, t_all, par=(A_opt, H_opt), rescale=True)
        Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_all, par=(A_opinf_2, H_opinf_2), rescale=True)
    # Q_opinf_6 = rk4_solver(func_surrogate, Q_0, t_all, par=(A_opinf_6, H_opinf_6))
    # Q_adjoint = rk4_solver(func_surrogate, Q_0, t_all, par=(A_opt, H_opt))
    # Q_opinf_2 = rk4_solver(func_surrogate, Q_0, t_all, par=(A_opinf_2, H_opinf_2))
    if meth == 'euler':
        Q_opinf_6 = euler_solver(func_surrogate, Q_0, t_all, par=(A_opinf_6, H_opinf_6))
        Q_adjoint = euler_solver(func_surrogate, Q_0, t_all, par=(A_opt, H_opt))
        Q_opinf_2 = euler_solver(func_surrogate, Q_0, t_all, par=(A_opinf_2, H_opinf_2))

    error_opinf_6_init_valid = np.mean((Q_valid_.T - Q_opinf_6[:,train_idx:valid_idx].T)**2)/np.mean(Q_valid_.T**2)
    error_adjoint_init_valid = np.mean((Q_valid_.T - Q_adjoint[:,train_idx:valid_idx].T)**2)/np.mean(Q_valid_.T**2)
    error_opinf_2_init_valid = np.mean((Q_valid_.T - Q_opinf_2[:,train_idx:valid_idx].T)**2)/np.mean(Q_valid_.T**2)
    
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
    
    
    ### valid time period prediction
    Q_0 = Q_valid_[:,0]
    if meth == 'continue':
        Q_opinf_6_val = ode_solver(func_surrogate, Q_0, t_valid, par=(A_opinf_6, H_opinf_6), rescale=True)
        Q_adjoint_val = ode_solver(func_surrogate, Q_0, t_valid, par=(A_opt, H_opt), rescale=True)
        Q_opinf_2_val = ode_solver(func_surrogate, Q_0, t_valid, par=(A_opinf_2, H_opinf_2), rescale=True)
    # Q_opinf_6_val = rk4_solver(func_surrogate, Q_0, t_valid, par=(A_opinf_6, H_opinf_6))
    # Q_adjoint_val = rk4_solver(func_surrogate, Q_0, t_valid, par=(A_opt, H_opt))
    # Q_opinf_2_val = rk4_solver(func_surrogate, Q_0, t_valid, par=(A_opinf_2, H_opinf_2))
    if meth == 'euler':
        Q_opinf_6_val = euler_solver(func_surrogate, Q_0, t_valid, par=(A_opinf_6, H_opinf_6))
        Q_adjoint_val = euler_solver(func_surrogate, Q_0, t_valid, par=(A_opt, H_opt))
        Q_opinf_2_val = euler_solver(func_surrogate, Q_0, t_valid, par=(A_opinf_2, H_opinf_2))
    error_opinf_6_valid = np.mean((Q_valid_.T - Q_opinf_6_val.T)**2)/np.mean(Q_valid_.T**2)
    error_adjoint_valid = np.mean((Q_valid_.T - Q_adjoint_val.T)**2)/np.mean(Q_valid_.T**2)
    error_opinf_2_valid = np.mean((Q_valid_.T - Q_opinf_2_val.T)**2)/np.mean(Q_valid_.T**2)
    
    ### train time period prediction
    Q_0 = Q_s[:,0]
    if meth == 'continue':
        Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6), rescale=True)
        Q_adjoint = ode_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt), rescale=True)
        Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2), rescale=True)
    # Q_opinf_6 = rk4_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6))
    # Q_adjoint = rk4_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt))
    # Q_opinf_2 = rk4_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2))
    if meth == 'euler':
        Q_opinf_6 = euler_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6))
        Q_adjoint = euler_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt))
        Q_opinf_2 = euler_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2))

    error_opinf_6_train = np.mean((Q_train_.T - Q_opinf_6.T)**2)/np.mean(Q_train_.T**2)
    error_adjoint_train = np.mean((Q_train_.T - Q_adjoint.T)**2)/np.mean(Q_train_.T**2)
    error_opinf_2_train = np.mean((Q_train_.T - Q_opinf_2.T)**2)/np.mean(Q_train_.T**2)
    
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
    
    
    axes[1,0].plot(t_valid, Q_opinf_6_val.T, '--')
    axes[1,0].axvline(x=t_all[valid_idx], ls='--')
    axes[1,1].plot(t_valid, Q_adjoint_val.T, '--')
    axes[1,1].axvline(x=t_all[valid_idx], ls='--')
    axes[1,2].plot(t_valid, Q_opinf_2_val.T, '--')
    axes[1,2].axvline(x=t_all[valid_idx], ls='--')


    print(f'opinf 6 train error: {np.log10(error_opinf_6_train):.6}, val error: {np.log10(error_opinf_6_valid):.6}')
    print(f'opinf 2 train error: {np.log10(error_opinf_2_train):.6}, val error: {np.log10(error_opinf_2_valid):.6}')
    print(f'adjoint train error: {np.log10(error_adjoint_train):.6}, val error: {np.log10(error_adjoint_valid):.6}')


    ### test time period prediction
    Q_0 = Q_test_[:,0]
    if meth == 'continue':
        Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_test, par=(A_opinf_6, H_opinf_6), rescale=True)
        Q_adjoint = ode_solver(func_surrogate, Q_0, t_test, par=(A_opt, H_opt), rescale=True)
        Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_test, par=(A_opinf_2, H_opinf_2), rescale=True)
    # Q_opinf_6 = rk4_solver(func_surrogate, Q_0, t_test, par=(A_opinf_6, H_opinf_6))
    # Q_adjoint = rk4_solver(func_surrogate, Q_0, t_test, par=(A_opt, H_opt))
    # Q_opinf_2 = rk4_solver(func_surrogate, Q_0, t_test, par=(A_opinf_2, H_opinf_2))
    if meth == 'euler':
        Q_opinf_6 = euler_solver(func_surrogate, Q_0, t_test, par=(A_opinf_6, H_opinf_6))
        Q_adjoint = euler_solver(func_surrogate, Q_0, t_test, par=(A_opt, H_opt))
        Q_opinf_2 = euler_solver(func_surrogate, Q_0, t_test, par=(A_opinf_2, H_opinf_2))
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
        fig.savefig(f'./results_exact/Results_exact_{name_suffix}.png')
    
    plt.close()

    # print(f'opinf 6 test error: {np.log10(error_opinf_6):.6}, val error: {np.log10(error_opinf_6_valid):.6}')
    # print(f'opinf 2 test error: {np.log10(error_opinf_2):.6}, val error: {np.log10(error_opinf_2_valid):.6}')
    # print(f'adjoint test error: {np.log10(error_adjoint):.6}, val error: {np.log10(error_adjoint_valid):.6}')

    # print(f'test error (opinf-adjoint): {error_opinf_6 - error_adjoint}')
    
    ### train time period prediction
    Q_0 = Q_s[:,0]
    if meth == 'continue':
        Q_opinf_6 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6), rescale=True)
        Q_adjoint = ode_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt), rescale=True)
        Q_opinf_2 = ode_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2), rescale=True)
    # Q_opinf_6 = rk4_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6))
    # Q_adjoint = rk4_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt))
    # Q_opinf_2 = rk4_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2))
    if meth == 'euler':
        Q_opinf_6 = euler_solver(func_surrogate, Q_0, t_train, par=(A_opinf_6, H_opinf_6))
        Q_adjoint = euler_solver(func_surrogate, Q_0, t_train, par=(A_opt, H_opt))
        Q_opinf_2 = euler_solver(func_surrogate, Q_0, t_train, par=(A_opinf_2, H_opinf_2))


    sol_ode = solve_ivp(func_surrogate, (t_all[0], t_all[-1]), Q_s[:,0], t_eval=t_all, args=(A_opt, H_opt), method='BDF') # or "BDF", "RK45", "Radau", "LSODA"
    success = sol_ode.success
    
    return Q_opinf_6, Q_adjoint, Q_opinf_2, \
            error_opinf_6, error_adjoint, error_opinf_2, \
            error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
            error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
            error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
            error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
            success


def symetric_noise(r):
    # Random tensor.
    E_tensor = np.random.randn(r, r, r)

    # Symmetrize the last two indices.
    E_tensor = 0.5 * (
        E_tensor
        + E_tensor.transpose(0, 2, 1)
    )
    return E_tensor

import copy
def main(data_name, r, noise_level, step, smoother, pieces, reg_Frobenius=0, \
         weighted=False, max_iter=10, split_ratio=.75, split_ratio_validation=.1, \
         opinf_use_val=True, name_suffix=None, save_results=True, pertube_level=0):
    ### get data ###
    # Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, num_samples = \
    #                                     data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

    if data_name == 'burgers':      

        ##just for get clean Q_train          
        data_file = glob.glob(os.path.join(os.getcwd(),f"./data/burgers/exact_intrusive_operator/exact_operator_reproj_noise0_{r}.npz"))[0]  ###   
        data = np.load(data_file)
    
        A_r_true, H_r_true, Q_reproj, t = data['A_r_true'], data['H_r_true'], data['Q_reproj'], data['t']
        Q_train, t_train, Q_valid, t_valid, Q_test, t_test, num_samples = \
            Q_reproj[:,:995], t[:995], Q_reproj[:,995:997], t[995:997], Q_reproj[:,997:], t[997:], t.shape[0]
        Q_train_clean = copy.deepcopy(Q_train)
            
        
        ## get Q_train
        data_file = glob.glob(os.path.join(os.getcwd(),f"./data/burgers/exact_intrusive_operator/exact_operator_reproj_noise{noise_level}_{r}.npz"))[0]  ###   
        data = np.load(data_file)
    
        A_r_true, H_r_true, Q_reproj, t = data['A_r_true'], data['H_r_true'], data['Q_reproj'], data['t']
        Q_train, t_train, Q_valid, t_valid, Q_test, t_test, num_samples = \
            Q_reproj[:,:995], t[:995], Q_reproj[:,995:997], t[995:997], Q_reproj[:,997:], t[997:], t.shape[0]
        # Q_train, t_train, Q_valid, t_valid, Q_test, t_test, num_samples = \
        #     Q_reproj[:,:980][:,::5], t[:980][::5], Q_reproj[:,980:990][:,::5], t[980:990][::5], Q_reproj[:,990:][:,::5], t[990:][::5], int(t.shape[0]//5)
        
    ### A and H by operator inference under reduced order dataset
    A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
        Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order = \
        operator_inference(Q_train, t_train, Q_valid, t_valid, Q_test, t_test, r, opinf_use_val, smoother=smoother, weighted=weighted)

    # A_opinf, H_opinf = A_r_true, H_r_true
    
    # # pertube_level = 1e-2 # 1e-6 # 0 #1e-4
    # A_opinf = A_opinf + pertube_level*np.random.randn(*A_opinf.shape)
    # H_opinf = H_opinf + pertube_level*symetric_noise(r).reshape(r,r*r)
    
    rng = np.random.default_rng() 
    E1 = rng.standard_normal(A_opinf.shape)
    E1 *= pertube_level * np.linalg.norm(A_opinf, "fro") / np.linalg.norm(E1, "fro")
    A_opinf = A_opinf + E1
    
    E2 = rng.standard_normal((r,r,r))
    E2 = 0.5*(E2+E2.transpose(0,2,1))
    E2 = E2.reshape(r,r*r)
    E2 *= pertube_level * np.linalg.norm(H_opinf, "fro") / np.linalg.norm(E2, "fro")
    H_opinf = H_opinf + E2
    
    # print(A_opinf)
    ### optimize A and H by adjoint method
    A_opt, H_opt = optimize_by_adjoint(A_opinf, H_opinf, Q_train_, t_train, Q_s, weights=weights,\
                                        pieces=pieces, reg_Frobenius=reg_Frobenius, max_iter=max_iter)  

    ### get errors and save results
    reg = str(reg_Frobenius).replace('.','p')
    per = f"{pertube_level:.0e}".split('e')[-1][-1]
    name_suffix = name_suffix+f'_dim{r}_{order}_{regularizer}_{abs(int(par_tsvd))}_reg{reg}_weighted{weighted}_per{per}'

    Q_opinf_6, Q_adjoint, Q_opinf_2, \
    error_opinf_6, error_adjoint, error_opinf_2, \
    error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
    error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
    error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
    error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, success = \
        predict_and_plot(A_opt, H_opt, A_opinf, H_opinf, A_opinf_2, H_opinf_2, \
                        Q_train_, 
                        # Q_train_clean,
                        Q_valid_, Q_test_, Q_s, t_train, t_valid, t_test, name_suffix=name_suffix, save_results=save_results)
    
    if save_results:
        np.savez(f"./results_exact/theta_adjoint_with_opinf_{name_suffix}.npz", \
                 A_adjoint=A_opt, H_adjoint=H_opt, A_opinf=A_opinf, H_opinf=H_opinf)
        
        np.savez(f'./results_exact/Predictions_{name_suffix}.npz', \
                 Q_train_=Q_train_, Q_valid_=Q_valid_, Q_test_=Q_test_, Q_s=Q_s, \
                 t_train=t_train, t_valid=t_valid, t_test=t_test, \
                 Q_opinf_6=Q_opinf_6, Q_adjoint=Q_adjoint, Q_opinf_2=Q_opinf_2)
        
    return error_opinf_6, error_adjoint, error_opinf_2, \
        error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
        error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
        error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
        error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, success

    
    
if __name__ == "__main__": 
    
    ###### config #####
    max_iter = 20
    # ###Perform piecewise integration and optimization; if it is a list, then divide it into segments in order and optimize accordingly.
    pieces = [3,2,3]  # [1] #  [5,1,5] # 
    split_ratio_validation = .1
    smoother = True # False
    
    save_results = True # False # 
    
    pertube_level = 0#1e-1#1e-4#1e-1#1e-4 # 1e-1 # 1e-2 # 1e-6 # 0 #1e-4
    meth = 'continue' #  'euler' # #'continue ## if perturbe_level is 0 and noise_level is 0, using 'euler' otherwise 'continue' 
    # for opinf_use_val in [True, False]:
    for opinf_use_val in [False]:

        # for data_name in ['burgers', 'fkpp', 'lcd']:
        for data_name in ['burgers']:
            if data_name=='fkpp':
                step = 10 ## 1, 2, 4, 10
                num_samples = 2001//step ## 2000 ##
                split_ratio = .75
                r_list = range(1,6)
                
            if data_name=='burgers':
                step = 10 # 1 # 10 # 100 # 500 # 
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
            # for noise_level in noise_level_list:
            for noise_level in [1,5,10,20]:
                
                name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{max_iter}_smooth{smoother}'

                # ### get data ###
                # Q_train, t_train, Q_test, t_test, Q_original_train, Q_original_test, num_samples = data_loader(data_name, step, split_ratio)
                
                error_opinf_6_init_list, error_adjoint_init_list, error_opinf_2_init_list = [], [], []
                error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = [], [], []
                error_opinf_6_list, error_adjoint_list, error_opinf_2_list = [], [], []
                error_opinf_6_valid_list, error_adjoint_valid_list, error_opinf_2_valid_list = [], [], []
                
                reg_best, weighted_best = [], []
                for r in r_list: #
                # for r in [3]:
                    print(f'dimension: {r}')
                    
                    # ### A and H by operator inference under reduced order dataset
                    # A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, Q_train_, svdvals, regularizer, par_tsvd, order = \
                    #     operator_inference(Q_train, t_train, Q_test, t_test, r, split_ratio_validation, opinf_use_val)
                    
                    # #### find the best reg_Frobenius value ###
                    # reg_Frobenius_list = [0, 1e-2, 1e-1, 1e0, 1e1]*2
                    # weighted_list = [True]*int(len(reg_Frobenius_list)//2) + \
                    #                 [False]*int(len(reg_Frobenius_list)//2)
                    
                    
                    reg_Frobenius_list = [0]
                    weighted_list = [False]
                    
                    
                    # reg_Frobenius_list, weighted_list = [0,0], [True,False]
                    choose_reg = []
                    for reg_Frobenius, weighted in zip(reg_Frobenius_list, weighted_list):
                        print(f'noise: {noise_level}, dimension: {r}')
                        print(f'reg_Frobenius: {reg_Frobenius}, weighted: {weighted}')
                        
                        error_opinf_6, error_adjoint, error_opinf_2, \
                        error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
                        error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
                        error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
                        error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
                        success \
                            = main(data_name, r, noise_level, step, smoother, pieces, reg_Frobenius, \
                                weighted, max_iter, split_ratio, split_ratio_validation, opinf_use_val, 
                                name_suffix, save_results=False, pertube_level=pertube_level)

                        # 判断是否快速下降或上升（积分爆炸）
                        if not success:
                            choose_reg.append(np.inf)
                        else:
                            choose_reg.append(error_adjoint_train)
                            # choose_reg.append(error_adjoint_valid)
                            
                    idx_ = np.argmin(choose_reg)
                    reg_Frobenius = reg_Frobenius_list[idx_]
                    weighted = weighted_list[idx_]
                    
                    reg_best.append(reg_Frobenius)
                    weighted_best.append(weighted)
                    ##########################################
                    

                    choose_seg = []
                    # pieces_list = [[5,5],[3,3],[3,2,3]]
                    pieces_list = [[1]]
                    for pieces in pieces_list:
                        error_opinf_6, error_adjoint, error_opinf_2, \
                        error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
                        error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
                        error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
                        error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
                        success \
                            = main(data_name, r, noise_level, step, smoother, pieces, reg_Frobenius, \
                                weighted, max_iter, split_ratio, split_ratio_validation, opinf_use_val, \
                                name_suffix, save_results=False, pertube_level=pertube_level)
                        
                        choose_seg.append(error_adjoint_train)
                        # choose_seg.append(error_adjoint_valid)
                    
                    pieces = pieces_list[np.argmin(choose_seg)]
                    # pieces = [3,3]

                    error_opinf_6, error_adjoint, error_opinf_2, \
                    error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
                    error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
                    error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
                    error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid, \
                    success \
                        = main(data_name, r, noise_level, step, smoother, pieces, reg_Frobenius, \
                            weighted, max_iter, split_ratio, split_ratio_validation, opinf_use_val, 
                            name_suffix+'_best', save_results=save_results, pertube_level=pertube_level)
            
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
                    per = f"{pertube_level:.0e}".split('e')[-1][-1]
                    np.savez(f"./results_exact/error_{name_suffix}_per{per}.npz", 
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
                axes[0,0].set_title(f'{data_name}, ord6, noise {noise_level} samples {num_samples}')
                axes[0,0].legend()
                
                axes[0,1].plot(np.log10(error_opinf_2_init_list), marker='+', label='opinf_2')
                axes[0,1].plot(np.log10(error_adjoint_init_list), marker='o', label='adjoint')
                # axes[0,1].set_xlabel('Model Dimension(r)', fontsize='x-large')
                axes[0,1].set_ylabel(r'test(init) relative error ($log_{10}$)', fontsize='x-large')
                axes[0,1].set_title(f'{data_name}, ord2, noise {noise_level} samples {num_samples}')
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
                
                axes[2,1].plot(np.log10(error_opinf_2_valid_list+1e-50), marker='+', label='opinf_2')
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
                
                
                if save_results:
                    per = f"{pertube_level:.0e}".split('e')[-1][-1]
                    fig.savefig(f'./results_exact/plot_exact_{name_suffix}_per{per}.png')
                    
                plt.close()
                
    
    for pertube in [0]:
        for r in r_list:
            theta_path = glob.glob(f'./results_exact/theta_adjoint_*_best_dim{r}*per{pertube}.npz')[0]
            theta_adjoint = np.load(theta_path)
            A_adjoint, H_adjoint, A_opinf, H_opinf = theta_adjoint['A_adjoint'], theta_adjoint['H_adjoint'], theta_adjoint['A_opinf'], theta_adjoint['H_opinf'] 
            
            theta_exact = np.load(f"./data/burgers/exact_intrusive_operator/exact_operator_reproj_noise{0}_{r}.npz")
            A_r_true, H_r_true = theta_exact['A_r_true'], theta_exact['H_r_true']
            
            operator_A_adjoint_error = np.sum((A_adjoint-A_r_true)**2)**.5 / np.sum(A_r_true**2)**.5
            operator_H_adjoint_error = np.sum((H_adjoint-H_r_true)**2)**.5 / np.sum(H_r_true**2)**.5
            operator_A_opinf_error = np.sum((A_opinf-A_r_true)**2)**.5 / np.sum(A_r_true**2)**.5
            operator_H_opinf_error = np.sum((H_opinf-H_r_true)**2)**.5 / np.sum(H_r_true**2)**.5
            print(f'{r} A operator adjoint error: {operator_A_adjoint_error: .4e}, opinf: {operator_A_opinf_error:.4e}')
            print(f'{r} H operator adjoint error: {operator_H_adjoint_error: .4e}, opinf: {operator_H_opinf_error:.4e}')
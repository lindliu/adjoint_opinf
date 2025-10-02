#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:34:27 2025

@author: dliu
"""


import numpy as np
import opinf
from scipy.integrate import solve_ivp

import random
random.seed(10)
np.random.seed(10) 

def integrate(Q_, t, method='spline'):

    if method=='trapezoid':
        from scipy.integrate import cumulative_trapezoid

        integral_ = []
        for k in range(Q_.shape[0]):
            integral_.append(cumulative_trapezoid(Q_[k, :], t, initial=0))
        integral = np.vstack(integral_)
        return integral
    
    if method=='spline':
        from scipy.interpolate import UnivariateSpline

        s = 5  # smoothing factor (tune as needed)
        integral_ = []
        for k in range(Q_.shape[0]):
            spl = UnivariateSpline(t, Q_[k], s=s)
            integral_.append(spl.antiderivative())
        integral = np.vstack([F(t) - F(t[0]) for F in integral_])
        return integral
    
    if method=='savgol+trapezoid':
        from scipy.signal import savgol_filter
        from scipy.integrate import cumulative_trapezoid

        integral_ = []
        for k in range(Q_.shape[0]):
            Q_smooth = savgol_filter(Q_[k, :], window_length=21, polyorder=3)
            integral_.append(cumulative_trapezoid(Q_smooth, t, initial=0))
        integral = np.vstack(integral_)
        return integral


def ode_solver(func, x0, t, par=None, method="BDF", rescale=True):
    if len(par)==2:
        A, H = par
        r = A.shape[0]
        
        sol_ode = solve_ivp(func, (t[0], t[-1]), x0, t_eval=t, args=(A, H),  method=method) # or "BDF", "RK45", "Radau", "LSODA"
        
        if rescale:
            scale = 1
            while not sol_ode.success:
                eigvals_A = np.linalg.eigvals(A)
                eigvals_H = np.linalg.eigvals(H.reshape(r,r,r))
                rho = max(np.max(abs(eigvals_A)), np.max(abs(eigvals_H)))
                alpha = 1 / (rho*scale)  # scale so largest eigenvalue ~ 1
                sol_ode = solve_ivp(func, (t[0], t[-1]), x0, t_eval=t, args=(alpha*A, alpha*H),  method=method) # or "BDF", "RK45", "Radau", "LSODA"
                scale += .5
        
        assert sol_ode.success, print('backward integration failed, '+sol_ode.message)
        
        return sol_ode['y']
    
    if len(par)==3:
        A, H, error = par
        r = A.shape[0]
        
        sol_ode = solve_ivp(func, (t[0], t[-1]), x0, t_eval=t, args=(A, H, error),  method=method) # or "BDF", "RK45", "Radau", "LSODA"
        
        if rescale:
            scale = 1
            while not sol_ode.success:
                eigvals_A = np.linalg.eigvals(A)
                eigvals_H = np.linalg.eigvals(H.reshape(r,r,r))
                rho = max(np.max(abs(eigvals_A)), np.max(abs(eigvals_H)))
                alpha = 1 / (rho*scale)  # scale so largest eigenvalue ~ 1
                sol_ode = solve_ivp(func, (t[0], t[-1]), x0, t_eval=t, args=(alpha*A, alpha*H, error),  method=method) # or "BDF", "RK45", "Radau", "LSODA"
                scale += .1
        
        assert sol_ode.success, print('backward integration failed, '+sol_ode.message)
        
        return sol_ode['y']
    
# def subsample_snapshots(Q_higher, t, step):
#     """Subsample high-resolution snapshots (k=1000) to lower density (k_low)."""
#     return Q_higher[:, ::step], t[::step]

def get_train_test_data(Q, t, split_ratio=.5, noise_level=1, noise_method="std"):
    num_samples = Q.shape[1]
    split = int(num_samples*split_ratio)
    
    Q_train = Q[:, :split]
    Q_test = Q[:, split:]
    
    ### Time vector
    t_train = t[:split]  # Training time
    t_test = t[split:]    # Prediction time
    # dt = t_train[1] - t_train[0]
    
    # np.save(f'./data/Q_train_density_{num_samples}.npy', Q_train)
    # np.save(f'./data/Q_test_density_{num_samples}.npy', Q_test)
    # np.save(f'./data/t_train_density_{num_samples}.npy', t_train)
    # np.save(f'./data/t_test_density_{num_samples}.npy', t_test)

    return Q_train, t_train, Q_test, t_test

def add_noise(Q, percentage, method="max_norm"):
    """
    Add Gaussian noise to a snapshot matrix Q.
    
    Parameters:
        Q (np.ndarray): Input matrix with shape (state_dimension, num_snapshots)
        percentage (float): Noise percentage (e.g., 5 for 5% noise)
        method (str): 
            - "max_norm": Noise scaled by max column norm of Q
            - "std": Noise scaled by standard deviation of Q
    
    Returns:
        Q_noisy (np.ndarray): Noisy version of Q
    """
    
    # Validate input
    if percentage < 0:
        raise ValueError("Percentage must be between 0 and 100")
    
    if method == "max_norm":
        # Calculate maximum column norm
        norms = np.linalg.norm(Q, axis=0)
        scale = np.max(norms) * (percentage / 100)
    elif method == "std":
        # Calculate standard deviation of entire matrix
        scale = np.std(Q) * (percentage / 100)
    else:
        raise ValueError("Invalid method. Choose 'max_norm' or 'std'")
    
    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=scale, size=Q.shape)
    
    # Add noise to original data
    Q_noisy = Q + noise
    
    return Q_noisy

def model_reducer(Q_train, Q_test, r):
    # Initialize a basis.
    #Vr = opinf.basis.PODBasis(cumulative_energy=0.9999)
    Vr = opinf.basis.PODBasis(num_vectors=r)
    
    # Fit the basis (compute Vr) using the snapshot data.
    Vr.fit(Q_train)
    
    # Compress the state snapshots to the reduced space defined by the basis.
    Q_ = Vr.compress(Q_train)
    Q_test_ = Vr.compress(Q_test)
    
    return Q_, Q_test_, Vr.svdvals

    
def func_surrogate(t, x, a, b):
    x = x.reshape(-1,1)
    A_opt = a
    H_opt = b
    dxdt = A_opt@x + H_opt@(np.kron(x,x))
    return dxdt.flatten()



def func_lambda(t, x, A, H, u_interp):
    # H_3d = H.reshape(r, r, r)
    r = H.shape[0]
    H_3d = H.reshape(r,r,r)   ## A.shape: r,r.   H.shape: r,r,r
    M = np.einsum("ijk,j->ki", H_3d, x)
    
    x = x.reshape(-1,1)
    u = u_interp(t).reshape(-1,1)

    dxdt = ((A.T + 2*M) @ x + u)
    return dxdt.flatten()


def smooth(y, t, window_size, poly_order=2, verbose=False):    
    from scipy.signal import savgol_filter
    from statsmodels.tsa.statespace.tools import diff

    y_ = np.zeros([*y.shape])
    # Automatic tunning of the window size 
    if window_size == None: 
        for k in range(y.shape[0]):
            y_norm0 = (y[k,:]-min(y[k,:]))/(max(y[k,:])-min(y[k,:]))
            smoothed_vec_0 = [y_norm0] 
            std_prev = np.std(diff(y_norm0,1))
            window_size_used = 1 
            std1 = [] 
            while True:
                std1.append(std_prev)
                window_size_used += 10
                y_norm0 = savgol_filter(y_norm0, window_size_used, poly_order)
                std_new = np.std(diff(y_norm0,1))
                if verbose: 
                    print('Prev STD: %.5f - New STD: %.5f - Percent change: %.5f' % (std_prev, std_new, 100*(std_new-std_prev)/std_prev))
                if abs((std_new-std_prev)/std_prev) < 0.1: 
                    window_size_used -= 10
                    break
                else:
                    std_prev = std_new
                    smoothed_vec_0.append(y_norm0)
                    y_norm0 = (y[k,:]-min(y[k,:]))/(max(y[k,:])-min(y[k,:]))  
                
            if window_size_used > 1: 
                print('Smoothing window size (dimension 1): '+str(window_size_used),'\n')
                
                y_[k,:] = savgol_filter(y[k,:], window_size_used, poly_order)
            else: 
                print('No smoothing applied')
                print('\n')
                
                return y, t
        
    # Pre-specified window size
    else: 
        for k in range(y.shape[1]):
            y_[k,:] = savgol_filter(y[k,:], window_size, poly_order)
            
            t = t[:len(y)]
    
    return y_, t




def get_theta_by_opinf(Q_, t_, order='ord6', regularizer='L2', par_tsvd=-1, reg_l2=1e-2):
    ### Q_ means reduced Q
    # # Estimate time derivatives using 6th-order finite differences.
    ddt_estimator = opinf.ddt.UniformFiniteDifferencer(t_, order)
    Qdot_ = ddt_estimator.estimate(Q_)[1]
    
    # # Build the quadratic continuous model
    if regularizer == 'no':
        model = opinf.models.ContinuousModel(operators=["A", "H"])
    if regularizer == 'L2':
        D = Q_.T
        Z = Qdot_
        l2solver = opinf.lstsq.L2Solver(regularizer=reg_l2).fit(D, Z)
        model = opinf.models.ContinuousModel(operators=["A", "H"], solver=l2solver)
    if regularizer == 'L2T':
        D = Q_.T
        Z = Qdot_
        tsvdsolver = opinf.lstsq.TruncatedSVDSolver(par_tsvd)  # 0 or -1   # this number is critical
        tsvdsolver.fit(D, Z)
        # tsvdsolver.num_svdmodes = 2
        model = opinf.models.ContinuousModel(operators=["A", "H"], solver=tsvdsolver)

    # # Fit the model
    model.fit(states=Q_, ddts=Qdot_)
    # print(model)
    
    # Operators A and H
    A_opinf = model.operators[0].entries
    H_opinf = model.operators[1].entries
    
    # # Expanded H: size (r,r^2)
    H_opinf = opinf.operators.QuadraticOperator.expand_entries(H_opinf)
    
    # np.save(f"./data/A_opinf_{order}_density_{k_samples}.npy", A_opinf)
    # np.save(f"./data/H_opinf_{order}_density_{k_samples}.npy", H_opinf)
    return A_opinf, H_opinf


def optimal_opinf(Q_, t, t_test, order='ord6', opinf_use_val=True, valid_ratio=0, Q_test_=None, M=100, T=5):
    assert valid_ratio>=0 and valid_ratio<1
    
    ### TruncatedSVDSolver for L2T is critical  ########
    loss_list = []
    rho_list = []
    regularizer_list = ['no','L2','L2','L2','L2T','L2T','L2T','L2T','L2T','L2T','L2T','L2T']
    par_tsvd_list = [0,1e-2,1e-1,1e0,0,-1,-2,-3,-4,-5,-6,-7]
    # regularizer_list = ['no','L2','L2T','L2T','L2T','L2T','L2T','L2T','L2T','L2T']
    # par_tsvd_list = [0,1e-2,0,-1,-2,-3,-4,-5,-6,-7]
    for regularizer_, par_tsvd_ in zip(regularizer_list, par_tsvd_list):
        A_opinf, H_opinf = get_theta_by_opinf(Q_, t, order=order, regularizer=regularizer_, par_tsvd=par_tsvd_, reg_l2=par_tsvd_)
        
        r = A_opinf.shape[0]
        eigvals_A = np.linalg.eigvals(A_opinf)
        eigvals_H = np.linalg.eigvals(H_opinf.reshape(r,r,r))
        rho = max(np.max(abs(eigvals_A)), np.max(abs(eigvals_H)))
        rho_list.append(rho)
        
        if rho>10:#np.log(M)/T:
            loss_list.append(np.inf)
            continue
        
        # try:
        t_all = np.r_[t, t_test]
        ### verify if operator inference works ###
        Q_opinf_ = solve_ivp(func_surrogate, (t_all[0], t_all[-1]), Q_[:,0], \
                            t_eval=t_all, args=(A_opinf, H_opinf),  method='BDF')
        
        if Q_opinf_.success:
            if opinf_use_val==False:
                #### choose model depend on train dataset
                Q_opinf_ = ode_solver(func_surrogate, Q_[:,0], t, par=(A_opinf, H_opinf), rescale=False)
                loss_list.append(np.mean((Q_ - Q_opinf_)**2))
            
            if opinf_use_val==True:
                assert valid_ratio>0 and valid_ratio<1
                assert Q_test_ is not None
                #### choose model depend on validataion dataset
                val_idx = int(t_all.shape[0]*valid_ratio)
                Q_opinf_ = ode_solver(func_surrogate, Q_test_[:,0], t_test[:val_idx], par=(A_opinf, H_opinf), rescale=True)
                loss_list.append(np.mean((Q_test_[:,:val_idx] - Q_opinf_)**2))
        else:
            print('FalseFalseFalseFalseFalse')
            loss_list.append(np.inf)
            
    # print(loss_list)
    if min(loss_list)==np.inf:
        idx = np.argmin(rho_list)
    else:
        idx = np.argmin(loss_list)
    
    loss_min = loss_list[idx]
    regularizer = regularizer_list[idx]
    par_tsvd = par_tsvd_list[idx]
    
    A_opinf, H_opinf = get_theta_by_opinf(Q_, t, order=order, regularizer=regularizer, par_tsvd=par_tsvd, reg_l2=par_tsvd)
    
    
    if regularizer=='L2':
        par_tsvd = np.log10(par_tsvd)
    return A_opinf, H_opinf, regularizer, par_tsvd, loss_min
    
    

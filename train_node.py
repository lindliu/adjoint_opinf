#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:12:30 2025

@author: dliu
"""

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import copy
import os 
import glob
import matplotlib.pyplot as plt
import numpy as np
import opinf
from scipy.interpolate import interp1d
from utils import get_train_test_data, get_theta_by_opinf, add_noise, model_reducer
from utils import optimal_opinf, smooth, optimal_opinf_euler
from train_adjoint import data_loader#, operator_inference

import random
random.seed(10)
np.random.seed(10)    # for numpy random


def symetric_noise(r):
    # Random tensor.
    E_tensor = np.random.randn(r, r, r)

    # Symmetrize the last two indices.
    E_tensor = 0.5 * (
        E_tensor
        + E_tensor.transpose(0, 2, 1)
    )
    return E_tensor

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
    Q_train_, Q_valid_, Q_test_, svdvals = model_reducer(Q_train, Q_valid, Q_original_test, r)
    
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
        Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order


# def operator_inference(Q_train, t_train, Q_valid, t_valid, Q_original_test, t_test, r, opinf_use_val=True, smoother=True, weighted=True):
#     ### Snapshot data Q = [q(t_0) q(t_1) ... q(t_k)], size=(r,k)
#     ### reduce data order to r
#     # Q_train_, Q_valid_, Q_test_, svdvals = model_reducer(Q_train, Q_valid, Q_original_test, r)
    
#     Q_train_, Q_valid_, Q_test_ = Q_train, Q_valid, Q_original_test
#     Vr = opinf.basis.PODBasis(num_vectors=r)
#     Vr.fit(Q_train_)
#     svdvals = Vr.svdvals
    
#     if smoother:
#         Q_s, var = get_smoothed(Q_train_, t_train)
#     else:
#         Q_s = Q_train_
#         var = 1
        
#     if weighted:
#         weights = get_weights(r, svdvals, var)
#     else:
#         weights = np.ones(r)
        
#     # split_ratio_validation = .1  ## if this is 0, then it means opinf choose model based on train dataset
#     ### select best A_opinf, H_opinf by grid search ####
#     A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
#         optimal_opinf(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
#     # A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
#     #     optimal_opinf_rk4(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
#     # A_opinf_6, H_opinf_6, regularizer_6, par_tsvd_6, loss_min_6 = \
#     #     optimal_opinf_euler(Q_train_, t_train, t_valid, t_test, 'ord6', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
        
        
#     ##### result by order='ord2'
#     A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
#         optimal_opinf(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
#     # A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
#     #     optimal_opinf_rk4(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)
#     # A_opinf_2, H_opinf_2, regularizer_2, par_tsvd_2, loss_min_2 = \
#     #     optimal_opinf_euler(Q_train_, t_train, t_valid, t_test, 'ord2', opinf_use_val, Q_valid_=Q_valid_, Q_s=Q_s)#, M=np.max(np.abs(Q_train_))*10, T=t[-1],)

#     #### ord2/6
#     if loss_min_6<=loss_min_2:
#         A_opinf = A_opinf_6
#         H_opinf = H_opinf_6
        
#         regularizer = regularizer_6
#         par_tsvd = par_tsvd_6
#         order = 'ord6'
    
#     else:
#         A_opinf = A_opinf_2
#         H_opinf = H_opinf_2
        
#         regularizer = regularizer_2
#         par_tsvd = par_tsvd_2
#         order = 'ord2'
    
        
#     return A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
#         Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order


import random
random.seed(10)
np.random.seed(10)    # for numpy random
torch.manual_seed(10)

# def get_smoothed(Q_train_, t_train):
#     ### smoother
#     # smoother = False
#     if smoother:
#         Q_s, _, smoothed = smooth(Q_train_, t_train, window_size=None, poly_order=3)
#         # Q_train_, _ = smooth(Q_train_, t_train, window_size=None, poly_order=3)
        
#         if smoothed:
#             resid = Q_s - Q_train_
#             var = np.var(resid, axis=1) + 1e-8
#         else:
#             var = 1
#     else:
#         var = 1
#     return Q_s, var

# Define dynamics
class ODEFunc(torch.nn.Module):
    def __init__(self, r, theta=None):
        super().__init__()
        self.theta = theta
        if theta is None:
            # # Random initialization if no theta provided
            # self.A = torch.nn.Parameter(torch.randn(r, r))
            # self.H = torch.nn.Parameter(torch.randn(r, r * r))
            
            hidden_dim = 20
            self.net = nn.Sequential(
                            nn.Linear(r, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, r))
            
        else:
            # Initialize from theta vector
            A = theta[:r**2].reshape(r, r)
            H = theta[r**2:].reshape(r, r * r)
            self.A = torch.nn.Parameter(A)
            self.H = torch.nn.Parameter(H)

    def forward(self, t, x):

        if self.theta is None:
            dxdt = self.net(x)
            return dxdt
        
        else:
            # Ensure x is column vector
            x = x.view(-1, 1)  # (n,1)
            
            # Kronecker product x⊗x, shape (n^2, 1)
            xx = torch.kron(x, x)

            # Dynamics: A x + H (x⊗x)
            dxdt = self.A @ x + self.H @ xx

            return dxdt.view(-1)  # flatten back to vector



def train_node(Q_train_, t_train, Q_test_, t_test, A_opinf, H_opinf, r, epochs=20, optimizer_name = 'Adam', condition=True):
    ### initial guess for A and H from operator inference
    theta_opinf_ = np.concatenate([A_opinf.ravel(), H_opinf.ravel()])
    
    # data = np.load(f'./data/reduced_{data_name}_{order}_noise{noise_level}_sam{num_samples}.npz')
    # Q_train, Q_test, t_train, t_test, theta_opinf, theta_adjoint = data['Q_train'], data['Q_test'], data['t_train'], data['t_test'], data['theta_opinf'], data['theta_adjoint']
    Q_train_ = torch.tensor(Q_train_,dtype=torch.float32)
    t_train = torch.tensor(t_train,dtype=torch.float32)
    Q_test_ = torch.tensor(Q_test_,dtype=torch.float32)
    t_test = torch.tensor(t_test,dtype=torch.float32)
    theta_opinf = torch.tensor(theta_opinf_,dtype=torch.float32)
    # theta_adjoint_manually = torch.tensor(theta_adjoint,dtype=torch.float32)
    
    torch.manual_seed(10)
    r, k_samples = Q_train_.shape
    if condition:
        func = ODEFunc(r=r, theta=theta_opinf)
    else:
        func = ODEFunc(r=r)
    
    lr = 5e-2
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(func.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        # optimizer = torch.optim.SGD(func.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.SGD(func.parameters(), lr=lr, momentum=0)
    elif optimizer_name == "LBFGS":
        optimizer = torch.optim.LBFGS(func.parameters(), lr=lr, max_iter=20)
    
    # Initial condition & parameters
    y0 = Q_train_[:,0]
    t = t_train
    
    # solution = odeint(func, y0, t, method='euler')  # shape: (T, r)
    # solution = odeint(func, y0, t)  # shape: (T, r)
    
    # print(f'loss : {torch.mean((solution - Q_train_.T)**2)}')
    target = Q_train_.T
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=0.5,
                                                           patience=20)
    patience = 25
    best_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            solution = odeint(func, y0, t)  # shape: (T, r)
            loss = torch.mean((solution - target)**2)
            loss.backward()
            
            
            # nn.utils.clip_grad_norm_(
            #     func.parameters(),
            #     max_norm=1.0
            # )
            
            return loss
        
    
        if optimizer_name != "LBFGS":
            loss = closure() 
            optimizer.step()
        else:
            # optimizer.zero_grad()
            # solution = odeint(func, y0, t)  # shape: (T, r)
            # loss = torch.mean((solution - target)**2)
            # loss.backward()
            
            loss = closure()
            optimizer.step()
            
            scheduler.step(loss.item())
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_no_improve = 0
            
            best_epoch = epoch
            best_model_state = copy.deepcopy(func.state_dict())
            
        else:
            epochs_no_improve += 1


        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.6f}")
            
        
        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best loss = {best_loss:.6e}"
            )
            break
    
        
    
        # ==================================================
        # Restore best model
        # ==================================================
        if best_model_state is not None:
            func.load_state_dict(best_model_state)
    return func


def predict_and_plot(func, r, A_opt, H_opt, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
                     Q_train_, Q_valid_, Q_test_, Q_s, t_train, t_valid, t_test, name_suffix=None, save_results=True):
    
    with torch.no_grad():

    
        # ############### opinf vs adjoint #############    
        t_all = np.r_[t_train,t_valid,t_test]
        t_train = torch.tensor(t_train, dtype=torch.float32)
        t_valid = torch.tensor(t_valid, dtype=torch.float32)
        t_test = torch.tensor(t_test, dtype=torch.float32)
        t_all = torch.tensor(t_all, dtype=torch.float32)
    
        Q_all_ = np.c_[Q_train_, Q_valid_, Q_test_]
        
        train_idx = Q_train_.shape[1]
        valid_idx = train_idx + Q_valid_.shape[1]
        
        # theta_opinf = torch.tensor(np.concatenate([A_opinf.ravel(), H_opinf.ravel()]), dtype=torch.float32)
        # func_opinf = ODEFunc(r=r, theta=theta_opinf)
        theta_opinf_6 = torch.tensor(np.concatenate([A_opinf_6.ravel(), H_opinf_6.ravel()]), dtype=torch.float32)
        func_opinf_6 = ODEFunc(r=r, theta=theta_opinf_6)
        theta_opinf_2 = torch.tensor(np.concatenate([A_opinf_2.ravel(), H_opinf_2.ravel()]), dtype=torch.float32)
        func_opinf_2 = ODEFunc(r=r, theta=theta_opinf_2)
    
        fig, axes = plt.subplots(3,3,figsize=[16,10])
        
        ### all time prediction
        Q_0 = torch.tensor(Q_s[:,0], dtype=torch.float32)
        
        Q_adjoint = odeint(func, Q_0, t_all).detach().cpu().numpy().T  # shape: (T, r)
        Q_opinf_6 = np.random.randn(*Q_adjoint.shape)#Q_opinf_6 = odeint(func_opinf_6, Q_0, t_all).detach().cpu().numpy().T
        Q_opinf_2 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_2, Q_0, t_all).detach().cpu().numpy().T
        
        error_adjoint_init_valid = np.mean((Q_valid_.T - Q_adjoint[:,train_idx:valid_idx].T)**2)/np.mean(Q_valid_.T**2)
        error_opinf_6_init_valid = np.mean((Q_valid_.T - Q_opinf_6[:,train_idx:valid_idx].T)**2)/np.mean(Q_valid_.T**2)
        error_opinf_2_init_valid = np.mean((Q_valid_.T - Q_opinf_2[:,train_idx:valid_idx].T)**2)/np.mean(Q_valid_.T**2)
        
        error_adjoint_init_test = np.mean((Q_test_.T - Q_adjoint[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
        error_opinf_6_init_test = np.mean((Q_test_.T - Q_opinf_6[:,valid_idx:].T)**2)/np.mean(Q_test_.T**2)
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
        Q_0 = torch.tensor(Q_valid_[:,0], dtype=torch.float32)
        Q_adjoint_val = odeint(func, Q_0, t_valid).detach().cpu().numpy().T  # shape: (T, r)
        Q_opinf_6_val = np.random.randn(*Q_adjoint_val.shape)#odeint(func_opinf_6, Q_0, t_valid).detach().cpu().numpy().T
        Q_opinf_2_val = np.random.randn(*Q_adjoint_val.shape)#odeint(func_opinf_2, Q_0, t_valid).detach().cpu().numpy().T
        
        error_adjoint_valid = np.mean((Q_valid_.T - Q_adjoint_val.T)**2)/np.mean(Q_valid_.T**2)
        error_opinf_6_valid = np.mean((Q_valid_.T - Q_opinf_6_val.T)**2)/np.mean(Q_valid_.T**2)
        error_opinf_2_valid = np.mean((Q_valid_.T - Q_opinf_2_val.T)**2)/np.mean(Q_valid_.T**2)
        
        ### train time period prediction
        Q_0 = torch.tensor(Q_s[:,0], dtype=torch.float32)
        Q_adjoint = odeint(func, Q_0, t_train).detach().cpu().numpy().T  # shape: (T, r)
        Q_opinf_6 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_6, Q_0, t_train).detach().cpu().numpy().T
        Q_opinf_2 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_2, Q_0, t_train).detach().cpu().numpy().T
        
        error_adjoint_train = np.mean((Q_train_.T - Q_adjoint.T)**2)/np.mean(Q_train_.T**2)
        error_opinf_6_train = np.mean((Q_train_.T - Q_opinf_6.T)**2)/np.mean(Q_train_.T**2)
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
    
    
        ### test time period prediction
        Q_0 = torch.tensor(Q_test_[:,0], dtype=torch.float32)
        
        Q_adjoint = odeint(func, Q_0, t_test).detach().cpu().numpy().T
        Q_opinf_6 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_6, Q_0, t_test).detach().cpu().numpy().T  # shape: (T, r)
        Q_opinf_2 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_2, Q_0, t_test).detach().cpu().numpy().T
        
        error_adjoint = np.mean((Q_test_.T - Q_adjoint.T)**2)/np.mean(Q_test_.T**2)
        error_opinf_6 = np.mean((Q_test_.T - Q_opinf_6.T)**2)/np.mean(Q_test_.T**2)
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
        
        # fig.suptitle(f'reg_Frobenius: {reg_Frobenius}, pieces: {pieces}, weighted: {weighted}')
        
        if save_results:
            fig.savefig(f'./results_node/Results_node_{name_suffix}.png')
        
        plt.close()
    
        print(f'opinf 6 test error: {np.log10(error_opinf_6):.6}, val error: {np.log10(error_opinf_6_valid):.6}')
        print(f'opinf 2 test error: {np.log10(error_opinf_2):.6}, val error: {np.log10(error_opinf_2_valid):.6}')
        print(f'adjoint test error: {np.log10(error_adjoint):.6}, val error: {np.log10(error_adjoint_valid):.6}')
    
    
    
        Q_0 = torch.tensor(Q_s[:,0], dtype=torch.float32)
        Q_adjoint = odeint(func, Q_0, t_train).detach().cpu().numpy().T  # shape: (T, r)
        Q_opinf_6 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_6, Q_0, t_train).detach().cpu().numpy().T
        Q_opinf_2 = np.random.randn(*Q_adjoint.shape)#odeint(func_opinf_2, Q_0, t_train).detach().cpu().numpy().T

    return Q_opinf_6, Q_adjoint, Q_opinf_2, \
            error_opinf_6, error_adjoint, error_opinf_2, \
            error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
            error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
            error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
            error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid

if __name__ == "__main__": 
    data_name = 'burgers'  ##  'fkpp'  ###  
    smoother = True # False
    # order = 'ord6'  # 'ord2' # 
    # noise_level = 100 ## 0 is no noise on train data
    # num_samples = 10000  ## 2000
    split_ratio_validation = .1
    # r = 2
    
    epochs = 200
    
    weighted = False
    save_results = True
    
    condition = False #True # False ## if True theta_opinf will be initial value, otherwise MLP with random initial value
    
    # for data_name in ['burgers', 'fkpp', 'lcd']:
    for data_name in ['fkpp']:
        if data_name=='fkpp':
            step = 10 ## 1, 2, 4, 10
            num_samples = 2001//step ## 2000 ##
            split_ratio = .75
            
        if data_name=='burgers':
            step = 10 # 1 # 10 # 100 # 500 # 
            num_samples = 10000//step # 10000
            split_ratio = .5
        
        if data_name=='lcd':
            step = 1 ## 1, 2, 4, 10
            num_samples = 2001//step ## 2000 ##
            split_ratio = .75
    
        opinf_use_val = False # True
        
        noise_level_list = [0, 40, 80, 120, 160, 200]
        for noise_level in noise_level_list: #[200]:#
        # for noise_level in [40]:
            
            error_opinf_6_init_list, error_adjoint_init_list, error_opinf_2_init_list = [], [], []
            error_opinf_6_train_list, error_adjoint_train_list, error_opinf_2_train_list = [], [], []
            error_opinf_6_list, error_adjoint_list, error_opinf_2_list = [], [], []
            error_opinf_6_valid_list, error_adjoint_valid_list, error_opinf_2_valid_list = [], [], []
                
            
            
            ### get data ###
            Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, num_samples = \
                                                data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)

            for r in range(1,6):#[3,4,5]:#range(1,6):
                print(f"\n========== ENTER r={r} ==========")

                ### A and H by operator inference under reduced order dataset
                A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
                    Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order = \
                    operator_inference(Q_train, t_train, Q_valid, t_valid, Q_original_test, t_test, r, opinf_use_val, smoother=smoother, weighted=weighted)
                
                # if data_name == 'burgers':                
                #     data_file = glob.glob(os.path.join(os.getcwd(),f"./data/burgers/exact_intrusive_operator/exact_operator_reproj_noise{noise_level}_{r}.npz"))[0]  ###   
                #     data = np.load(data_file)
                
                #     A_r_true, H_r_true, Q_reproj, t = data['A_r_true'], data['H_r_true'], data['Q_reproj'], data['t']
                #     Q_train, t_train, Q_valid, t_valid, Q_test, t_test, num_samples = \
                #         Q_reproj[:,:995], t[:995], Q_reproj[:,995:997], t[995:997], Q_reproj[:,997:], t[997:], t.shape[0]
                #     # Q_train, t_train, Q_valid, t_valid, Q_test, t_test, num_samples = \
                #     #     Q_reproj[:,:980][:,::5], t[:980][::5], Q_reproj[:,980:990][:,::5], t[980:990][::5], Q_reproj[:,990:][:,::5], t[990:][::5], int(t.shape[0]//5)
                    
                # ### A and H by operator inference under reduced order dataset
                # A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
                #     Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order = \
                #     operator_inference(Q_train, t_train, Q_valid, t_valid, Q_test, t_test, r, opinf_use_val, smoother=smoother, weighted=weighted)
                
                # pertube_level = 0 #1e-1 # 0 #1e-4
                # A_opinf = A_opinf + pertube_level*np.random.randn(*A_opinf.shape)
                # H_opinf = H_opinf + pertube_level*symetric_noise(r).reshape(r,r*r)
                
                func = train_node(Q_train_, t_train, Q_test_, t_test, A_opinf, H_opinf, r, epochs, optimizer_name='Adam', condition=condition)

                
                ratio = str(split_ratio).replace('.','p')
                name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{epochs}_smooth{smoother}_{r}_{condition}'
                
                Q_opinf_6, Q_adjoint, Q_opinf_2, \
                error_opinf_6, error_adjoint, error_opinf_2, \
                error_opinf_6_init_test, error_adjoint_init_test, error_opinf_2_init_test, \
                error_opinf_6_train, error_adjoint_train, error_opinf_2_train, \
                error_opinf_6_init_valid, error_adjoint_init_valid, error_opinf_2_init_valid, \
                error_opinf_6_valid, error_adjoint_valid, error_opinf_2_valid = \
                    predict_and_plot(func, r, A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, Q_train_, Q_valid_, Q_test_, \
                                 Q_s, t_train, t_valid, t_test, name_suffix=name_suffix, save_results=True)
                
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
                
                
                if save_results:
                    if hasattr(func, "A") and hasattr(func, "H"):
                        A_opt = func.A.detach().cpu().numpy()
                        H_opt = func.H.detach().cpu().numpy()
                    
                        np.savez(f"./results_node/theta_node_with_opinf_{name_suffix}.npz", \
                                 A_NODE=A_opt, H_NODE=H_opt, A_opinf=A_opinf, H_opinf=H_opinf)
                        
                        # np.savez(f'./results_node/Predictions_{name_suffix}.npz', \
                        #          Q_train_=Q_train_, Q_valid_=Q_valid_, Q_test_=Q_test_, Q_s=Q_s, \
                        #          t_train=t_train, t_valid=t_valid, t_test=t_test, Q_NODE=Q_NODE)
                            
                    else:
                        torch.save(
                            func.state_dict(),
                            f"./results_node/neural_ode_mlp_{name_suffix}.pt"
                        )
                        
        
                
                # torch.cuda.empty_cache()
                # del func

            
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
            
            if save_results:
                np.savez(f"./results_node/error_node_{name_suffix}.npz", 
                        error_opinf_6_list=error_opinf_6_list, error_adjoint_list=error_adjoint_list, error_opinf_2_list=error_opinf_2_list,
                        error_opinf_6_init_list=error_opinf_6_init_list, error_adjoint_init_list=error_adjoint_init_list, error_opinf_2_init_list=error_opinf_2_init_list,
                        error_opinf_6_train_list=error_opinf_6_train_list, error_adjoint_train_list=error_adjoint_train_list, error_opinf_2_train_list=error_opinf_2_train_list,
                        error_opinf_6_valid_list=error_opinf_6_valid_list, error_adjoint_valid_list=error_adjoint_valid_list, error_opinf_2_valid_list=error_opinf_2_valid_list,
                        )
            
            
                
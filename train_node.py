#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:12:30 2025

@author: dliu
"""

import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np
import opinf
from scipy.interpolate import interp1d
from utils import get_train_test_data, get_theta_by_opinf, add_noise, model_reducer
from utils import optimal_opinf, smooth
from train_adjoint import data_loader, operator_inference


import random
random.seed(10)
np.random.seed(10)    # for numpy random
torch.manual_seed(10)

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

# Define dynamics
class ODEFunc(torch.nn.Module):
    def __init__(self, r, theta=None):
        super().__init__()
        if theta is None:
            # Random initialization if no theta provided
            self.A = torch.nn.Parameter(torch.randn(r, r))
            self.H = torch.nn.Parameter(torch.randn(r, r * r))
        else:
            # Initialize from theta vector
            A = theta[:r**2].reshape(r, r)
            H = theta[r**2:].reshape(r, r * r)
            self.A = torch.nn.Parameter(A)
            self.H = torch.nn.Parameter(H)

    def forward(self, t, x):
        # Ensure x is column vector
        x = x.view(-1, 1)  # (n,1)

        # Kronecker product x⊗x, shape (n^2, 1)
        xx = torch.kron(x, x)

        # Dynamics: A x + H (x⊗x)
        dxdt = self.A @ x + self.H @ xx

        return dxdt.view(-1)  # flatten back to vector


def train_node(Q_train_, t_train, Q_test_, t_test, A_opinf, H_opinf, r, epochs=20, optimizer_name = 'Adam'):
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
    
    
    r, k_samples = Q_train_.shape
    func = ODEFunc(r=r, theta=theta_opinf)
    
    lr = 1e-5
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
    
    solution = odeint(func, y0, t)  # shape: (T, r)
    print(f'opinf loss : {torch.mean((solution - Q_train_.T)**2)}')
    
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            solution = odeint(func, y0, t)  # shape: (T, r)
            loss = torch.mean((solution - Q_train_.T)**2)
            loss.backward()
            # optimizer.step()
            return loss
        
    
        if optimizer_name != "LBFGS":
            loss = closure() 
            optimizer.step()
        else:
            loss = optimizer.step(closure)
                
    
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.6f}")
            
    return func

if __name__ == "__main__": 
    data_name = 'burgers'  ##  'fkpp'  ###  
    smoother = False
    order = 'ord6'  # 'ord2' # 
    noise_level = 100 ## 0 is no noise on train data
    num_samples = 10000  ## 2000
    split_ratio_validation = .1
    r = 2
    
    
    weighted = False
    save_results = True
    
    
    
    # for data_name in ['burgers', 'fkpp', 'lcd']:
    for data_name in ['burgers']:
        if data_name=='fkpp':
            step = 10 ## 1, 2, 4, 10
            num_samples = 2001//step ## 2000 ##
            split_ratio = .75
            
        if data_name=='burgers':
            step = 1 # 1 # 10 # 100 # 500 # 
            num_samples = 10000//step # 10000
            split_ratio = .5
        
        if data_name=='lcd':
            step = 1 ## 1, 2, 4, 10
            num_samples = 2001//step ## 2000 ##
            split_ratio = .75
    
        opinf_use_val = True
        
        noise_level_list = [0, 40, 80, 120, 160, 200]
        for noise_level in noise_level_list:
        # for noise_level in [40]:

            ### get data ###
            Q_train, t_train, Q_valid, t_valid, Q_test, t_test, Q_original_train, Q_original_valid, Q_original_test, num_samples = \
                                                data_loader(data_name, step, noise_level, split_ratio, split_ratio_validation)
            
            ### A and H by operator inference under reduced order dataset
            A_opinf, H_opinf, A_opinf_6, H_opinf_6, A_opinf_2, H_opinf_2, \
                Q_train_, Q_valid_, Q_test_, Q_s, weights, regularizer, par_tsvd, order = \
                operator_inference(Q_train, t_train, Q_valid, t_valid, Q_original_test, t_test, r, opinf_use_val, smoother=smoother, weighted=weighted)
            
            
            epochs = 100
            func = train_node(Q_train_, t_train, Q_test_, t_test, A_opinf, H_opinf, r, epochs, optimizer_name='Adam')
            
            t_all = np.r_[t_train,t_test]
            Q_all = np.c_[Q_train_,Q_test_]
            
            ### solution by NODE
            Q_0 = torch.tensor(Q_test_[:,0], dtype=torch.float32)
            t_test_tensor = torch.tensor(t_test, dtype=torch.float32)
            Q_NODE = odeint(func, Q_0, t_test_tensor).detach().cpu().numpy()  # shape: (T, r)
            
            ### solution by operator inference
            theta_opinf = torch.tensor(np.concatenate([A_opinf.ravel(), H_opinf.ravel()]), dtype=torch.float32)
            func_opinf = ODEFunc(r=r, theta=theta_opinf)
            Q_opinf = odeint(func_opinf, Q_0, t_test_tensor).detach().cpu().numpy()
            
            
            
            error_state_opinf = Q_test_.T - Q_opinf
            error_state_NODE = Q_test_.T - Q_NODE
            # error_state_adjoint_manually = Q_all.T - Q_adjoint_manually
            
            error_opinf = np.mean(error_state_opinf**2)/np.mean(Q_test_**2)
            error_NODE = np.mean(error_state_NODE**2)/np.mean(Q_test_.T**2)
            # error_adjoint_manually = np.mean(error_state_adjoint_manually[k_samples:]**2)/np.mean(Q_all.T[k_samples:]**2)
            
            print(f'opinf test error: {np.log10(error_opinf)}')
            print(f'NODE test error: {np.log10(error_NODE)}')
            # print(f'adjoint test error: {np.mean(error_state_adjoint_manually[k_samples:]**2)}')
            
            
            
            A_opt = func.A.detach().cpu().numpy()
            H_opt = func.H.detach().cpu().numpy()
            ratio = str(split_ratio).replace('.','p')
            name_suffix = f'{data_name}_sam{num_samples}_ratio{ratio}_useVal{opinf_use_val}_noise{noise_level}_iter{epochs}_smooth{smoother}'
            if save_results:
                np.savez(f"./results_node/theta_node_with_opinf_{name_suffix}.npz", \
                         A_NODE=A_opt, H_NODE=H_opt, A_opinf=A_opinf, H_opinf=H_opinf)
                
                np.savez(f'./results_node/Predictions_{name_suffix}.npz', \
                         Q_train_=Q_train_, Q_valid_=Q_valid_, Q_test_=Q_test_, Q_s=Q_s, \
                         t_train=t_train, t_valid=t_valid, t_test=t_test, Q_NODE=Q_NODE)
                    
            
            
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].plot(t_all, Q_all.T, label='true')
            axes[0].plot(t_test, Q_NODE, '--', label='adjoint')
            axes[0].plot(t_test, Q_opinf, '--', label='opinf')
            axes[0].axvline(x=t_train[-1], ls='--')
            
            axes[1].plot(np.mean(abs(error_state_opinf),axis=1), label='opinf')
            axes[1].plot(np.mean(abs(error_state_NODE),axis=1), label='adjoint')
            # axes[1].plot(np.mean(abs(error_state_adjoint_manually[k_samples:]),axis=1), label='adjoint manually')
            axes[1].legend()
            fig.savefig(f"./results_node/theta_node_with_opinf_{name_suffix}.png")
            
            
            
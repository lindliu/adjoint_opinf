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

import random
random.seed(10)
np.random.seed(10)    # for numpy random
################################ OpInf ###############################
######################################################################


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



data_name = 'burgers'  ##  'fkpp'  ###  
smoother = False
order = 'ord6'  # 'ord2' # 
noise_level = 100 ## 0 is no noise on train data
num_samples = 10000  ## 2000

r = 7



import random
random.seed(10)
np.random.seed(10)    # for numpy random
torch.manual_seed(10)

if data_name == 'burgers':
    data_file = "./data/total_burgers_snapshots_nu_01.npz"  ###  
    split_ratio = .5   

if data_name == 'fkpp':
    data_file = "./data/total_fkpp.npz"     ## 
    split_ratio = .75   

data = np.load(data_file)
Q_original, t = data['Q'], data['t']

if data_name == 'fkpp':
    Q_original = Q_original[::4,::4,:]
Q_original = Q_original.reshape(-1, Q_original.shape[-1])


Q_original_train, t_train, Q_original_test, t_test = get_train_test_data(Q_original, t, num_samples=num_samples, split_ratio=split_ratio)

Q = add_noise(Q_original, percentage=noise_level, method="std")
Q_train, t_train, Q_test, t_test = get_train_test_data(Q, t, num_samples=num_samples, split_ratio=split_ratio)


# # Snapshot data Q = [q(t_0) q(t_1) ... q(t_k)], size=(r,k)
t = t_train
dt = t[1] - t[0]


### reduce data order to r
Q_, Q_test_ = model_reducer(Q_train, Q_test, r)

if smoother:
    Q_, _ = smooth(Q_, t, window_size=None, poly_order=3)


k_samples = Q_.shape[1]  # number of samples for training(snapshot data)


### select best A_opinf, H_opinf by grid search #### 
### TruncatedSVDSolver for L2T is critical  ########
A_opinf, H_opinf, regularizer, par_tsvd = optimal_opinf(Q_, t, order)

##### result by order='ord2'
A_opinf_ord2, H_opinf_ord2, _, _ = optimal_opinf(Q_, t, 'ord2')


### initial guess for A and H from operator inference
theta_opinf_ = np.concatenate([A_opinf.ravel(), H_opinf.ravel()])

# print(theta_opinf_)

# data = np.load(f'./data/reduced_{data_name}_{order}_noise{noise_level}_sam{num_samples}.npz')
# Q_train, Q_test, t_train, t_test, theta_opinf, theta_adjoint = data['Q_train'], data['Q_test'], data['t_train'], data['t_test'], data['theta_opinf'], data['theta_adjoint']
Q_ = torch.tensor(Q_,dtype=torch.float32)
t_train = torch.tensor(t_train,dtype=torch.float32)
Q_test = torch.tensor(Q_test_,dtype=torch.float32)
t_test = torch.tensor(t_test,dtype=torch.float32)
theta_opinf = torch.tensor(theta_opinf_,dtype=torch.float32)
# theta_adjoint_manually = torch.tensor(theta_adjoint,dtype=torch.float32)


r, k_samples = Q_.shape
func = ODEFunc(r=r, theta=theta_opinf)


optimizer_name = 'SGD'  ## 'Adam' # 'LBFGS' # 
lr = 1e-5
if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(func.parameters(), lr=lr)
elif optimizer_name == "SGD":
    # optimizer = torch.optim.SGD(func.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.SGD(func.parameters(), lr=lr, momentum=0)
elif optimizer_name == "LBFGS":
    optimizer = torch.optim.LBFGS(func.parameters(), lr=lr, max_iter=20)




# Initial condition & parameters
y0 = Q_[:,0]
t = t_train

solution = odeint(func, y0, t)  # shape: (T, r)
print(f'opinf loss : {torch.mean((solution - Q_.T)**2)}')

for epoch in range(20):
    def closure():
        optimizer.zero_grad()
        solution = odeint(func, y0, t)  # shape: (T, r)
        loss = torch.mean((solution - Q_.T)**2)
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
        




t_all = torch.cat((t_train,t_test),dim=0)
Q_, Q_test_ = model_reducer(Q_train, Q_original_test, r)
Q_all = torch.tensor(np.c_[Q_,Q_test_], dtype=torch.float32)
Q_0 = Q_all[:,k_samples]
    
# Q_all = torch.cat((Q_train,Q_test),dim=1)

Q_adjoint = odeint(func, Q_0, t_test)  # shape: (T, r)


theta_opinf = torch.tensor(theta_opinf_,dtype=torch.float32)
# theta_adjoint_manually = torch.tensor(theta_adjoint_manually,dtype=torch.float32)


func_opinf = ODEFunc(r=r, theta=theta_opinf)
Q_opinf = odeint(func_opinf, Q_0, t_test) 

# func_adjoint_manually = ODEFunc(r=r, theta=theta_adjoint_manually)
# Q_adjoint_manually = odeint(func_adjoint_manually, y0, t_test) 






Q_all = np.array(Q_all.detach())
Q_opinf = np.array(Q_opinf.detach())
Q_adjoint = np.array(Q_adjoint.detach())
# Q_adjoint_manually = np.array(Q_adjoint_manually.detach())

error_state_opinf = Q_test_.T - Q_opinf
error_state_adjoint = Q_test_.T - Q_adjoint
# error_state_adjoint_manually = Q_all.T - Q_adjoint_manually

error_opinf = np.mean(error_state_opinf[k_samples:]**2)/np.mean(Q_test_.T**2)
error_adjoint = np.mean(error_state_adjoint[k_samples:]**2)/np.mean(Q_test_.T**2)
# error_adjoint_manually = np.mean(error_state_adjoint_manually[k_samples:]**2)/np.mean(Q_all.T[k_samples:]**2)

print(f'opinf test error: {np.mean(error_state_opinf**2)}')
print(f'adjoint test error: {np.mean(error_state_adjoint**2)}')
# print(f'adjoint test error: {np.mean(error_state_adjoint_manually[k_samples:]**2)}')




fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].plot(t_all, Q_all.T, label='true')
axes[0].plot(t_test, Q_adjoint, '--', label='adjoint')
axes[0].plot(t_test, Q_opinf, '--', label='opinf')
axes[0].axvline(x=t_train[-1], ls='--')

axes[1].plot(np.mean(abs(error_state_opinf),axis=1), label='opinf')
axes[1].plot(np.mean(abs(error_state_adjoint),axis=1), label='adjoint')
# axes[1].plot(np.mean(abs(error_state_adjoint_manually[k_samples:]),axis=1), label='adjoint manually')
axes[1].legend()











# t_all = torch.cat((t_train,t_test),dim=0)
# Q_, Q_test_ = model_reducer(Q_train, Q_original_test, r)
# Q_all = torch.tensor(np.c_[Q_,Q_test_], dtype=torch.float32)
# Q_0 = Q_all[:,0]
    
# # Q_all = torch.cat((Q_train,Q_test),dim=1)
# Q_adjoint = odeint(func, Q_0, t_all)  # shape: (T, r)


# theta_opinf = torch.tensor(theta_opinf_,dtype=torch.float32)
# # theta_adjoint_manually = torch.tensor(theta_adjoint_manually,dtype=torch.float32)


# func_opinf = ODEFunc(r=r, theta=theta_opinf)
# Q_opinf = odeint(func_opinf, Q_0, t_all) 

# # func_adjoint_manually = ODEFunc(r=r, theta=theta_adjoint_manually)
# # Q_adjoint_manually = odeint(func_adjoint_manually, y0, t_all) 


# Q_all = np.array(Q_all.detach())
# Q_opinf = np.array(Q_opinf.detach())
# Q_adjoint = np.array(Q_adjoint.detach())
# # Q_adjoint_manually = np.array(Q_adjoint_manually.detach())

# error_state_opinf = Q_all.T - Q_opinf
# error_state_adjoint = Q_all.T - Q_adjoint
# # error_state_adjoint_manually = Q_all.T - Q_adjoint_manually

# error_opinf = np.mean(error_state_opinf[k_samples:]**2)/np.mean(Q_all.T[k_samples:]**2)
# error_adjoint = np.mean(error_state_adjoint[k_samples:]**2)/np.mean(Q_all.T[k_samples:]**2)
# # error_adjoint_manually = np.mean(error_state_adjoint_manually[k_samples:]**2)/np.mean(Q_all.T[k_samples:]**2)

# print(f'opinf test error: {np.mean(error_state_opinf[k_samples:]**2)}')
# print(f'adjoint test error: {np.mean(error_state_adjoint[k_samples:]**2)}')
# # print(f'adjoint test error: {np.mean(error_state_adjoint_manually[k_samples:]**2)}')




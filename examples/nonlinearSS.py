"""
Nonlinear system
----------------
Example of nonlinear system identified using LAVA-R and nonlinear Laplace basis functions.
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
# local imports
from context import lava


# parameters
M = 3
n_a = 1
n_b = 1
n_c = 1
n_d = 1

# setup system matrices
A = np.array([[0.9, 0], [0.08, 0.9]])
B = np.array([[0.3, 0], [0, 0.6]])
C = np.eye(2)

# generate U
N = 500
N_fit = int(math.ceil(N / 2))
N_test = N - N_fit
U = 4*np.random.randn(2,N)

# Initial values
x = np.zeros((2,N))
Y = np.zeros((C.shape[1], N))

for t in range(N-1):
    x[:,t+1] = A@x[:,t] + B@U[:,t]

    if np.absolute(x[0, t+1]) > 2 :
        x[0, t+1] = np.sign( x[0, t+1] )*2

    Y[:, t+1] = C@x[:, t+1]

# get lava object
estimate = lava.core.LavaLaplace(
    n_y=Y.shape[0],
    n_u_phi=2,
    n_u_gamma=2,
    n_u_binary=0,
    n_a=n_a,
    n_b=n_b,
    n_c=n_c,
    n_d=n_d,
    M=M,
    L=3,
    L_vec_u=[3, 3],
    L_vec_y=[3, 3]
)

# train lava model
for t in range(N_fit):
    estimate.step(y=Y[:, t], u_phi=U[:, t], u_gamma=U[:, t], u_binary=[])

# predict using lava model
Y_hat, Theta_phi, Z_gamma = estimate.forecast(u_phi=U[:, N_fit:N_fit+N_test], u_gamma=U[:, N_fit:N_fit+N_test], u_binary=np.zeros(N_test))

# plot results
plt.subplot(2,1,1)
plt.plot(Y[0,N_fit:N_fit+N_test])
plt.plot(Y_hat[0,1:N_test])
plt.legend(["y", "y_hat"])
plt.subplot(2,1,2)
plt.plot(Y[1,N_fit:N_fit+N_test])
plt.plot(Y_hat[1,1:N_test])
plt.legend(["y", "y_hat"])
plt.show()


"""
Time series
-----------
Example of periodic time series identified using LAVA-R and Fourier basis function, with separate inputs to the nominal and lava model respectively.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
# local imports
from context import lava

# Generate U
N = 1000
N_fit = int(math.ceil(N / 2))
N_test = N - N_fit
Y = np.zeros((1, N))
U = np.zeros((2, N))

# generate synthetic data
for t in range(N):
    U[0, t] = np.remainder(t, 24)
    U[1, t] = t + 4*np.random.randn(1,1)
    p = 24
    r = np.remainder(t + 5, p)
    if r > (p / 2):
        Y[:, t] = 100 + t
    else:
        Y[:, t] = 200 + t

# parameters
M = 20
n_a = 0
n_b = 2
n_c = 0
n_d = 2
L_vec_u = [24, 24]
L_vec_y = [500]



# create lava object
estimate = lava.core.LavaLinearFourier(n_y=Y.shape[0], n_u_phi=1, n_u_gamma=1, n_u_binary=0, n_a=n_a, n_b=n_b, n_c=n_c, n_d=n_d, M=M, L_vec_u=L_vec_u, L_vec_y=L_vec_y, L=3)

# identify parameters
for t in range(N_fit):
    print(t)
    estimate.step(y=Y[:, t], u_gamma=U[0, t], u_binary=[], u_phi=U[1, t])

# forecast
Y_hat, Theta_phi, Z_gamma = estimate.forecast(u_phi=U[1, N_fit:N_fit+N_test], u_gamma=U[0, N_fit:N_fit+N_test], u_binary=np.zeros((0,N_test)))

# plot results
plt.plot(Y[0, N_fit:N_fit+N_test])
plt.plot(Y_hat[0, 0:N_test])
plt.plot(Z_gamma[0, 0:N_test])
plt.plot(Theta_phi[0, 0:N_test])
plt.legend(["y", "y_hat", "Z_gamma", "Theta_phi"])
plt.show()
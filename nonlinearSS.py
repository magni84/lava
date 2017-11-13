import numpy as np
import lava



# Setup system matrices
A = np.array( [[ 0.9, 0 ], [0.08, 0.9]])
B = np.array( [[ 0.3, 0], [0, 0.6]])
C = np.eye(2)

# Generate U
# TODO Generate U according to Wigren's code.
N = 100
U = 2*np.random.randn(2,N)

# Initial values
x = np.zeros((2,N))
Y = np.zeros( (C.shape[1], N) )

for t in range(N-1):
    x[:,t+1] = A@x[:,t] + B@U[:,t]

    if np.absolute(x[0, t+1]) > 2 :
        x[0, t+1] = np.sign( x[0, t+1] )*2

    Y[:, t+1] = C@x[:, t+1];


# Identify parameters
estimate = lava.LavaLaplace(2,2,1,1,3,[3,3,3,3])
Phi = np.empty( (estimate.n_phi, N))
for t in range(N):
    estimate.step(Y[:,t], U[:,t])
    Phi[:, t] = estimate.phi


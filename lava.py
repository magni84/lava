import numpy as np
import math

class Lava:

    """Docstring for MyClass. """

    def __init__(self, n_y, n_phi, n_gamma):
        self.n_y = n_y
        self.n_phi = n_phi
        self.n_gamma = n_gamma
        self.L = 10

        self.reset_id()
        
    def reset_id(self):
        self.n = 0
        self.Psi = {'phi_phi': np.zeros((self.n_phi, self.n_phi)),
                    'gamma_gamma': np.zeros((self.n_gamma, self.n_gamma)),
                    'y_y': np.zeros((self.n_y, self.n_y)),
                    'phi_gamma': np.zeros((self.n_phi, self.n_gamma)),
                    'phi_y': np.zeros((self.n_phi, self.n_y)),
                    'gamma_y': np.zeros((self.n_gamma, self.n_y))}

        self.Z = np.zeros((self.n_y, self.n_gamma))
        self.Theta_bar = np.zeros((self.n_y, self.n_phi))
        self.P = 100000*np.eye(self.n_phi)
        self.H = np.zeros((self.n_phi, self.n_gamma))



    def step(self, y, phi, gamma):
        # Upe all PSI rices
        self.Psi['phi_phi'] += np.outer(phi, phi) 
        self.Psi['gamma_gamma'] += np.outer(gamma, gamma)
        self.Psi['y_y'] += np.outer(y, y)
        self.Psi['phi_gamma'] += np.outer(phi,gamma)
        self.Psi['phi_y'] += np.outer(phi, y)
        self.Psi['gamma_y'] += np.outer(gamma, y)

        # Update esties for Theta_bar
        self.P -= (self.P@np.outer(phi,phi)@self.P)/(1+phi@self.P@phi)
        self.H +=  self.P@( np.outer(phi,gamma)-np.outer(phi,phi)@self.H )
        self.Theta_bar += np.outer(y - self.Theta_bar@phi,phi)@self.P

        # Upe esties of Z
        T = (self.Psi['gamma_gamma'] - self.Psi['phi_gamma'].T@self.H 
             - self.H.T@self.Psi['phi_gamma'] 
             + self.H.T@self.Psi['phi_phi']@self.H)

        self.n += 1

        for i in range(self.n_y):
            kappa = (self.Psi['y_y'][i,i] 
                     + self.Theta_bar[i,:]@self.Psi['phi_phi']
                     @ self.Theta_bar[i,:].T 
                     - 2*self.Theta_bar[i,:]@self.Psi['phi_y'][:,i])

            rho = (self.Psi['gamma_y'][:,i] 
                   - self.Psi['phi_gamma'].T@self.Theta_bar[i,:].T 
                   - self.H.T@self.Psi['phi_y'][:,i] 
                   + self.H.T@self.Psi['phi_phi']@self.Theta_bar[i,:].T)

            eta = (kappa - 2*rho.T@self.Z[i,:].T 
                   + self.Z[i,:]@T@self.Z[i,:].T)

            zeta = rho - T@self.Z[i,:].T

            for k in range(self.L):
                for j in range(self.n_gamma):
                    alpha = (eta + T[j,j]*self.Z[i,j]**2 
                             + 2*zeta[j])
                    g = zeta[j] + T[j,j]*self.Z[i,j]
                    w = np.sqrt(self.Psi['gamma_gamma'][j,j]/self.n)

                    if alpha*w**2 < g**2 :
                        try:
                            z_new = (np.sign(g)
                                     *( np.abs(g)/T[j,j] 
                                       - w/(T[j,j]*math.sqrt(T[j,j]-w**2))
                                       * math.sqrt(alpha*T[j,j]-g**2) ))
                        except ValueError:
                            z_new = 0
                    else:
                        z_new = 0

                    z_diff = self.Z[i,j] - z_new
                    eta += T[j,j]*z_diff**2 + 2*z_diff*zeta[j]
                    zeta += T[:,j]*z_diff
                    self.Z[i,j] = z_new

        self.Theta = self.Theta_bar - self.Z@self.H.T



class LavaLaplace(Lava):

    """Docstring for LavaLaplace. """

    def __init__(self, n_y, n_u, n_a, n_b, M, L_vec):
        self.n_y = n_y
        self.n_u = n_u
        self.n_a = n_a
        self.n_b = n_b
        self.M = M
        self.L_vec = L_vec

        super().__init__(n_y, n_y*n_a+n_b*n_u+1, M**(n_y*n_a+n_u*n_b))

        self.phi = np.zeros(self.n_phi)
        self.phi[-1] = 1
        self.gamma = np.empty(self.n_gamma)

    def reset_id(self):
        self.phi = np.zeros(self.n_phi)
        self.phi[-1] = 1
        self.gamma = np.empty(self.n_gamma)
        super().reset_id()

    def batch_id(self, Y, U):
        N = Y.shape[1]

        for t in range(N):
            self.step(Y[:,t], U[:,t])

    def step(self, y, u):
        self.update_gamma(self.phi[:-1])
        Lava.step(self, y, self.phi, self.gamma)
        self.update_phi(y,u)

    def update_phi(self,y,u):
        # Update y part
        self.phi[self.n_y : self.n_y*self.n_a] =self.phi[0 : self.n_y*(self.n_a-1)]
        self.phi[0 : self.n_y] = y

        # Update u part
        self.phi[-self.n_u*(self.n_b-1)-1 :-1 ] = self.phi[-self.n_u*self.n_b-1 : -self.n_u-1]
        self.phi[-self.n_u*self.n_b-1 : -1] = u

    def update_gamma(self, x):
        D = len(x)
        P = np.ones( self.M**D )
        j_vec = np.ones( D )

        for c in range(self.M**D):
            for k in range(D):
                P[c] *= (np.sin( np.pi * j_vec[k]  
                               * (x[k] + self.L_vec[k] )/(2*self.L_vec[k])) 
                         / np.sqrt(self.L_vec[k]))

            j_vec[0] = j_vec[0] + 1
            if D>1:
                for k in range(1,D):
                    if np.mod( j_vec[k-1], self.M+1) == 0:
                        j_vec[k-1] = 1
                        j_vec[k] = j_vec[k] + 1

        self.gamma = P

    def simulate(self, u):
        N = u.shape[1]
        oldphi = self.phi
        self.phi = 0*self.phi

        y = np.empty((self.n_y, N))
        phi = np.zeros(self.n_phi)

        for t in range(N):
            self.update_gamma(self.phi[:-1])
            y[:,t] = self.Theta@self.phi + self.Z@self.gamma
            self.update_phi(y[:,t], u[:,t])

        self.phi = oldphi
        return y

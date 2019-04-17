import numpy as np
import math


class LavaBase:
    """Base LAVA model that is inherited for using different base functions for latent variable model.
    """

    def __init__(
        self, n_y, n_u_phi, n_u_gamma, n_u_binary, n_a, n_b, n_c, n_d, M, L, L_vec_u, L_vec_y):
        """
        Args:
            n_y (int): :math:`n_y` - Number of outputs
            n_u_phi (int): :math:`n_{u,\phi}` - Number of inputs to nominal model
            n_u_gamma (int): :math:`n_{u,\phi}` - Number of continuous inputs to latent variable model
            n_u_binary (int): :math:`n_{u,binary}` - Number of binary inputs to latent variable model
            n_a (int): :math:`n_c` - Order of nominal model (number of output lags)
            n_b (int): :math:`n_d` - Order of nominal model (number of input lags)
            n_c (int): :math:`n_c` - Order of latent variable model (number of output lags)
            n_d (int): :math:`n_d` - Order of latent variable model (number of input lags)
            M: :math:`M` - Resolution of basis function
            L: :math:`L` - Resolution of basis function
            L_vec_u: :math:`L_u`  - (u_max - u_min) of inputs
            L_vec_y: :math:`L_y`  - (y_max - y_min) of outputs
        """
        self.n_y = n_y
        self.n_u_phi = n_u_phi
        self.n_u_gamma = n_u_gamma
        self.n_u_binary = n_u_binary
        self.n_gamma = 0
        self.n_a = n_a
        self.n_b = n_b
        self.n_c = n_c
        self.n_d = n_d
        self.M = M
        self.L = L
        self.L_vec_u = L_vec_u
        self.L_vec_y = L_vec_y
        self.n_phi = n_y * n_a + n_u_phi * n_b + 1
        self.reset_var()
        self.reset_id()

    def reset_id(self):
        """Resets identification parameters to default values.

        Attributes:
            n (int): Number of steps performed
            Psi: :math:`\Psi` - Matrix of products of inputs and outputs :math:`\phi\phi, \gamma\gamma, yy, \phi\gamma, \phi y, \gamma y`
            Z: :math:`Z` - Latent parameter model matrix
            Theta_bar: :math:`\overline{\Theta}` - Intermediate nominal model parameter matrix
            P: :math:`P` - Covariance matrix for recursive least squares estimation for nominal model parameters
            H: :math:`H` - Regression matrix for recursive least squares estimation for nominal model parameters
        """
        self.n = 0
        self.Psi_phi_phi = np.zeros((self.n_phi, self.n_phi))
        self.Psi_gamma_gamma = np.zeros(
            (self.n_gamma, self.n_gamma)
        )  # , chunks=(1000,1000))
        self.Psi_y_y = np.zeros((self.n_y, self.n_y))
        self.Psi_phi_gamma = np.zeros((self.n_phi, self.n_gamma))
        self.Psi_phi_y = np.zeros((self.n_phi, self.n_y))
        self.Psi_gamma_y = np.zeros((self.n_gamma, self.n_y))
        self.Z = np.zeros((self.n_y, self.n_gamma))
        self.Theta_bar = np.zeros((self.n_y, self.n_phi))
        self.P = 100000 * np.eye(self.n_phi)
        self.H = np.zeros((self.n_phi, self.n_gamma))

    def reset_var(self):
        """Resets input variables matrices.
        """
        self.phi = np.zeros(self.n_phi)
        self.phi[-1] = 1
        self.gamma = np.zeros(self.n_gamma) # empty
        self.y_phi = np.empty((self.n_y, self.n_a))
        self.u_phi = np.empty((self.n_u_phi, self.n_b))
        self.y_gamma = np.empty((self.n_y, self.n_c))
        self.u_gamma = np.zeros((self.n_u_gamma, self.n_d))
        self.u_binary = np.zeros((self.n_u_binary, 1))

    def _shift_update(self, x, u):
        """"Shifts a vector one step.
        """
        if x.size > 0:
            x_r = np.column_stack([u, x[:, 0:-1]])
        else:
            x_r = x
        return x_r

    def _update_phi(self, y, u_phi):
        """
        Updates phi vector.
        """
        # shift and update y and u
        self.y_phi = self._shift_update(self.y_phi, y)
        self.u_phi = self._shift_update(self.u_phi, u_phi)

        self.phi = np.concatenate(
            [self.y_phi.flatten(), self.u_phi.flatten(), np.ones(1)], axis=0
        )

    def _get_basis(self):
        pass

    def _update_gamma(self, y, u_gamma, u_binary):
        # shift and update y and u
        self.y_gamma = self._shift_update(self.y_gamma, y)
        self.u_gamma = self._shift_update(self.u_gamma, u_gamma)
        self.u_binary = self._shift_update(self.u_binary, u_binary)

        # get gamma from basis functions
        self.gamma = self._get_basis()

    def step(self, y, u_phi, u_gamma, u_binary):
        """Performs one estimation step:

        Args:
            y: :math:`y` - Outputs of model
            phi: :math:`\phi` - Inputs to nominal model
            gamma: :math:`\gamma` - Inputs to latent variable model

        * Update vector of products of inputs and outputs :math:`\Psi`
        * Update covariance matrix :math:`P`
        * Update regression matrix :math:`H`
        * Update matrix of parameters for nominal model :math:`\overline{\Theta}`
        * Update matrix of parameters for latent variable model :math:`Z`
        """

        # Update psi vector
        self.Psi_phi_phi += np.outer(self.phi, self.phi)
        self.Psi_gamma_gamma += np.outer(self.gamma, self.gamma)
        self.Psi_y_y += np.outer(y, y)
        self.Psi_phi_gamma += np.outer(self.phi, self.gamma)
        self.Psi_phi_y += np.outer(self.phi, y)
        self.Psi_gamma_y += np.outer(self.gamma, y)

        # Update estimates for Theta_bar
        self.P -= (self.P @ np.outer(self.phi, self.phi) @ self.P) / (
            1 + self.phi @ self.P @ self.phi
        )
        self.H += self.P @ (
            np.outer(self.phi, self.gamma) - np.outer(self.phi, self.phi) @ self.H
        )
        self.Theta_bar += np.outer(y - self.Theta_bar @ self.phi, self.phi) @ self.P

        # Update estimates of Z
        T = (
            self.Psi_gamma_gamma
            - self.Psi_phi_gamma.T @ self.H
            - self.H.T @ self.Psi_phi_gamma
            + self.H.T @ self.Psi_phi_phi @ self.H
        )

        # increment step
        self.n += 1

        for i in range(self.n_y):
            kappa = (
                self.Psi_y_y[i, i]
                + self.Theta_bar[i, :] @ self.Psi_phi_phi @ self.Theta_bar[i, :].T
                - 2 * self.Theta_bar[i, :] @ self.Psi_phi_y[:, i]
            )

            rho = (
                self.Psi_gamma_y[:, i]
                - self.Psi_phi_gamma.T @ self.Theta_bar[i, :].T
                - self.H.T @ self.Psi_phi_y[:, i]
                + self.H.T @ self.Psi_phi_phi @ self.Theta_bar[i, :].T
            )

            eta = kappa - 2 * rho.T @ self.Z[i, :].T + self.Z[i, :] @ T @ self.Z[i, :].T
            zeta = rho - T @ self.Z[i, :].T

            for k in range(self.L):
                for j in range(self.n_gamma):
                    alpha = (
                        eta + T[j, j] * self.Z[i, j] ** 2 + 2 * zeta[j] * self.Z[i, j]
                    )
                    g = zeta[j] + T[j, j] * self.Z[i, j]
                    # LIKES
                    w = np.sqrt(self.Psi_gamma_gamma[j, j] / self.n)
                    if alpha * w ** 2 < g ** 2:
                        try:
                            z_new = np.sign(g) * (
                                np.abs(g) / T[j, j]
                                - w
                                / (T[j, j] * math.sqrt(T[j, j] - w ** 2))
                                * math.sqrt(alpha * T[j, j] - g ** 2)
                            )
                        except ValueError:
                            # TODO This can only happen
                            # due to numerical round-off errors.
                            # It should be z_new = 0 in this case,
                            z_new = 0
                    else:
                        z_new = 0

                    z_diff = self.Z[i, j] - z_new
                    eta += T[j, j] * z_diff ** 2 + 2 * z_diff * zeta[j]
                    zeta += T[:, j] * z_diff
                    self.Z[i, j] = z_new

        self.Theta = self.Theta_bar - self.Z @ self.H.T

        # update phi and gamma
        self._update_phi(y, u_phi)
        self._update_gamma(y, u_gamma, u_binary)

    # perform one prediction step
    def forecast_step(self, u_phi, u_gamma, u_binary):
        """Perform one step ahead forecast.
        """
        # make forecast step
        Theta_phi = self.Theta @ self.phi
        Z_gamma = self.Z @ self.gamma
        y_hat = Theta_phi + Z_gamma  # self.Theta @ self.phi + self.Z @ self.gamma

        # update phi and gamma
        #self._update_phi(y_hat, u_phi)
        #self._update_gamma(y_hat, u_gamma, u_binary)

        return y_hat, Theta_phi, Z_gamma

    def forecast(self, u_phi, u_gamma, u_binary):
        """Forecast for the length of the input vector.
        """
        N = u_phi.shape[1] if u_phi.shape[1:] else u_phi.shape[0]
        y_hat = np.zeros((self.n_y, N))
        Theta_phi = np.zeros((self.n_y, N))
        Z_gamma = np.zeros((self.n_y, N))

        for t in range(N-1):
            # update gamma
            self._update_gamma(y_hat[..., t + 1], u_gamma[..., t], u_binary[..., t])

            # forecast step
            y_hat_f, Theta_phi_f, Z_gamma_f = self.forecast_step(
                u_phi[..., t], u_gamma[..., t], u_binary[..., t]
            )

            # update forecast
            y_hat[:, t + 1] = y_hat_f
            Theta_phi[:, t + 1] = Theta_phi_f
            Z_gamma[:, t + 1] = Z_gamma_f

            # update phi
            self._update_phi(y_hat[..., t + 1], u_phi[..., t])


        return y_hat, Theta_phi, Z_gamma


class LavaLinearFourier(LavaBase):
    """LAVA model using ARX nominal model and Fourier base functions.
    """

    def __init__(
        self, n_y, n_u_phi, n_u_gamma, n_u_binary, n_a, n_b, n_c, n_d, M, L, L_vec_u, L_vec_y):
        super().__init__(n_y, n_u_phi, n_u_gamma, n_u_binary, n_a, n_b, n_c, n_d, M, L, L_vec_u, L_vec_y)
        self.n_gamma = 2 * M * (n_y * n_c + n_u_gamma * n_d) * (n_u_binary + 1)
        self.M = M
        self.reset_var()
        self.reset_id()


    def _get_basis(self):
        """Get the linear Fourier basis expansion.
        """
        # basis matrices for inputs and outputs
        F_y = np.zeros((self.n_y * self.M, self.n_c * 2))
        F_u = np.zeros((self.n_u_gamma * self.M, self.n_d * 2))

        # generate basis functions
        j = 0
        k = 0
        for m in range(self.M):
            if self.n_c > 0:
                for i in range(self.n_y):
                    F_y[j + i, :] = np.array([np.cos(np.pi * (m + 1) * self.y_gamma[i, :] / self.L_vec_y[i]),
                                 np.sin(np.pi * (m + 1) * self.y_gamma[i, :] / self.L_vec_y[i])]).flatten()
                j += self.n_y
            if self.n_d > 0:
                for i in range(self.n_u_gamma):
                    F_u[k + i, :] = np.array([np.cos(np.pi * (m + 1) * self.u_gamma[i, :] / self.L_vec_u[i]),
                                 np.sin(np.pi * (m + 1) * self.u_gamma[i, :] / self.L_vec_u[i])]).flatten()
                k += self.n_u_gamma

        # matrices -> gamma-vector
        F = np.concatenate([F_u.flatten(), F_y.flatten()], axis=0)

        # expand by binary terms
        for b in self.u_binary:
            F_1 = F * b
            F_2 = F * (1 - b)
            F = np.concatenate([F_1, F_2], axis=0)

        return F

class LavaLaplace(LavaBase):
    """LAVA model using ARX nominal model and a nonlinear Laplace base function.
    """

    def __init__(
        self, n_y, n_u_phi, n_u_gamma, n_u_binary, n_a, n_b, n_c, n_d, M, L, L_vec_u, L_vec_y):
        super().__init__(n_y, n_u_phi, n_u_gamma, n_u_binary, n_a, n_b, n_c, n_d, M, L, L_vec_u, L_vec_y)
        self.n_gamma = M ** (n_y * n_c + n_u_gamma * n_d) * (n_u_binary + 1)
        self.M = M
        self.reset_id()
        self.reset_var()

    def _get_basis(self):
        """Get the nonlinear Laplace basis expansion.
        """
        x = np.concatenate((self.y_gamma.flatten(), self.u_gamma.flatten()), axis=0)
        D = x.shape[0]
        P = np.ones(self.M ** D)
        j_vec = np.ones(D)
        L_vec = np.concatenate((np.outer(self.L_vec_y, np.ones(self.n_c)).flatten(),
                                np.outer(self.L_vec_u, np.ones(self.n_d)).flatten()),
                               axis=0)

        for c in range(self.M ** D):
            for k in range(D):
                P[c] *= np.sin(np.pi * j_vec[k] * (x[k] + L_vec[k])
                               / (2 * L_vec[k])) / np.sqrt(L_vec[k])
            j_vec[0] = j_vec[0] + 1
            if D > 1:
                for k in range(1, D):
                    if np.mod(j_vec[k - 1], self.M + 1) == 0:
                        j_vec[k - 1] = 1
                        j_vec[k] = j_vec[k] + 1

        # expand by binary terms
        for b in self.u_binary:
            P_1 = P * b
            P_2 = P * (1 - b)
            P = np.concatenate([P_1, P_2], axis=0)

        return P
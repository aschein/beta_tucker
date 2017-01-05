#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import sys
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, log

from mcmc_model cimport MCMCModel
from sample cimport _sample_gamma, _sample_beta, _sample_dirichlet
from bessel cimport _sample as _sample_bessel
from bessel cimport _mode as _bessel_mode


cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])
    double gsl_ran_beta_pdf(double x, double a, double b)

cdef extern from "gsl/gsl_sf_psi.h" nogil:
    double gsl_sf_psi(double)


cdef double _gamma_geo_ev(double shp, double sca) nogil:
    """
    Computes the geometric expected value of a Gamma random variable.
                        G[X] = exp[E[logX]]

    Useful as a proxy for the mode when the shape < 1.
    """
    return exp(gsl_sf_psi(shp) + log(sca))


cdef double _beta_geo_ev(double shp1, double shp2) nogil:
    """
    Computes the geometric expected value of a Beta random variable.
                        G[X] = exp[E[logX]]

    Useful as a proxy for the mode when one of the shapes < 1.
    """
    return exp(gsl_sf_psi(shp1) - gsl_sf_psi(shp1 + shp2))


cdef class BTD(MCMCModel):

    cdef:
        int I, J, C, K, point_est, dirichlet, debug
        double b, e, f, eta, gam, theta_, phi_
        double[::1] c_I, P_C, lambda_I_
        double[:,::1] Theta_IC, Phi_JK, Pi_CK, data_IJ, P_CK, shp_IC, shp_JK
        double[:,:,::1] Lambda_2IJ, Z_2IJ, Z_2JC, shp_2CK
        int[::1] Y_I_
        int[:,:,::1] Y_2IJ
        unsigned int[::1] N_C, N_I, N_J
        unsigned int[:,::1] N_CK

    def __init__(self, int I, int J, int C, int K, double b=1.,
                 double e=0.1, double f=1., double eta=1., double gam=1.,
                 int dirichlet=0, int point_est=0, int debug=0, object seed=None):

        super(BTD, self).__init__(seed)

        # Params
        self.I = self.param_list['I'] = I
        self.J = self.param_list['J'] = J
        self.C = self.param_list['C'] = C
        self.K = self.param_list['K'] = K
        self.b = self.param_list['b'] = b
        self.e = self.param_list['e'] = e
        self.f = self.param_list['f'] = f
        self.gam = self.param_list['gam'] = gam
        self.eta = self.param_list['eta'] = eta
        self.dirichlet = self.param_list['dirichlet'] = dirichlet
        self.point_est = self.param_list['point_est'] = point_est
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.c_I = np.ones(I)
        self.Phi_JK = np.zeros((J, K))
        self.Theta_IC = np.zeros((I, C))
        self.Pi_CK = np.zeros((C, K))
        self.Lambda_2IJ = np.zeros((2, I, J))
        self.shp_2CK = np.zeros((2, C, K))
        self.shp_IC = np.zeros((I, C))
        self.shp_JK = np.zeros((J, K))
        self.Y_2IJ = np.zeros((2, I, J), dtype=np.int32)

        # Cache 
        self.phi_ = 0
        self.theta_ = 0
        self.lambda_I_ = np.zeros(I)
        self.Y_I_ = np.zeros(I, dtype=np.int32)

        # Auxiliary data structures
        self.Z_2IJ = np.zeros((2, I, J))
        self.Z_2JC = np.zeros((2, J, C))
        self.P_C = np.zeros(C)
        self.P_CK = np.zeros((C, K))
        self.N_C = np.zeros(C, dtype=np.uint32)
        self.N_CK = np.zeros((C, K), dtype=np.uint32)
        self.N_I = np.zeros(I, dtype=np.uint32)
        self.N_J = np.zeros(J, dtype=np.uint32)

        # Copy of the data
        self.data_IJ = np.zeros((I, J))

    def fit(self, data, num_itns=1000, verbose=True, print_every=1,
             initialize=True, burnin={}, point_est=False):
        if isinstance(data, np.ndarray):
            data = np.ma.core.MaskedArray(data, mask=None)
        assert isinstance(data, np.ma.core.MaskedArray)

        assert data.shape == (self.I, self.J)
        assert (0 <= data).all() and (data <= 1).all()

        filled_data = data.filled(fill_value=-1) # missing values are -1
        self.data_IJ = np.copy(filled_data)
        self._update_Lambda_2IJ()

        self.print_every = print_every
        self.point_est = int(point_est)

        if initialize:
            if verbose:
                print '\nINITIALIZING...\n'
            
            tmp = self.gam
            
            self.gam = 1.
            self._init_state()
            for gam in [0.1, 1., 2., tmp]:
                self.gam = gam
                self._update(num_itns=10, verbose=int(verbose), burnin={})
            self.total_itns = 0
        
        if verbose:
            print '\nSTARTING INFERENCE...\n'
        self._update(num_itns=num_itns, verbose=int(verbose), burnin=burnin)

    def reconstruct(self, subs=()):
        # TODO: Compute geometric expected value.
        shp1_P, shp2_P = self.get_predictive_dist(subs=subs)
        return shp1_P / (shp1_P + shp2_P)

    def get_predictive_dist(self, subs=()):
        self._tucker_prod()
        b, gam = self.b, self.gam
        Z_2IJ = np.array(self.Z_2IJ)
        Z_m_P = Z_2IJ[0][subs]
        Z_u_P = Z_2IJ[1][subs]
        return (b + gam * Z_m_P), (b + gam * Z_u_P)

    def set_hyperparams(self, param_list):
        for k, v in param_list.iteritems():
            if k == 'b':
                self.b = self.param_list['b'] = v
            elif k == 'e':
                self.e = self.param_list['e'] = v
            elif k == 'f':
                self.f = self.param_list['f'] = v
            elif k == 'gam':
                self.gam = self.param_list['gam'] = v
            elif k == 'eta':
                self.eta = self.param_list['eta'] = v
            else:
                print 'Unrecognized hyperparameter: %s' % k

    def set_state(self, state):
        # TODO: Factor this out into MCMCModel
        for k, v, _ in self._get_variables():
            if k not in state.keys():
                print '%s not in given state.' % v
            else:
                dim = np.ndim(v)
                assert 1 <= dim <= 3
                if dim == 1:
                    v[:] = state[k]
                elif dim == 2:
                    v[:, :] = state[k]
                elif dim == 3:
                    v[:] = state[k]
            
            self.phi_ = np.sum(self.Phi_JK)
            self.theta_ = np.sum(self.Theta_IC)
            for i in xrange(self.I):
                self.Y_I_[i] = np.sum(self.Y_2IJ[:, i, :])
                self.lambda_I_[i] = np.sum(self.Lambda_2IJ[:, i, :])

    cdef void _print_state(self):
        cdef:
            int num_tokens
            double sparse

        print np.max(self.Y_2IJ), np.sum(self.Y_2IJ)

        sparse = np.count_nonzero(self.Y_2IJ) / float(self.Y_2IJ.size)
        num_tokens = np.sum(self.Y_2IJ)

        print 'ITERATION %d: sparsity: %f, num_tokens: %d\n' % \
              (self.total_itns, sparse, num_tokens)

    cdef void _init_state(self):
        """
        Initialize internal state.
        """

        cdef:
            double ev, smoothness, tmp_e, tmp_f
        
        ev = 0.5
        smoothness = 100.

        tmp_e = self.e
        tmp_f = self.f
        self.e = smoothness
        self.f = smoothness / ev
        self._generate_before_Y_2ICKJ()
        self.e = tmp_e
        self.f = tmp_f
        self.c_I[:] = 1.

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """

        return [('Lambda_2IJ', self.Lambda_2IJ, self._update_Lambda_2IJ),
                ('Y_2IJ', self.Y_2IJ, self._update_Y_2IJ),
                ('shp_2CK', self.shp_2CK, self._update_Y_2ICKJ),
                ('shp_IC', self.shp_IC, lambda x: None),
                ('shp_JK', self.shp_JK, lambda x: None),
                ('Theta_IC', self.Theta_IC, self._update_Theta_IC),
                ('Phi_JK', self.Phi_JK, self._update_Phi_JK),
                ('Pi_CK', self.Pi_CK, self._update_Pi_CK),
                ('c_I', self.c_I, self._update_c_I)]

    cdef void _tucker_prod(self) nogil:
        """
        Compute the Tucker product of Theta_IC, Pi_CK, Phi_JK.
        """
        cdef:
            int i, j, c, k
            double theta_ic, pi_ck, phi_jk
        
        self.Z_2IJ[:] = 0
        self.Z_2JC[:] = 0
        for j in range(self.J):
            for k in range(self.K):
                phi_jk = self.Phi_JK[j, k]
                for c in range(self.C):
                    pi_ck = self.Pi_CK[c, k]
                    self.Z_2JC[0, j, c] += phi_jk * pi_ck
                    self.Z_2JC[1, j, c] += phi_jk * (1-pi_ck)

        for i in range(self.I):
            for c in range(self.C):
                theta_ic = self.Theta_IC[i, c]
                for j in range(self.J):
                    self.Z_2IJ[0, i, j] += theta_ic * self.Z_2JC[0, j, c]
                    self.Z_2IJ[1, i, j] += theta_ic * self.Z_2JC[1, j, c]

        if self.gam != 1:
            for i in range(self.I):
                for j in range(self.J):
                    self.Z_2IJ[0, i, j] *= self.gam
                    self.Z_2IJ[1, i, j] *= self.gam            

    cpdef void generate_state(self):
        """
        Thin wrapper for _generate_state(...).
        """
        self._generate_state()

    cdef void _generate_state(self):
        self._generate_before_Y_2ICKJ()
        self._generate_Y_2ICKJ()

    cdef void _generate_before_Y_2ICKJ(self):
        """
        Generate internal state up to the latent Poisson counts.
        """

        cdef:
            int i, j, c, k
            double e, f, theta_ic, phi_jk, pi_ck
            gsl_rng * rng

        rng = self.rng
        e = self.e
        f = self.f
        eta = self.eta

        for c in range(self.C):
            for k in range(self.K):
                self.Pi_CK[c, k] = _sample_beta(rng, eta, eta)
        
        for i in range(self.I):
            self.c_I[i] = _sample_gamma(rng, e, 1. / f)

        if self.dirichlet == 1:
            self.theta_ = self.I
            self.shp_IC[:] = e
            for i in range(self.I):
                _sample_dirichlet(rng, self.shp_IC[i], self.Theta_IC[i])

            self.phi_ = self.J
            self.shp_JK[:] = e
            for j in range(self.J):
                _sample_dirichlet(rng, self.shp_JK[j], self.Phi_JK[j])

        else:
            self.theta_ = 0
            for i in range(self.I):
                for c in range(self.C):
                    theta_ic = _sample_gamma(rng, e, 1. / f)
                    self.Theta_IC[i, c] = theta_ic
                    self.theta_ += theta_ic

            self.phi_ = 0
            for j in range(self.J):
                for k in range(self.K):
                    phi_jk = _sample_gamma(rng, e, 1. / f)
                    self.Phi_JK[j, k] = phi_jk
                    self.phi_ += phi_jk

    cdef void _generate_Y_2ICKJ(self):

        cdef:
            int i, j, c, k, m, y_mck, y_mckj, y_mickj
            double gam, theta_ic, phi_jk, pi_ck, z_mck, z_mij
            gsl_rng * rng

        rng = self.rng

        if (self.I * self.J) > 100:
            self.Y_I_[:] = 0
            self._tucker_prod()
            for i in range(self.I):
                for j in range(self.J):
                    for m in range(2):
                        z_mij = self.Z_2IJ[m, i, j]
                        y_mij = gsl_ran_poisson(rng, z_mij)
                        self.Y_2IJ[m, i, j] = y_mij
                        self.Y_I_[i] += y_mij
            self._update_Y_2ICKJ()
        
        else:
            # This version is used for debugging (doesn't call update_Y_2ICKJ)
            self.shp_2CK[:] = self.eta
            self.shp_IC[:] = self.e
            self.shp_JK[:] = self.e
            self.Y_2IJ[:] = 0
            self.Y_I_[:] = 0
            gam = self.gam
            for i in range(self.I):
                for c in range(self.C):
                    theta_ic = self.Theta_IC[i, c]
                    for k in range(self.K):
                        pi_ck = self.Pi_CK[c, k]
                        for j in range(self.J):
                            phi_jk = self.Phi_JK[j, k]
                            for m in range(2):
                                z_mickj = gam * theta_ic * phi_jk
                                z_mickj *= pi_ck if m == 0 else (1-pi_ck)
                                y_mickj = gsl_ran_poisson(rng, z_mickj)
                                if y_mickj == 0:
                                    continue
                                self.shp_2CK[m, c, k] += y_mickj
                                self.shp_IC[i, c] += y_mickj
                                self.shp_JK[j, k] += y_mickj
                                self.Y_2IJ[m, i, j] += y_mickj
                                self.Y_I_[i] += y_mickj

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """

        cdef:
            int i, j
            double shp, sca, l_ij

        self.lambda_I_[:] = 0
        for i in range(self.I):
            sca = 1. / self.c_I[i]
            for j in range(self.J):
                shp = self.b + self.Y_2IJ[0, i, j]
                self.Lambda_2IJ[0, i, j] = _sample_gamma(self.rng, shp, sca)

                shp = self.b + self.Y_2IJ[1, i, j]
                self.Lambda_2IJ[1, i, j] = _sample_gamma(self.rng, shp, sca)

                l_ij = self.Lambda_2IJ[0, i, j] + self.Lambda_2IJ[1, i, j]
                self.data_IJ[i, j] = self.Lambda_2IJ[0, i, j] / l_ij
                self.lambda_I_[i] += l_ij

    cdef void _update_Lambda_2IJ(self):
        cdef:
            int i, j, m
            double shp1_ij, shp2_ij, sca, l_ij, beta_ij

        self.lambda_I_[:] = 0
        for i in range(self.I):
            sca = 1. / self.c_I[i]
            for j in range(self.J):
                shp1_ij = self.b + self.Y_2IJ[0, i, j]
                shp2_ij = self.b + self.Y_2IJ[1, i, j]

                if self.debug == 1:
                    assert shp1_ij > 0 and shp2_ij > 0
                    assert np.isfinite(shp1_ij) and np.isfinite(shp2_ij)
                
                beta_ij = self.data_IJ[i, j]
                if beta_ij == -1: # missing value
                    beta_ij = _sample_beta(self.rng, shp1_ij, shp2_ij)

                l_ij = _sample_gamma(self.rng, shp1_ij + shp2_ij, sca)
                self.Lambda_2IJ[0, i, j] = l_ij * beta_ij
                self.Lambda_2IJ[1, i, j] = l_ij * (1-beta_ij)
                self.lambda_I_[i] += l_ij

    cdef void _update_Theta_IC(self):
        cdef:
            int i, c
            double sca, theta_ic

        if self.dirichlet == 0:
            self.theta_ = 0
            sca = 1. / (self.f + self.phi_)
            for i in range(self.I):
                for c in range(self.C):
                    theta_ic = _sample_gamma(self.rng, self.shp_IC[i, c], sca)
                    self.Theta_IC[i, c] = theta_ic
                    self.theta_ += theta_ic
        else:
            self.theta_ = self.I
            for i in range(self.I):
                _sample_dirichlet(self.rng, self.shp_IC[i], self.Theta_IC[i])
                
                if self.debug == 1:
                    assert self.Theta_IC[i, 0] != -1

    cdef void _update_Phi_JK(self):
        cdef:
            int j, k
            double sca, phi_jk

        if self.dirichlet == 0:
            self.phi_ = 0
            sca = 1. / (self.f + self.theta_)
            for k in range(self.K):
                for j in range(self.J):
                    phi_jk = _sample_gamma(self.rng, self.shp_JK[j, k], sca)
                    self.Phi_JK[j, k] = phi_jk
                    self.phi_ += phi_jk
        else:
            self.phi_ = self.J
            for j in range(self.J):
                _sample_dirichlet(self.rng, self.shp_JK[j], self.Phi_JK[j])
                if self.debug == 1:
                    assert self.Phi_JK[j, 0] != -1

    cdef void _update_Pi_CK(self):
        cdef:
            int c, k
            double shp1_ck, shp2_ck

        for c in range(self.C):
            for k in range(self.K):
                shp1_ck = self.shp_2CK[0, c, k]
                shp2_ck = self.shp_2CK[1, c, k]

                if self.debug == 1:
                    assert shp1_ck > 0 and shp2_ck > 0
                    assert np.isfinite(shp1_ck) and np.isfinite(shp2_ck)

                self.Pi_CK[c, k] = _sample_beta(self.rng, shp1_ck, shp2_ck)

    cdef void _update_c_I(self):
        cdef:
            int i
            double tmp, shp, sca

        tmp = self.e + 2 * self.b * self.J
        for i in range(self.I):
            shp = tmp + self.Y_I_[i]
            sca = 1. / (self.f + self.lambda_I_[i])

            if self.debug == 1:
                assert shp > 0 and sca > 0
                assert np.isfinite(shp) and np.isfinite(sca)

            self.c_I[i] = _sample_gamma(self.rng, shp, sca)
    
    cdef void _update_Y_2IJ(self):
        cdef:
            int i, j, m, y_mij
            double c_i, shp, sca, z_mij, beta_ij

        self.Y_I_[:] = 0

        self._tucker_prod()
        shp = self.b - 1

        for i in range(self.I):
            c_i = self.c_I[i]
            for j in range(self.J):
                for m in range(2):
                    z_mij = self.Z_2IJ[m, i, j]
                    
                    beta_ij = self.data_IJ[i, j]
                    if beta_ij == -1:  # missing data
                        y_mij = gsl_ran_poisson(self.rng, self.gam * z_mij) 
                    
                    elif (beta_ij == 0. and m == 0) or (beta_ij == 1. and m == 1):
                        y_mij = 0

                    else:
                        sca = 2 * sqrt(c_i * z_mij * self.Lambda_2IJ[m, i, j])

                        if self.debug == 1:
                            assert shp > -1 and sca > 0

                        if self.point_est == 1:
                            y_mij = _bessel_mode(shp, sca)
                        else:
                            y_mij = _sample_bessel(self.rng, shp, sca)
                        
                        if self.debug == 1:
                            assert y_mij >= 0
                        
                    self.Y_2IJ[m, i, j] = y_mij
                    self.Y_I_[i] += y_mij
    
    cdef void _update_Y_2ICKJ(self) nogil:
        cdef:
            int C, K, i, j, c, k, m, y_mij, y_micj, y_mickj
            double phi_jk, pi_ck
            double[::1] P_C, P_K
            unsigned int[::1] N_C, N_K
            gsl_rng * rng

        C = self.C
        K = self.K
        P_C = self.P_C
        N_C = self.N_C
        rng = self.rng

        self.shp_2CK[:] = self.eta
        self.shp_IC[:] = self.e
        self.shp_JK[:] = self.e

        for i in range(self.I):
            for j in range(self.J):
                for m in range(2):
                    y_mij = self.Y_2IJ[m, i, j]
                    if y_mij == 0:
                        continue    

                    for c in range(C):
                        self.P_C[c] = 0
                        for k in range(K):
                            pi_ck = self.Pi_CK[c, k]
                            self.P_CK[c, k] = pi_ck if m == 0 else (1-pi_ck)
                            self.P_CK[c, k] *= self.Phi_JK[j, k]
                            self.P_C[c] += self.P_CK[c, k]
                        self.P_C[c] *= self.Theta_IC[i, c]

                    gsl_ran_multinomial(rng, C, y_mij, &P_C[0], &N_C[0])

                    for c in range(C):
                        y_micj = self.N_C[c]
                        if y_micj == 0:
                            continue
                        
                        P_K = self.P_CK[c]
                        N_K = self.N_CK[c]
                        gsl_ran_multinomial(rng, K, y_micj, &P_K[0], &N_K[0])
                        
                        for k in range(K):
                            y_mickj = self.N_CK[c, k]
                            if y_mickj == 0:
                                continue

                            self.shp_2CK[m, c, k] += y_mickj
                            self.shp_IC[i, c] += y_mickj
                            self.shp_JK[j, k] += y_mickj
                            

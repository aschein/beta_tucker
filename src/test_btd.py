import numpy as np
import numpy.random as rn
import scipy.stats as st

from btd import BTD
from IPython import embed

if __name__ == '__main__':
    I = 5
    J = 4
    C = 3
    K = 3
    b = 0.1
    e = 0.5
    f = 2.
    eta = 0.8
    gam = 25.
    dirichlet = 1
    debug = 1

    seed = rn.randint(10000)
    print seed

    model = BTD(I=I, J=J, C=C, K=K, b=b, e=e, f=f, eta=eta,
                gam=gam, dirichlet=dirichlet,
                seed=seed, debug=debug)

    burnin = {'Lambda_2IJ': 0,
              'Y_2IJ': 0,
              'shp_2CK': 0,
              'shp_IC': 0,
              'shp_JK': 0,
              'Theta_IC': 0,
              'Phi_JK': 0,
              'Pi_CK': 0,
              'c_I': 0}

    var_funcs = {}
    if dirichlet == 1:
        entropy_funcs = {'Entropy min': lambda x: np.min(st.entropy(x)),
                         'Entropy max': lambda x: np.max(st.entropy(x)),
                         'Entropy mean': lambda x: np.mean(st.entropy(x)),
                         'Entropy var': lambda x: np.var(st.entropy(x))}

        var_funcs = {'Theta_IC': entropy_funcs,
                     'Phi_JK': entropy_funcs}

    model.schein(25000, var_funcs=var_funcs, burnin=burnin)

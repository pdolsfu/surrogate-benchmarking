# contains all the function definitions
import numpy as np

class true_f1:              # CHANGE HERE 
    def __init__(self, ndim):
        self.ndim = ndim
        self.xlimits = np.tile([[-3.0, 2.0]], (ndim, 1))   
    def __call__(self, X):
        X = np.atleast_2d(X)
        assert X.shape[1] == 10, "Input must have 10 columns"

        term1 = (X[:, 0] - 1)**2
        term2 = (X[:, 9] - 1)**2

        term3 = np.zeros(X.shape[0])
        for i in range(9):  # the term with the riemann sum
            term3 += (10 - (i + 1)) * (X[:, i]**2 - X[:, i+1])**2

        fx = term1 + term2 + 10 * term3
        return fx
    
class true_f2:             
    def __init__(self, ndim):
        self.ndim = ndim
        self.xlimits = np.tile([[-3.0, 3.0]], (ndim, 1))  
    def __call__(self, X):
        X = np.atleast_2d(X)
        assert X.shape[1] == 10, "Input must have 10 columns"  
        
        i = np.arange(1, 11)  # i = [1, 2, ..., 10]
        S = np.sum(i**3 * (X - 1)**2, axis=1)
        return S**3

class true_f3:            
    def __init__(self, ndim):
        self.ndim = ndim
        self.xlimits = np.tile([[-3.0, 5.0]], (ndim, 1))       

    def __call__(self, x):
        x = np.asarray(x)
        assert x.shape[1] == 20, "Each input vector must be 20-dimensional"

        xi = x[:, :10]
        xi_plus_10 = x[:, 10:]

        term1 = 100 * (xi - xi_plus_10)**2
        term2 = (xi - 1)**2

        return np.sum(term1 + term2, axis=1)


class true_f4:
    def __init__(self, ndim):
        self.ndim = ndim
        self.xlimits = np.tile([[-3.0, 5.0]], (ndim, 1))         # BOUNDS VARYING DEPENDS ON FUNCTION
    def __call__(self, x):
        x = np.asarray(x)
        assert x.shape[1] == 20, "Each input vector must be 20-dimensional"

        # Initialize output
        out = np.zeros(x.shape[0], dtype=float)

        # Loop over i = 0..4 (which corresponds to terms i=1..5 in the formula)
        for i in range(5):
            xi   = x[:,     i]
            xi5  = x[:, i + 5]
            xi10 = x[:, i + 10]
            xi15 = x[:, i + 15]

            term1 = 100.0 * (xi**2  + xi5)**2
            term2 =        (xi - 1)**2
            term3 =  90.0 * (xi10**2 + xi15)**2
            term4 =        (xi10 - 1)**2
            term5 =  10.1 * ((xi5  - 1)**2 + (xi15 - 1)**2)
            term6 =  19.8 * (xi5  - 1) * (xi15 - 1)

            out += (term1 + term2 + term3 + term4 + term5 + term6)

        return out    


class true_f5:
    """
    f(x) = sum_{i=1}^5 [
        (x_i + 10*x_{i+5})^2
      + 5*(x_{i+10} - x_{i+15})^2
      + (x_{i+5} - 2*x_{i+10})^4
      + 10*(x_i - x_{i+15})^4
    ]
    with -2 <= x_j <= 5 for j=1…20.
    """
    def __init__(self, ndim=20):
        self.ndim = ndim
        # same bound for all dims:
        self.xlimits = np.tile([[-2.0, 5.0]], (ndim, 1))

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim, \
               f"Each input vector must be {self.ndim}-dimensional"

        # split into blocks of length 5
        xi    = x[:,   :5]
        x5    = x[:,  5:10]
        x10   = x[:, 10:15]
        x15   = x[:, 15:20]

        term1 = (xi + 10*x5)**2
        term2 = 5*(x10 - x15)**2
        term3 = (x5 - 2*x10)**4
        term4 = 10*(xi - x15)**4

        return np.sum(term1 + term2 + term3 + term4, axis=1)


class true_f7:
    """
    f7(x) = ( x^T A x )^2,  A = diag(1,2,3,...,30)
    bounds: -2 ≤ x_i ≤ 3
    """
    def __init__(self, ndim=30):
        self.ndim = ndim
        # -2 ≤ x_i ≤ 3 for all i
        self.xlimits = np.tile([[-2.0, 3.0]], (ndim, 1))
        # store the diagonal entries of A
        self.A_diag = np.arange(1, ndim + 1)

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim, \
            f"Each input vector must be {self.ndim}-dimensional"
        # x^T A x for each row is sum_i A_diag[i] * x[:,i]^2
        quad = np.sum(self.A_diag * x**2, axis=1)
        return quad**2


class true_f8:
    """
    f8(x) = sum_{i=1}^{29} [100*(x_{i+1} − x_i^2)^2 + (1 − x_i)^2]
    bounds: -2 ≤ x_i ≤ 2
    """
    def __init__(self, ndim=30):
        assert ndim >= 2, "Need at least 2 dimensions for Rosenbrock"
        self.ndim = ndim
        # -2 ≤ x_i ≤ 2 for all i
        self.xlimits = np.tile([[-2.0, 2.0]], (ndim, 1))

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim, \
            f"Each input vector must be {self.ndim}-dimensional"
        xi    = x[:, :-1]   # x1…x29
        xnext = x[:, 1:]    # x2…x30
        term1 = 100.0 * (xnext - xi**2)**2
        term2 = (1.0 - xi)**2
        return np.sum(term1 + term2, axis=1)


class true_f9:
    """
    f9(x) = xᵀ A x − 2 x₁,
    where A is the tridiagonal matrix
        [ 1  −1         0  …         0 ]
        [−1   2  −1     0  …         0 ]
        [ 0  −1   2  −1  0  …         0 ]
        [       …             …       ]
        [ 0  …        −1   2  −1      ]
        [ 0  …         0  −1   2     ]
    bounds:  0 ≤ xᵢ ≤ 25 for i=1…20.
    """
    def __init__(self, ndim=20):
        self.ndim = ndim
        # same [0,25] bounds on every coordinate
        self.xlimits = np.tile([[0.0, 25.0]], (ndim, 1))
        # store the diagonal of A: 1 for the first entry, 2 for all the rest
        self._A_diag = np.full(ndim, 2.0)
        self._A_diag[0] = 1.0

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim, \
            f"Each input vector must be {self.ndim}-dimensional"
        # main diagonal term: sum_i A_ii * x_i^2
        main = np.sum(self._A_diag * x**2, axis=1)
        # off-diagonal neighbors: 2 * sum_{i=1 to d-1} A_{i,i+1} x_i x_{i+1},
        # but A_{i,i+1} = -1, so that's -2 * sum x_i*x_{i+1}
        off  = -2.0 * np.sum(x[:, :-1] * x[:, 1:], axis=1)
        # quadratic form:
        quad = main + off
        # subtract the linear term 2*x₁
        return quad - 2.0 * x[:, 0]

class true_f10:
    def __init__(self, ndim=20):
        self.ndim = ndim
        self.xlimits = np.tile([[0.0, 5.0]], (ndim, 1))

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim
        term1 = np.sum(x**2, axis=1)
        s = 0.5 * np.sum(x, axis=1)
        return term1 + s**2 + s**4


class true_f11:
    def __init__(self, ndim=11):
        self.ndim = ndim
        # bounds for the 11 “active” parameters; extras unused
        self.xlimits = np.zeros((ndim, 2))
        self.xlimits[0]    = [0.0, 1.6]
        self.xlimits[1:5]  = [0.0, 2.0]
        self.xlimits[5:8]  = [2.0, 8.0]
        self.xlimits[8]    = [1.0, 6.0]
        self.xlimits[9:11] = [4.5, 6.0]
        # precompute t₁…t₆₅ and the data y₁…y₆₅
        self.t = 0.1 * np.arange(65)
        self.y = np.array([
            1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746,
            0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649,
            0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395,
            0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653,
            0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739,
            0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054
        ])

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim
        p = x[:, :11]
        a1, a2, a3, a4 = p[:,0], p[:,1], p[:,2], p[:,3]
        b1, b2, b3, b4 = p[:,4], p[:,5], p[:,6], p[:,7]
        c1, c2, c3   = p[:,8], p[:,9], p[:,10]
        # four‐term model evaluated at each t
        M = (
            a1[:,None] * np.exp(-b1[:,None] * self.t[None,:])
          + a2[:,None] * np.exp(-b2[:,None] * (self.t[None,:] - c1[:,None])**2)
          + a3[:,None] * np.exp(-b3[:,None] * (self.t[None,:] - c2[:,None])**2)
          + a4[:,None] * np.exp(-b4[:,None] * (self.t[None,:] - c3[:,None])**2)
        )
        # sum of squared residuals
        return np.sum((self.y[None,:] - M)**2, axis=1)


class true_f12:
    def __init__(self, ndim=11):
        self.ndim = ndim
        self.xlimits = np.tile([[0.1, 100.0]], (ndim, 1))
        self.alpha = np.array([
            -0.00133172, -0.002270927, -0.00248546,
            -4.67,       -4.671973,   -0.00814,
            -0.008092,   -0.005,      -0.000909,
            -0.00088,    -0.00119
        ])

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim
        return 1e6 * np.prod(x[:, :len(self.alpha)]**self.alpha, axis=1)


class true_f13:
    def __init__(self, ndim=14):
        self.ndim = ndim
        # default lower bound 1e-4 everywhere, then tighten first 5 dims to 0.04
        self.xlimits = np.tile([[1e-4, 0.03]], (ndim, 1))
        self.xlimits[:5,1] = 0.04
        self.a = np.array([
            12842.275, 634.25,  634.25,  634.125,
             1268,    633.875, 633.75,  1267,
             760.05,   33.25, 1266.25, 632.875,
             394.46,   940.838
        ])

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim
        return np.sum(self.a / x[:, :len(self.a)], axis=1)


class true_f14:
    def __init__(self, ndim=30):
        self.ndim = ndim
        self.xlimits = np.tile([[-2.0, 2.0]], (ndim, 1))
        i = np.arange(1, ndim+1)[:, None]
        j = np.arange(1, ndim+1)[None, :]
        v = np.sqrt(j**2 + i/j)
        trig = np.sin(np.log(v))**5 + np.cos(np.log(v))**5
        # precompute the Σ_j vᵢⱼ·[sin(log(vᵢⱼ))⁵+cos(log(vᵢⱼ))⁵] term
        self.C = np.sum(v * trig, axis=1)
        self.i = np.arange(1, ndim+1)

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.ndim
        α = 420.0 * x + (self.i - 15.0)**3 + self.C[None, :]
        return np.sum(α**2, axis=1)

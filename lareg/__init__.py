import numpy as np

__all__ = ["lareg"]


class ndfunc:
    '''
    class implementing a callable array of functions
    calling with scalar argument will return numpy array of return values of the individual functions
    reshaping and transposing in numpy style is possible
    '''

    def __init__(self, functions):
        self.functions = np.asarray(functions)
        self.f = np.vectorize(lambda f, y: f(y))

    def __call__(self, x):
        return self.f(self.functions, x)

    def reshape(self, *shape):
        freedims = sum(int(i < 0) for i in shape)
        if freedims > 1:
            raise ValueError('can only specify one unknown dimension')
        elif freedims == 0:
            if np.product(shape) != len(self.functions):
                raise ValueError(
                    'cannot reshape array of size {} into shape {}'.format(
                        self.functions.shape, shape))
        return ndfunc(self.functions.reshape(shape))

    def __iter__(self):
        return iter(self.functions)

    def __getattr__(self, name):
        if name == 'T':
            return ndfunc(self.functions.T)


def lareg(flist, x, y, dy=None, V=None, justfit=False):
    '''
    Linear fit for list of function
    
    Parameters
    ----
    flist: list or ndarray of functions
    x: list or array
    x-values for the fit
    y: list or ndarray
    function values for the corresponding x
    dy: list or ndarray
    uncorrelated errors of the y-values
    V: ndarray
    covariance matrix for the y-values
    
    Returns
    ----
    [a,V_a,R^2,regfunc]: list
    a: ndarray
    best coefficients for linear fit with functions in flist
    V_a: ndarray
    covariance matrix for a
    R^2: float
    determination coefficient for the fit
    regfunc: function
    function of linear fit
    '''
    func = ndfunc(flist)
    N = len(x)
    x, y = map(lambda x: np.asarray(x).reshape(-1),
               (x, y))  # working with numpy arrays
    C = ndfunc(flist)(x.reshape(
        -1, 1))  # Matrix with values of all functions at all x
    if dy is None and V is None:  # calculate errors from deviation
        fit = lareg(flist, x, y, np.full_like(
            y, 1))  # doing fit for equally weighed uncorrelated errors
        if justfit:
            return fit
        d = len(flist)
        V = np.diag(
            np.full_like(x,
                         np.sum((fit["fitfunc"](x) - y)**2) / (N - d)))  # calculating deviation
        fit[1] = np.linalg.inv(C.T.dot(np.linalg.inv(V)).dot(C))
        return fit
    elif V is None:  # build covariance matrix from uncorrelated errors
        V = np.diag(np.asarray(dy).reshape(-1)**2)
    Vinv = np.linalg.inv(V)
    V_a = np.linalg.inv(C.T.dot(Vinv).dot(C))  # Vₐ=(Cᵀ·V⁻¹·C)⁻¹
    M = V_a.dot(C.T).dot(Vinv)  # M = Vₐ·Cᵀ·V⁻¹
    a = M.dot(y)  # a = M·y
    regfunc = np.vectorize(
        lambda x: np.sum(func(x) * a))  # buildung the fitted function
    TSS = np.sum(
        (y - y.mean())**2)  # calculations for the determination coefficient
    RSS = np.sum((regfunc(x) - y)**2)
    return a, V_a, 1 - RSS / TSS, regfunc

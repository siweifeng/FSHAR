import numpy as np
from scipy import linalg, optimize

def lls_l2l1(A, b, penalty, groups, max_iter, rtol=1e-6,
             verbose=False):
    """
    Linear least-squares with l2/l1 regularization solver.
    Solves problem of the form:
               .5 * |Xb - y| + n_samples * penalty * Sum(|b_j|)
    where |.| is the l2-norm and b_j is the coefficients of b in the
    j-th group. This is commonly known as the `group lasso`.
    Parameters
    ----------
    A : array of shape (n_samples, n_features)
        Design Matrix.
    b : array of shape (n_samples,)
    penalty : float
        Amount of penalization to use.
    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.
    rtol : float
        Relative tolerance. ensures ||x - x_|| / x_ < rtol,
        where x_ is the approximate solution and x is the
        true solution.
    Returns
    -------
    x : array
        vector of coefficients
    References
    ----------
    "Efficient Block-coordinate Descent Algorithms for the Group Lasso",
    Qin, Scheninberg, Goldfarb
    """

    # .. local variables ..
    A, b, groups, penalty = map(np.asanyarray, (A, b, groups, penalty))
    if len(groups) != A.shape[1]:
        raise ValueError("Incorrect shape for groups")
    x = np.zeros(A.shape[1], dtype=A.dtype)
    penalty = penalty * A.shape[0]

    # .. precompute ..
    H = np.dot(A.T, A)
    group_labels = [groups == i for i in np.unique(groups)]
    H_groups = [np.dot(A[:, g].T, A[:, g]) for g in group_labels]
    eig = [linalg.eigh(H_groups[i]) for i in range(len(group_labels))]
    Ab = np.dot(A.T, b)

    for n_iter in range(max_iter):
        def phi(nabla_, qp2, eigvals, penalty):
            return 1 - np.sum( qp2 / ((nabla_ * eigvals + penalty) ** 2))
        def dphi(nabla_, alpha, eigvals, penalty):
            # .. first derivative of phi ..
            return np.sum((2 * alpha * eigvals) / ((penalty + nabla_ * eigvals) ** 3))
        for i, g in enumerate(group_labels):
            # .. shrinkage operator ..
            eigvals, eigvects = eig[i]
            x_i = x.copy()
            x_i[g] = 0.
            A_residual = np.dot(H[g], x_i) - Ab[g]
            qp = np.dot(eigvects.T, A_residual)
            if penalty < linalg.norm(A_residual, 2):
                initial_guess = 0. # TODO: better initial guess
                nabla = optimize.newton(phi, initial_guess, dphi, tol=1e-3, maxiter=int(1e4),
                        args=(qp ** 2, eigvals, penalty))
                x[g] = - nabla * np.dot(eigvects /  (eigvals * nabla + penalty), qp)
            else:
                x[g] = 0.

        # .. dual gap ..
        if n_iter % 3:
            residual = np.dot(A, x) - b
            group_norm = penalty * np.sum([linalg.norm(x[g], 2)
                         for i, g in enumerate(group_labels)])
            norm_Anu = [linalg.norm(np.dot(H[g], x) - Ab[g]) \
                       for g in group_labels]
            if np.any(norm_Anu > penalty):
                nnu = residual * np.min(penalty / norm_Anu)
            else:
                nnu = residual
            primal_obj =  .5 * np.dot(residual, residual) + group_norm
            dual_obj   = -.5 * np.dot(nnu, nnu) - np.dot(nnu, b)
            dual_gap = primal_obj - dual_obj
            if verbose:
                print('Relative error: %s' % (dual_gap / dual_obj))
            if np.abs(dual_gap / dual_obj) < rtol:
                break

    return x


def check_kkt(A, b, x, penalty, groups):
    """Check KKT conditions for the group lasso
    Returns True if conditions are satisfied, False otherwise
    """
    group_labels = [groups == i for i in np.unique(groups)]
    penalty = penalty * A.shape[0]
    z = np.dot(A.T, np.dot(A, x) - b)
    safety_net = 1e-1 # sort of tolerance
    for g in group_labels:
        if linalg.norm(x[g]) == 0:
            if not linalg.norm(z[g]) < penalty + safety_net:
                return False
        else:
            w = - penalty * x[g] / linalg.norm(x[g], 2)
            if not np.allclose(z[g], w, safety_net):
                return False
    return True


if __name__ == '__main__':
    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    penalty = .1
    groups = np.r_[[0, 0], np.arange(X.shape[1] - 2)]
    coefs = lls_l2l1(X, y, penalty, groups, verbose=True)
    print('KKT conditions verified:', check_kkt(X, y, coefs, penalty, groups))
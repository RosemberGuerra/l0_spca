# Module with functions to estimate the component weights of sparse PCA using 
# l0 penalty.

# Function to compute the objective Original function
def objective_function(C, w, lmda):
    """	
    Compute the objective function.
    Args:
    C: Covariance matrix
    w: weight vector
    regularization: regularization parameter

    Returns:
    obj: objective function value
    """
    # importing needed packages
    import numpy as np
    # computing the objective function
    obj = w.T @ C @ w - lmda *sum(w!=0)
    return obj

# Thresholding function
def thresholding(x, threshold_value):
    """	
    Perform L0 thresholding.
    Args:
    w: weight vector
    threshold_value: threshold value

    Returns:
    w: thresholded weight vector
    """
    # importing needed packages
    import numpy as np
    # thresholding
    new_x = x**2 / np.linalg.norm(x, axis=0) <= threshold_value/2
    x[new_x ] = 0 # thre is an alpha/2 in the algorithm and not in the paper!
    return x

# Function to perform sparse PCA with L0 penalty
def l0_pca(X,w0,regularization = .1, method = "data", max_iter = 1000, tol = 1e-6):
    """	
    Perform PCA with L0 regularization.
    Args:
    X: data matrix, where each row is a sample. 
        X could also be the covariance matrix of the data.
    w0: initial weight vector.
    regularization: regularization parameter
    method: "data" or "cov", indicating whether X is the data matrix or the covariance matrix of the data
    max_iter: maximum number of iterations
    tol: tolerance for convergence

    Returns:
    w: First component weight vector
    obj: objective function value at each iteration
    """
    # importing needed packages
    import numpy as np

    if method == "data":
        # centering the data
        X = X - np.mean(X, axis = 0)
        # computing the covariance matrix
        Sigma = np.cov(X, rowvar = False)
    elif method == "cov":
        Sigma = X
    # Guarantee that w0 a column vector
    w0 = w0.reshape(-1,1)
    Sigma_idn = Sigma - np.identity(Sigma.shape[0])
    iter = 1
    obj = []
    obj.append(objective_function(Sigma, w0, regularization))
    while True:
        w_full =  Sigma_idn @ w0
        # thresholding
        w_sparse = thresholding(w_full, regularization) 
        # normalizing
        if np.linalg.norm(w_sparse) == 0:
            w = w_sparse
            break
        else:
            w = w_sparse/np.linalg.norm(w_sparse)
        if (iter > 3 and np.linalg.norm(w - w0) < tol) or iter > max_iter :
            print("Convergence achieved after", iter, "iterations")
            break
        w0 = w
        iter += 1
        obj.append(objective_function(Sigma, w, regularization))
    return w , np.array(obj).reshape(-1,1)

# Function to extract multiple component weights via deflation

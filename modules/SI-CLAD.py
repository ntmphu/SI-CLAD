import numpy as np
import parametric
import DBSCAN
import util

def SI_CLAD(X, Sigma, minpts, eps, j = None):
    n = X.shape[0]
    d = X.shape[1]

    dbscan_labels, _ = DBSCAN.DBSCAN(eps, minpts).fit(X)
    label = np.array(dbscan_labels)
    O = [i for i in range(label.shape[0]) if label[i] == -1]
    if len(O) == 0 or len(O) == n:
        return None
    if j is None:
        j = np.random.choice(O)
    #contruct eta and sign
    minusO = [i for i in range(label.shape[0]) if label[i] != -1]
    eT_minusO = np.zeros((1, label.shape[0]))
    eT_minusO[:,minusO] = 1
    x = util.vec(X)
    I_d = np.identity(d)
    eT_mean_minusO = np.kron(I_d, eT_minusO)/(n - len(O))
    e_j = np.zeros((1, n))
    e_j[:,j] = 1
    temp = np.kron(I_d, e_j) - eT_mean_minusO
    Xj_meanXminusO = np.dot(temp, x)
    S_obs = np.sign(Xj_meanXminusO) #sign
    etaT = np.dot(S_obs.T, temp)/d
    eta = np.transpose(etaT)
    etaTx = np.dot(etaT, x) #test statistic
    
    etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
    #there is an slightly different in the notation here: X = az + c (instead of X = a + bz)
    a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
    c = np.dot(np.identity(int(n*d)) - np.dot(a,eta.T), x)
    #compute truncated region Z
    truncated_region = parametric.run_parametric(j, n, d, O, S_obs, minpts, eps, a, c)
    cdf = util.pivot_with_specified_interval(truncated_region, eta, etaTx[0][0] , Sigma, 0)
    selective_p_value = 2 * min(cdf, 1 - cdf)
    return selective_p_value


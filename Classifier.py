from iceberk import classifier, mathutil, mpi
import numpy as np

def l2logreg_onevsall(X, Y, gamma, weight = None, **kwargs):
    if Y.ndim == 1:
        Y = classifier.to_one_of_k_coding(Y,0)
    #solver = classifier.SolverStochastic(gamma,
    #    classifier.Loss.loss_multiclass_logreg,
    #    classifier.Reg.reg_l2,
    #    args = {'mode': 'adagrad', 'base_lr': 1e-3, 'minibatch': 100,
    #            'num_iter': 1000},
    #    **kwargs)
    solver = classifier.SolverMC(gamma, classifier.Loss.loss_multiclass_logistic, classifier.Reg.reg_l2, **kwargs)
    #sampler = mathutil.NdarraySampler((X, Y, None))
    return solver.solve(X, Y, weight)

def loss_multiclass_logreg(Y, X, weights):
    X = mpi.COMM.gather(X)
    Y = mpi.COMM.gather(Y)
    if mpi.is_root():
        Y = np.hstack(Y)
        X = np.vstack(X)
        pred = mathutil.dot(X,weights[0])+weights[1]
        return classifier.Loss.loss_multiclass_logistic(classifier.to_one_of_k_coding(Y, 0), pred, None)[0] / float(Y.shape[0])
    else:
        return 0

def get_predictions_logreg(X, weights):
    pred = mathutil.dot(X,weights[0])+weights[1]
    prob = pred - pred.max(axis=1)[:,np.newaxis]
    mathutil.exp(prob, out=prob)
    prob /= prob.sum(axis=1)[:, np.newaxis]
    return prob


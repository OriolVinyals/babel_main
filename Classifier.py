from iceberk import classifier, mathutil, mpi
import numpy as np

class Classifier:
    def __init__(self,Xtrain,Ytrain,Xtest,Ytest):
        self._Xtrain=Xtrain
        self._Ytrain=Ytrain
        self._Xtest=Xtest
        self._Ytest=Ytest
        self.features=Xtrain.keys
        
    def Train(self,feat_list=None):
        return
        

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
    pred = mathutil.dot(X,weights[0])+weights[1]
    local_likelihood = classifier.Loss.loss_multiclass_logistic(classifier.to_one_of_k_coding(Y, 0), pred, None)[0]
    likelihood = mpi.COMM.allreduce(local_likelihood)
    num_data = mpi.COMM.allreduce(len(Y))
    return float(likelihood) / num_data

def get_predictions_logreg(X, weights):
    pred = mathutil.dot(X,weights[0])+weights[1]
    prob = pred - pred.max(axis=1)[:,np.newaxis]
    mathutil.exp(prob, out=prob)
    prob /= prob.sum(axis=1)[:, np.newaxis]
    prob = mpi.COMM.gather(prob)
    if mpi.is_root():
        return np.vstack(prob)
    else:
        return np.zeros((0))


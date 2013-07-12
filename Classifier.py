from iceberk import classifier, mathutil, mpi
import numpy as np

class Classifier:
    def __init__(self,Xtrain,Ytrain):
        self._Xtrain=Xtrain
        self._Ytrain=Ytrain
        self.features=Xtrain.keys()
        
    def Train(self,feat_list=None,type='logreg',gamma=0.0):
        if feat_list==None:
            feat_list=self.features
        self.feat_list=feat_list
        self._gamma=gamma
        Xtrain_feats=np.hstack((self._Xtrain[feat_list[i]] for i in range(len(feat_list))))
        self.m, self.std = classifier.feature_meanstd(Xtrain_feats)
        Xtrain_feats -= self.m
        Xtrain_feats /= self.std
        '''Classifier stage'''
        if type=='linsvm':
            self.w, self.b = classifier.l2svm_onevsall(Xtrain_feats, self._Ytrain, self._gamma)
        elif type=='logreg':
            self.w, self.b = l2logreg_onevsall(Xtrain_feats, self._Ytrain, self._gamma)
        return (self.w,self.b)
    
    def Accuracy(self, X, Y):
        X_feats=np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list))))
        X_feats -= self.m
        X_feats /= self.std
        self.test_accu = classifier.Evaluator.accuracy(Y, np.dot(X_feats,self.w)+self.b)
        return self.test_accu
    
    def loss_multiclass_logreg(self, X, Y):
        X_feats=np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list))))
        X_feats -= self.m
        X_feats /= self.std
        return loss_multiclass_logreg(Y, X_feats, (self.w,self.b))

    def get_predictions_logreg(self, X):
        X_feats=np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list))))
        X_feats -= self.m
        X_feats /= self.std
        return get_predictions_logreg(X_feats, (self.w,self.b))
        

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


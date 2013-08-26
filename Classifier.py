from iceberk import classifier, mathutil, mpi
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer

class Classifier:
    def __init__(self,Xtrain,Ytrain):
        self._Xtrain=Xtrain
        self._Ytrain=Ytrain
        self.features=Xtrain.keys()
        
    def Train(self,feat_list=None,type='logreg',gamma=0.0,domeanstd=True,special_bias=None,add_bias=True, weight=None):
        if feat_list==None:
            feat_list=self.features
        self.feat_list=feat_list
        self._gamma=gamma
        self._type=type
        self._special_bias = special_bias
        self._add_bias = add_bias
        Xtrain_feats = np.ascontiguousarray(np.hstack((self._Xtrain[feat] for feat in feat_list)))
        self.m, self.std = classifier.feature_meanstd(Xtrain_feats)
        if domeanstd==False: #hacky, overwrite the things we computed
            self.m[:] = 0
            self.std[:] = 1
        Xtrain_feats -= self.m
        Xtrain_feats /= self.std
        if special_bias != None:
            Xtrain_feats = np.ascontiguousarray(np.hstack((Xtrain_feats, special_bias)))
        '''Classifier stage'''
        if type=='linsvm':
            self.w, self.b = classifier.svm_onevsall(Xtrain_feats, self._Ytrain, self._gamma, weight = weight, special_bias=special_bias, add_bias=add_bias)
            return (self.w,self.b)
        elif type=='logreg':
            self.w, self.b = l2logreg_onevsall(Xtrain_feats, self._Ytrain, self._gamma, weight = weight, special_bias=special_bias, add_bias=add_bias)
            return (self.w,self.b)
        elif type=='nn_debug':
            if mpi.COMM.Get_size() > 1:
                print 'Warning!!! Running NN training with MPI with more than one Node!'
                #FIXME: Collect X and Y at root to avoid this
#                 prob = mpi.COMM.gather(prob)
#                 if mpi.is_root():
#                     np.vstack(prob)
#                     #Train
#                     mpi.COMM.Bcast(self._nn)
#                 mpi.distribute(prob)                
            DS = ClassificationDataSet( Xtrain_feats.shape[1], 1, nb_classes=2 )
            #for i in range(Xtrain_feats.shape[0]):
            #    DS.addSample( Xtrain_feats[i,:], [self._Ytrain[i]] )
            DS.setField('input', Xtrain_feats)
            DS.setField('target', self._Ytrain[:,np.newaxis])
            DS._convertToOneOfMany()
            self._nn = buildNetwork(DS.indim, 10, DS.outdim, outclass=SoftmaxLayer, fast=True)
            self._nn_trainer = BackpropTrainer( self._nn, dataset=DS, momentum=0.1, verbose=True, weightdecay=gamma, learningrate=0.01, lrdecay=1.0)
            self._nn_trainer.trainOnDataset(DS,epochs=8)
            self._nn_trainer = BackpropTrainer( self._nn, dataset=DS, momentum=0.1, verbose=True, weightdecay=gamma, learningrate=0.001, lrdecay=1.0)
            self._nn_trainer.trainOnDataset(DS,epochs=8)
            self._nn_trainer = BackpropTrainer( self._nn, dataset=DS, momentum=0.1, verbose=True, weightdecay=gamma, learningrate=0.0001, lrdecay=1.0)
            self._nn_trainer.trainOnDataset(DS,epochs=5)
            return self._nn
    
    def Accuracy(self, X, Y, special_bias = None):
        X_feats = np.ascontiguousarray(np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list)))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        if self._type=='linsvm' or self._type=='logreg':
            self.test_accu = classifier.Evaluator.accuracy(Y, np.dot(X_feats,self.w)+self.b)
        else:
            DS = ClassificationDataSet( X_feats.shape[1], 1, nb_classes=2 )
            #for i in range(X_feats.shape[0]):
            #    DS.addSample( X_feats[i,:], [Y[i]] )
            DS.setField('input', X_feats)
            DS.setField('target', Y[:,np.newaxis])
            DS._convertToOneOfMany()
            predict,targts = self._nn_trainer.testOnClassData(DS, verbose=True,return_targets=True)
            self.test_accu = np.sum(np.array(predict)==np.array(targts))/float(len(targts))
        return self.test_accu
    
    def loss_multiclass_logreg(self, X, Y, special_bias=None):
        X_feats=np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        return loss_multiclass_logreg(Y, X_feats, (self.w,self.b))
    
    def loss_multiclass_nn(self, X, Y, special_bias=None):
        X_feats = np.ascontiguousarray(np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list)))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        return loss_multiclass_nn(X_feats, Y, self._nn)

    def get_predictions_logreg(self, X, special_bias=None):
        X_feats=np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        return get_predictions_logreg(X_feats, (self.w,self.b))
    
    def get_predictions_nn(self, X, special_bias=None):
        X_feats = np.ascontiguousarray(np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list)))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        DS = ClassificationDataSet( X_feats.shape[1], 1, nb_classes=2 )
        #for i in range(X_feats.shape[0]):
        #    DS.addSample( X_feats[i,:], [0.0] )
        DS.setField('input', X_feats)
        DS.setField('target', np.zeros((X_feats.shape[0],1)))
        DS._convertToOneOfMany()
        prob = self._nn.activateOnDataset(DS)
        prob = mpi.COMM.gather(prob)
        if mpi.is_root():
            return np.vstack(prob)
        else:
            return np.zeros((0))
        

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

def loss_multiclass_nn(X_feats, Y, nn):
    DS = ClassificationDataSet( X_feats.shape[1], 1, nb_classes=2 )
    #for i in range(X_feats.shape[0]):
    #    DS.addSample( X_feats[i,:], [0.0] )
    DS.setField('input', X_feats)
    DS.setField('target', np.zeros((X_feats.shape[0],1)))
    DS._convertToOneOfMany()
    prob = nn.activateOnDataset(DS)
    Y2 = classifier.to_one_of_k_coding(Y, 0)
    local_likelihood = -np.dot(np.log(prob).flat, Y2.flat)
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


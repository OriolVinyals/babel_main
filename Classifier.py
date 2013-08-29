from iceberk import classifier, mathutil, mpi
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
from scipy import optimize
import sys

class Classifier:
    def __init__(self,Xtrain,Ytrain):
        self._Xtrain=Xtrain
        self._Ytrain=Ytrain
        self.features=Xtrain.keys()
        
    def Train(self,feat_list=None,type='logreg',gamma=0.0,domeanstd=True,special_bias=None,add_bias=True, weight=None, class_instance=None, method='sigmoid',factor=10.0,arch=[10],
              cv_feats=None, cv_special_bias=None,cv_class_instance=None):
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
        #CV
        if cv_feats!=None:
            cv_feats = np.ascontiguousarray(np.hstack((cv_feats[feat] for feat in feat_list)))
            cv_feats -= self.m
            cv_feats /= self.std
            if special_bias != None:
                cv_feats = np.ascontiguousarray(np.hstack((cv_feats, cv_special_bias)))
        '''Classifier stage'''
        if type=='linsvm':
            self.w, self.b = classifier.svm_onevsall(Xtrain_feats, self._Ytrain, self._gamma, weight = weight, special_bias=special_bias, add_bias=add_bias)
            return (self.w,self.b)
        elif type=='logreg':
            self.w, self.b = l2logreg_onevsall(Xtrain_feats, self._Ytrain, self._gamma, weight = weight, special_bias=special_bias, add_bias=add_bias)
            return (self.w,self.b)
        elif type=='logreg_atwv':
            self.w, self.b = Train_atwv(Xtrain_feats,class_instance=class_instance,weight=weight,special_bias=special_bias, add_bias=add_bias, method=method, 
                                        factor=factor, gamma=self._gamma, cv_class_instance=cv_class_instance, cv_feats=cv_feats)
        elif type=='nn_atwv':
            self._arch = arch
            self._weights_nn = Train_atwv_nn(Xtrain_feats,class_instance=class_instance,weight=weight,special_bias=special_bias, add_bias=add_bias, 
                                             arch=self._arch, method=method, factor=factor, gamma=self._gamma, cv_class_instance=cv_class_instance, cv_feats=cv_feats)
            #self._weights_nn = Train_atwv_nn(Xtrain_feats,class_instance=class_instance,weight=self._weights_nn,special_bias=special_bias, add_bias=add_bias, 
            #                                 arch=self._arch, method=method, factor=factor*10.0)
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
        if self._type=='linsvm' or self._type=='logreg' or self._type=='logreg_atwv':
            self.test_accu = classifier.Evaluator.accuracy(Y, np.dot(X_feats,self.w)+self.b)
        elif self._type=='nn_atwv':
            pred = get_predictions_nn(X_feats, self._weights_nn, arch=[10])[0]
            pred[:,0] = 0.5
            self.test_accu = classifier.Evaluator.accuracy(Y, pred)
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
        return loss_multiclass_nn(X_feats, Y, self._weights_nn, self._arch)

    def get_predictions_logreg(self, X, special_bias=None):
        X_feats=np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        return get_predictions_logreg(X_feats, (self.w,self.b))
    
    def get_predictions_nn_old(self, X, special_bias=None):
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
    
    def get_predictions_nn(self, X, special_bias=None):
        X_feats = np.ascontiguousarray(np.hstack((X[self.feat_list[i]] for i in range(len(self.feat_list)))))
        X_feats -= self.m
        X_feats /= self.std
        if special_bias != None:
            X_feats = np.ascontiguousarray(np.hstack((X_feats, special_bias)))
        prob = get_predictions_nn(X_feats, self._weights_nn, self._arch)[0]
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

def loss_multiclass_nn_old(X_feats, Y, nn):
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

def loss_multiclass_nn(X_feats, Y, nn, arch):
    prob = get_predictions_nn(X_feats, nn, arch)[0]
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
    
def get_predictions_logreg_perclass(X, weights):
    pred = mathutil.dot(X,weights[0])+weights[1]
    prob = 1.0/(1.0+np.exp(-pred))
    prob = mpi.COMM.gather(prob)
    if mpi.is_root():
        return np.vstack(prob)
    else:
        return np.zeros((0))
    
def get_predictions_nn(X, weights,arch):
    hid = mathutil.dot(X,weights[0])+weights[1]
    hid = 1.0/(1+np.exp(-hid)) #sigmoid
    pred = mathutil.dot(hid,weights[2])+weights[3]
    #prob = pred - pred.max(axis=1)[:,np.newaxis]
    #mathutil.exp(prob, out=prob)
    #prob /= prob.sum(axis=1)[:, np.newaxis]
    prob = 1.0/(1.0+np.exp(-pred))
    prob = mpi.COMM.gather(prob)
    hid = mpi.COMM.gather(hid)
    if mpi.is_root():
        return np.vstack(prob),np.vstack(hid)
    else:
        return np.zeros((0)),np.zeros((0))
    
def Train_atwv(Xtrain_feats,class_instance=None,weight=None,special_bias=None,add_bias=True,method='sigmoid',factor=10.0,gamma=0.0,
               cv_feats=None,cv_class_instance=None):
    K=2
    dim=Xtrain_feats.shape[1]
    if weight==None:
        params = []
        w = np.zeros((dim,K))
        b = np.zeros((1,K))
        w[0,0]=-0.09
        w[0,1]=0.09
        w[-1,0]=-1
        w[-1,1]=1
        params.append(w)
        params.append(b)
        weight = np.hstack((p.flatten() for p in params))
    else:
        weight = np.hstack((p.flatten() for p in weight))
    #weight = optimize.fmin(f_atwv,weight,(Xtrain_feats,class_instance,special_bias,add_bias),disp=True,xtol=0.01)
    print 'Error',optimize.check_grad(lambda x: f_atwv(x, Xtrain_feats,class_instance,special_bias,add_bias,method,factor,0,gamma)[0], 
                        lambda x: f_atwv(x, Xtrain_feats,class_instance,special_bias,add_bias,method,factor,0,gamma)[1],
                        weight)
    #weight = sgd(f_atwv,weight,args=(Xtrain_feats,class_instance,special_bias,add_bias,method,factor,10),disp=True)[0]
    if cv_feats != None:
        callback_f = lambda x: sys.stdout.write('CV ATWV ' + repr(-f_atwv(x, cv_feats,cv_class_instance,special_bias,add_bias,'exact',0,0,0)[0]) +
                                                ' Train ATWV ' + repr(-f_atwv(x, Xtrain_feats,class_instance,special_bias,add_bias,'exact',0,0,0)[0]))
    else:
        callback_f = lambda x: sys.stdout.write('Train ATWV ' + repr(-f_atwv(x, Xtrain_feats,class_instance,special_bias,add_bias,'exact',0,0,0)[0]))
    #callback_f = lambda x: sys.stdout.write('Dummy CB')
    weight = optimize.fmin_l_bfgs_b(f_atwv,weight,args=(Xtrain_feats,class_instance,special_bias,add_bias,method,factor,0,gamma),disp=True,callback=callback_f)[0]
    w = weight[: K * dim].reshape(dim, K)
    b = weight[K * dim :]
    return w,b
    
    
def f_atwv(weights, X,class_instance,special_bias,add_bias,method,factor,mb,gamma):
    weights_unfl = []
    K=2
    dim=X.shape[1]
    w = weights[: K * dim].reshape(dim, K)
    b = weights[K * dim :]
    if add_bias==False:
        b[:] = 0
    if special_bias != None:
        w[-1,0] = -1
        w[-1,1] = 1
    weights_unfl.append(w)
    weights_unfl.append(b)
    #scores = get_predictions_logreg(X, weights_unfl)
    if mb>0:
        n_samp = X.shape[0]
        ind_sel = np.random.choice(n_samp, mb, replace=False)
        X = X[ind_sel]
        scores = get_predictions_logreg(X, weights_unfl)
        f,gpred = class_instance.GetATWVsmooth(scores[:,1],method=method,factor=factor,inds=ind_sel)
    else:
        scores = get_predictions_logreg(X, weights_unfl)
        f,gpred = class_instance.GetATWVsmooth(scores[:,1],method=method,factor=factor)
    g_w_0 = np.dot(X.T, gpred*scores[:,1]*(-scores[:,0]))
    #g_w_0 = np.dot(X.T, 0*gpred*scores[:,1]*(-scores[:,0]))
    g_w_1 = np.dot(X.T, gpred*scores[:,1]*(1-scores[:,1]))
    g_w = np.hstack((g_w_0[:,np.newaxis],g_w_1[:,np.newaxis]))
    g_b_0 = np.sum(gpred*scores[:,1]*(-scores[:,0]))
    #g_b_0 = np.sum(0*gpred*scores[:,1]*(-scores[:,0]))
    g_b_1 = np.sum(gpred*scores[:,1]*(1-scores[:,1]))
    g_b = np.hstack((g_b_0,g_b_1))
    if add_bias==False:
        g_b[:]=0
    if special_bias != None:
        g_w[-1,:] = 0
    f = f - gamma*np.sum(w**2)
    g_w = g_w - gamma*2.0*w
    g = np.hstack((g_w.flatten(),g_b.flatten()))
    return -f,-g

def Train_atwv_nn(Xtrain_feats,class_instance=None,weight=None,special_bias=None,add_bias=True,arch=[10],method='sigmoid',factor=10.0,gamma=0.0,
                  cv_feats=None,cv_class_instance=None):
    #1 hidden layer NN
    K=2
    dim=Xtrain_feats.shape[1]
    n_hid = arch[0]
    if weight==None:
        params = []
        w_h = np.random.randn(dim,n_hid)
        interval = 4.0*np.sqrt(6.0/(dim+n_hid))
        #w_h = np.random.uniform(low=-interval,high=interval,size=(dim,n_hid))
        b_h = np.zeros((1,n_hid))
        w_s = np.random.randn(n_hid,K)
        interval = 4.0*np.sqrt(6.0/(n_hid+K))
        #w_s = np.random.uniform(low=-interval,high=interval,size=(n_hid,K))
        b_s = np.zeros((1,K))
        params.append(w_h)
        params.append(b_h)
        params.append(w_s)
        params.append(b_s)
        weight = np.hstack((p.flatten() for p in params))
    else:
        weight = np.hstack((p.flatten() for p in weight))
    print 'Error',optimize.check_grad(lambda x: f_atwv_nn(x, Xtrain_feats,class_instance,special_bias,add_bias,arch,method,factor,gamma)[0], 
                        lambda x: f_atwv_nn(x, Xtrain_feats,class_instance,special_bias,add_bias,arch,method,factor,gamma)[1],
                        weight)
    if cv_feats != None:
        callback_f = lambda x: sys.stdout.write('CV ATWV ' + repr(-f_atwv_nn(x, cv_feats,cv_class_instance,special_bias,add_bias,arch,'exact',0,0)[0]) +
                                                'Train ATWV ' + repr(-f_atwv_nn(x, Xtrain_feats,class_instance,special_bias,add_bias,arch,'exact',0,0)[0]))
    else:
        callback_f = lambda x: sys.stdout.write('Train ATWV ' + repr(-f_atwv_nn(x, Xtrain_feats,class_instance,special_bias,add_bias,arch,'exact',0,0)[0]))
    #callback_f = lambda x: sys.stdout.write('Dummy CB')
    weight = optimize.fmin_l_bfgs_b(f_atwv_nn,weight,args=(Xtrain_feats,class_instance,special_bias,add_bias,arch,method,factor,gamma),disp=True,pgtol=1e-6,
                                    callback=callback_f)[0]
    ind = 0
    w_h = weight[ind: (ind+n_hid * dim)].reshape(dim, n_hid)
    ind += n_hid * dim
    b_h = weight[ind:(ind+n_hid)]
    ind += n_hid
    w_s = weight[ind: (ind + n_hid*K)].reshape(n_hid, K)
    ind += n_hid*K
    b_s = weight[ind:]
    return w_h,b_h,w_s,b_s
    
    
def f_atwv_nn(weights, X,class_instance,special_bias,add_bias,arch,method,factor,gamma):
    weights_unfl = []
    K=2
    dim=X.shape[1]
    n_hid=arch[0]
    ind = 0
    w_h = weights[ind: (ind+n_hid * dim)].reshape(dim, n_hid)
    ind += n_hid * dim
    b_h = weights[ind:(ind+n_hid)]
    ind += n_hid
    w_s = weights[ind: (ind + n_hid*K)].reshape(n_hid, K)
    ind += n_hid*K
    b_s = weights[ind:]
    if add_bias==False:
        b_s[:] = 0
    weights_unfl.append(w_h)
    weights_unfl.append(b_h)
    weights_unfl.append(w_s)
    weights_unfl.append(b_s)
    #Forward
    scores,hidden = get_predictions_nn(X, weights_unfl,arch)
    #Backward, factorize!
    f,gpred = class_instance.GetATWVsmooth(scores[:,1],method=method,factor=factor)
    #g_w_s_0 = np.dot(hidden.T, gpred*scores[:,1]*(-scores[:,0]))
    ##g_w_s_0 = np.dot(hidden.T, 0*gpred*scores[:,1]*(-scores[:,0]))
    g_w_s_1 = np.dot(hidden.T, gpred*scores[:,1]*(1-scores[:,1]))
    g_w_s_0 = np.zeros(g_w_s_1.shape)
    g_w_s = np.hstack((g_w_s_0[:,np.newaxis],g_w_s_1[:,np.newaxis]))
    #g_b_s_0 = np.sum(gpred*scores[:,1]*(-scores[:,0]))
    ##g_b_s_0 = np.sum(0*gpred*scores[:,1]*(-scores[:,0]))
    g_b_s_1 = np.sum(gpred*scores[:,1]*(1-scores[:,1]))
    g_b_s_0 = np.zeros(g_b_s_1.shape)
    g_b_s = np.hstack((g_b_s_0,g_b_s_1))
    
    #dEdh_0 = gpred*scores[:,1]*(-scores[:,0])
    #dEdh_1 = gpred*scores[:,1]*(1-scores[:,1])
    #dEdh = np.hstack((dEdh_0[:,np.newaxis],dEdh_1[:,np.newaxis]))
    dEdh = gpred*scores[:,1]*(1-scores[:,1])
    dEdh = np.outer(dEdh,w_s[:,1])
    g_w_h = np.dot(X.T, dEdh*hidden*(1-hidden))
    g_b_h = np.sum(dEdh*hidden*(1-hidden),axis=0)
    f = f - gamma*np.sum(w_h**2) - gamma*np.sum(w_s**2)
    g_w_h = g_w_h - gamma*2.0*w_h
    g_w_s = g_w_s - gamma*2.0*w_s
    if add_bias==False:
        g_b_s[:]=0
    g = np.hstack((g_w_h.flatten(),g_b_h.flatten(),g_w_s.flatten(),g_b_s.flatten()))
    return -f,-g

def sgd(f,w,args=[],disp=True):
    lr = 0.01
    momentum = 0.1
    v = w.copy()
    v[:]=0.0
    for i in range(500):
        f_val,g = f(w,*args)
        v = momentum*v - lr*g
        w = w + v
        if disp:
            print 'f: ',f_val,'|g|: ',np.sum(g**2)
    return w,f_val



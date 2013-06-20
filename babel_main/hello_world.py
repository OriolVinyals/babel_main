'''
Created on Jun 18, 2013

@author: vinyals
'''
#from iceberk import classifier  
import htkmfc
import numpy as np
import gflags
gflags.DEFINE_string("root", "mujamuja",
                     "The root to the cifar dataset (python format)")
FLAGS = gflags.FLAGS

if __name__ == '__main__':
    np.__config__.show()
    a = np.zeros((100,100))
    b = np.dot(a,a)
    file_htk = htkmfc.HTKFeat_read("./data/BABEL_BP_104_85455_20120310_210107_outLine")
    data = file_htk.getall()
    file_htk = ""
    print data.shape[:]
    print data[0]
    print "Hello world!" + FLAGS.root
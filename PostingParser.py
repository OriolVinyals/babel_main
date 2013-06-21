'''
Created on Jun 18, 2013

@author: vinyals
'''

from external import htkmfc
import numpy as np
import gflags
import csv
gflags.DEFINE_string("root", "mujamuja",
                     "The root to the cifar dataset (python format)")
FLAGS = gflags.FLAGS

class PostingParser:
    def __init__(self, csvfile):
        self.dictReader = csv.DictReader(open(csvfile, 'rb'),delimiter = ',', quotechar = '"')
        self.data = []
        for row in self.dictReader:
            self.data.append(row)
    
    def GetFields(self):
        return self.dictReader.fieldnames
    
    def num_total(self):
        return len(self.data)

if __name__ == '__main__':
    testParser = PostingParser("./data/word.kwlist.alignment.csv")
    print testParser.GetFields()
    print testParser.num_total()

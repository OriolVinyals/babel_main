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
            #print row
            self.data.append(row)
    
    def GetFields(self):
        return self.dictReader.fieldnames
    
    def num_total(self):
        return len(self.data)

if __name__ == '__main__':
    testParser = PostingParser("./data/word.kwlist.alignment.csv")
    print testParser.GetFields()
    print testParser.num_total()
    lol = {}
    for i in range(testParser.num_total()):
        lol[testParser.data[i]['file']]=1
    print len(lol)
    num_words = 0
    words_gt = {}
    for i in range(testParser.num_total()):
        if testParser.data[i]['alignment']=='CORR' or testParser.data[i]['alignment']=='MISS':
            num_words += 1
            words_gt[testParser.data[i]['termid']]=1
    print num_words
    print len(words_gt)
'''
Created on Jun 18, 2013

@author: vinyals
'''

from external import htkmfc
import numpy as np
import gflags
import csv
import PostingParser

FLAGS = gflags.FLAGS

class Sampler:
    def __init__(self,posting_parser):
        self.posting_parser = posting_parser
        
    def SamplePositive(self):
        self.positive_data = []
        self.positive_labels = []
        num_words = 0
        for i in range(self.posting_parser.num_total()):
            if self.posting_parser.data[i]['alignment']=='CORR' or testParser.data[i]['alignment']=='MISS':
                self.positive_data.append(self.posting_parser.data[i])
                self.positive_labels.append(self.posting_parser.data[i]['termid'])
                num_words += 1
                
    def SampleNegative(self):
        self.negative_data = []
        self.negative_labels = []
        num_words = 0
        for i in range(self.posting_parser.num_total()):
            if self.posting_parser.data[i]['alignment']=='CORR!DET' or testParser.data[i]['alignment']=='FA':
                self.negative_data.append(self.posting_parser.data[i])
                self.negative_labels.append(self.posting_parser.data[i]['termid'])
                num_words += 1
                
if __name__ == '__main__':
    testParser = PostingParser.PostingParser("./data/word.kwlist.alignment.csv")
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
    sampler = Sampler(testParser)
    sampler.SamplePositive()
    sampler.SampleNegative()
    print len(sampler.positive_data)
    print len(sampler.negative_data)

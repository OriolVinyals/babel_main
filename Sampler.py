'''
Created on Jun 18, 2013

@author: vinyals
'''

import random
import PostingParser

class Sampler:
    def __init__(self,posting_parser):
        self.posting_parser = posting_parser
        
    def GetPositive(self):
        self.positive_data = []
        self.positive_labels = []
        self.num_positive = 0
        for i in range(self.posting_parser.num_total()):
            if self.posting_parser.data[i]['alignment']=='CORR' or testParser.data[i]['alignment']=='MISS':
                self.positive_data.append(self.posting_parser.data[i])
                self.positive_labels.append(self.posting_parser.data[i]['termid'])
                self.num_positive += 1
                
    def GetNegative(self):
        self.negative_data = []
        self.negative_labels = []
        self.num_negative = 0
        for i in range(self.posting_parser.num_total()):
            if self.posting_parser.data[i]['alignment']=='CORR!DET' or testParser.data[i]['alignment']=='FA':
                self.negative_data.append(self.posting_parser.data[i])
                self.negative_labels.append(self.posting_parser.data[i]['termid'])
                self.num_negative += 1
                
    def SampleData(self,percpos):
        if self.num_positive/(self.num_positive + self.num_negative) >= percpos:
            print 'We already have lots of positive examples'
            return
        else:
            '''Need to find how many negative examples to keep and just random subsample for now'''
            target_negative = int(self.num_positive/percpos - self.num_positive)
            inds = random.sample(range(self.num_negative),target_negative)
            self.negative_data = [self.negative_data[i] for i in inds]
            self.negative_labels = [self.negative_labels[i] for i in inds]
            self.num_negative = len(self.negative_data)
                
if __name__ == '__main__':
    testParser = PostingParser.PostingParser("./data/word.kwlist.alignment.csv")
    sampler = Sampler(testParser)
    sampler.GetPositive()
    sampler.GetNegative()
    print len(sampler.positive_data)
    print len(sampler.negative_data)
    sampler.SampleData(0.2)
    print len(sampler.negative_data)


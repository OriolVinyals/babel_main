'''
Created on Jun 18, 2013

@author: vinyals
'''

import csv

class PostingParser:
    def __init__(self, csvfile):
        self.dictReader = csv.DictReader(open(csvfile, 'rb'),delimiter = ',', quotechar = '"')
        self.data = []
        num_rows = 0
        for row in self.dictReader:
            self.data.append(row)
            num_rows += 1
            if __debug__:
                if(num_rows>1000):
                    break
    
    def GetFields(self):
        return self.dictReader.fieldnames
    
    def num_total(self):
        return len(self.data)

if __name__ == '__main__':
    testParser = PostingParser('./data/word.kwlist.alignment.csv')
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
from math import ceil
import numpy as np
import pdb
from Tools import readFile2

class Histogram():
  def __init__(self, bin_width, min_value, max_value, center):
    self.size = int(ceil((max_value-min_value)/bin_width))
    self.hist = np.zeros(self.size)
    self.center = center
    self.overflow = 0
    self.underflow = 0
    self.n_entries = 0
    self.min = min_value
    self.max = max_value
    #print '__init__: binwidth: %d' % bin_width
    self.bin_width = float(bin_width)
    #print '__init__: self.bin_width: %d' % self.bin_width

  def printHistogramVariables(self):
    print '#########################'
    print 'Size: %d' % self.size
    print 'Overflow: %d' % self.overflow
    print 'Underflow: %d' % self.underflow
    print 'Number of entries: %d' % self.n_entries
    print 'Minimum value: %s' % self.min
    print 'Maximum value: %s' % self.max
    print 'Bin width: %s' % self.bin_width
    print '#########################'


  def findBin(self, value):
    return float((value-self.min)/self.bin_width)

  def printHistogram(self):
    print 'Underflow: %d' % self.underflow
    i = 0
    for entry in self.hist:
        print '%s \t %s' % (entry, self.getValue(i))
        i += 1
    print 'Overflow: %d' % self.overflow

  def addValue(self, value):
    bin = int((value-self.min)/self.bin_width)
    if bin>=0 and bin < self.size:
        self.hist[bin] += 1
    elif bin < 0:
        self.underflow += 1
    else:
        self.overflow += 1

  def getValue(self, bin):
    #print 'bin: %d \t self.bin_width: %d \t self.min: %d' % (bin, self.bin_width, self.min)
    return float(bin*self.bin_width)+ self.bin_width/2. + self.min

if __name__ == '__main__':
  test_r = [1,3,2,6,7,8,2,4]
  test_c = [(1,1), (3,3), (2,2), (6,6), (7,7), (8,8), (2,2), (4,4)]
  data = zip(test_r, test_c)
  
  H,edges,data = PHistogram(data, 8, [0,4], 0)
  print H, edges, data

from math import ceil
import numpy as np
import pdb


def PHistogram(data, number_of_bins, hrange=None, position=None):
  """ Creates a histogram for a given with number_of_bins entries within the boundaries given by range. 
      The important thing is that in addition to the histogram we also create an array that stores
      the single data objects that were filled in the histogram in the same bin they have been place
      in the histogram. So if we have data that corresponds in some way and we don't want to lose the
      correlation we can still find out, which data objects are placed in a bin and do further analysis.
      For example consider data of the form 
      data[0] = list of different radiuses
      data[1] = list of (x,y) tuples
      So we create a histogram for data[0]
      To choose from which data set we want to form a histogram we specify position. So position=0
      creates a histogram for the list of radiuses.
  """
  dtype = [('radius', float), ('center', tuple)]

  sa = np.array(data, dtype=dtype)
  if (hrange is not None):
    mn, mx = hrange
    if mn > mx:
      raise AttributeError('max must be larger than min')
  
  data_array = [list() for _ in range(number_of_bins)]
  bins = np.linspace(mn, mx, number_of_bins+1)
  n = np.zeros(number_of_bins, int)
  block = 65536

  for i in np.arange(0, len(sa), block):
    # sort the array for radius
    sa = np.sort(sa[i:i+block], order='radius')

    # split radius and center data
    a,b = zip(*sa)

    # convert radius object to array and center object to list
    a = np.asarray(a)
    b = list(b)

    # intermediate calculation of the histogram VW used to add center entries to their
    # respective bins
    VW = np.r_[a.searchsorted(bins[:-1], 'left'), a.searchsorted(bins[-1], 'right')]
    VW = np.diff(VW)
    n += VW
    j = 0
    for i in VW:
      if i > 0:
        index = i
      else:
        j+=1
        continue
      data_array[j].append(b[:index])
      j+=1
      b = b[index:]

  return n, bins, data_array

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

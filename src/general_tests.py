from Tools import readFile
from CircleDetection3Points import *
import numpy as np
from Queue import Queue
import threading
import pdb
import itertools
import matplotlib.pyplot as plt
from scipy import misc as sp



def tstCombListST(  ):
    data = readFile( '../data/lhcb_data/Event00009999.txt' )
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r = calculateCircleFromPoints( combinationsList )
    print len(r)

def tstBackgroundHistogram(  ):
  data = readFile( '../data/0_circles_200_bg.txt')
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r1 = calculateCircleFromPoints( combinationsList, onlyRadius=True )


  data = readFile( '../data/0_circles_130_bg.txt')
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r2 = calculateCircleFromPoints( combinationsList, onlyRadius=True )

  data = readFile( '../data/0_circles_150_bg.txt')
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r3 = calculateCircleFromPoints( combinationsList, onlyRadius=True )

  data = readFile( '../data/0_circles_185_bg.txt')
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r4 = calculateCircleFromPoints( combinationsList, onlyRadius=True )

  plt.hist(r1, 200, alpha=0.5, label='200')
  plt.hist(r2, 200, alpha=0.5, label='130')
  plt.hist(r3, 200, alpha=0.5, label='150')
  plt.hist(r4, 200, alpha=0.5, label='185')
  plt.legend(loc='upper right')
  plt.show()

def tstScaling():
  factor = 3.39
  data = readFile( '../data/0_circles_200_bg.txt')
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r2 = calculateCircleFromPoints( combinationsList, onlyRadius=True )
  h2, edges = np.histogram( r2, 100, [0,1])
  #np.savetxt('r2.txt', h2)
  h2 = factor*h2

  data = readFile( '../data/0_circles_300_bg.txt')
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r1 = calculateCircleFromPoints( combinationsList, onlyRadius=True )
  h1, edges = np.histogram( r1, 100, [0,1])

  # plt.hist(r1, 200, alpha=0.5, label='200')
  # plt.hist(r2, 200, alpha=0.5, label='130')
  # plt.legend(loc='upper right')
  # plt.show()
  step = 1./100
  edges = np.arange(0,1,step)
  plt.bar(edges,h1, width=step, alpha = 0.5, label='300')
  plt.bar(edges,h2, width=step, alpha = 0.5, label='200', color='y')
  plt.legend(loc='upper right')
  plt.xlim(min(edges), max(edges))
  
  plt.show()

def createBackgroundHistogram():
  bins = 1000
  data = readFile( '../data/0_circles_600_bg.txt' )
  combinationsList = list( itertools.combinations( data['allPoints'],3 ) )
  xy, r = calculateCircleFromPoints( combinationsList, onlyRadius=True )
  h, edges = np.histogram( r, bins, [0,1] )
  np.savetxt('600_bg_r.txt',h)
  step = 1./bins
  edges = np.arange(0,1,step)
  plt.bar(edges, h, width=step)
  plt.show()

def diffBackground():
  background_hits = range(51)
  for i in background_hits:
    path = '../data/1_ring_diff_bg/1_circle_%s_bg.txt' % i
    data = readFile( path )
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    print "%s bg hits" % i
    res = main( combinationsList )

def diffCircleHits( ):
  circle_hits = range(5,10)
  circle_hits += range(10,30,5)
  print circle_hits

  for i in circle_hits:
    path = '../data/1_ring_diff_hits_const_bg/1_circle_%s_hits_20_bg.txt' % i
    data = readFile( path )
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    print "%s circle hits" % i
    res = main( combinationsList )


def rHistogram():
  r = np.loadtxt('600_bg_r.txt')
  print len(r)
  step = 1./len(r)
  edges = np.arange(0,1, step)
  plt.bar(edges, r, width=step)
  plt.show()

def fullRun():
  data = '../data/lhcb_data/Event00002346.txt'
  res = setUp(data)
  print res['Value']

def binomialCoefficient(n, k):
  return sp.comb(n,k)

if __name__ == '__main__':
  createBackgroundHistogram()
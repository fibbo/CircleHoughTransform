'''
Created on Apr 29, 2015

@author: phi
'''

import sys
import itertools
import pprint
import pdb
import cPickle as pickle
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import misc as sp
from visualizeData import plotData, convertTuplesToList
from artificial_circle import generateCircle

from Tools import readFile, S_ERROR, S_OK, combinations
from timer import Timer



NUMBER_OF_R_BINS = 1000 #bins for radius
NUMBER_OF_S_BINS = 1000 #bins for space
MAX_POINT_DISTANCE = 0.30
MAX_CENTER_DISTANCE = 0.025
MAX_RADIUS_DISTANCE = 0.010
RADIUS_THRESHOLD = 120
CENTER_THRESHOLD = 120
VISUALISATION = True
EVENTNUMBER = -1

def SHistogram(data, number_of_bins, hrange=None):
  """ Creates a histogram for a given with number_of_bins entries within the boundaries given by range. 
      The important thing is that in addition to the histogram we also create an array that stores
      the single data objects that were filled in the histogram in the same bin they have been place
      in the histogram. So if we have data that corresponds in some way and we don't want to lose the
      correlation we can still find out, which data objects are placed in a bin and do further analysis.

      To choose from which data set we want to form a histogram we specify position. So position=0
      creates a histogram for the list of radiuses.

      :param: list data: list of 2tuples where the first part of the tuple is of 
      dimension one. The 2nd part of the tuple can be anything.
      :returns: np.array n - Histogram of 
  """
  
  if (hrange is not None):
    mn, mx = hrange
    if mn > mx:
      raise AttributeError('max must be larger than min')
  else:
    mn, mx = 0,1
  
  data_array = [None for _ in range(number_of_bins)]
  bins = np.linspace(mn, mx, number_of_bins+1)
  n = np.zeros(number_of_bins, int)
  block = 65536

  for i in np.arange(0, len(data), block):
    # sort the array for radius
    sa = np.sort(data[i:i+block], order='radius')

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
    for counter,i in enumerate(VW):
      if i > 0:
        if data_array[counter]:
          data_array[counter] += b[:i]
        else:
          data_array[counter] = b[:i]
        b = b[i:]
      else:
        continue


  return S_OK( (n, bins, data_array) )


def calculateCircleFromPoints(combinationsList, onlyRadius=False):
  """ Calculate the circle from 3 given points. Discard points that are further than 0.3 apart. Discard points that create a
  Circle with a radius larger than 0.15.

  For radius background histograms we have the parameter onlyRadius when we are only interested in the radiuses.

  """
  r = []
  xy = []

  for points in combinationsList:
    A = points[0]
    B = points[1]
    C = points[2]
    a = np.linalg.norm( C - B )
    b = np.linalg.norm( C - A )
    c = np.linalg.norm( B - A )
    #maximum distance between 2 points on a circle is 2*r, we know r shouldn't be bigger than 0.15 so if 2
    #points are further apart this triple isn't interesting for us
    
    if a > MAX_POINT_DISTANCE or b > MAX_POINT_DISTANCE or c > MAX_POINT_DISTANCE:
      continue
    s = ( a + b + c ) / 2

    R = a * b * c / 4 / np.sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) )
    
    #maximum radius is 0.15 everything larger we forget
    if R<MAX_POINT_DISTANCE/2:
      if onlyRadius:
        r.append(R)
      else: 
        a2 = a*a
        b2 = b*b
        c2 = c*c
        lambda1 = a2 * ( b2 + c2 - a2 )
        lambda2 = b2 * ( a2 + c2 - b2 )
        lambda3 = c2 * ( a2 + b2 - c2 )
        P = np.column_stack( ( A, B, C ) ).dot( np.hstack( ( lambda1, lambda2, lambda3 ) ) )
        P /= lambda1 + lambda2 + lambda3
        xy.append( (P[0],P[1]) )
        r.append(R)

  return xy, r

def fullCenterHistogram( xy, bins=NUMBER_OF_S_BINS ):
  x,y = zip(*xy)
  H, xedges, yedges, _ = plt.hist2d(x,y,bins, [[-0.5,0.5],[-0.5,0.5]])
  plt.colorbar()
  plt.show()

def main( combinationsList, n ):
  """ Main logic of the algorithm. Here we put all the results from the
  other methods together and finally return a list of possible circles.

  With the help of barycentric coordinates we calculate the radius and the center defined by each tuple of 3 points given as parameter

  @param: combinationsList: a list of all possible combinations of 3 points
  @param int n: number of triples in combinationsList
  @returns: S_OK( resList ): resList is a list of dictionaries of possible circles.

  """

  xy, r = calculateCircleFromPoints( combinationsList )
  #fullCenterHistogram( xy )
  data = zip(r, xy)
  dtype = [('radius', float), ('center', tuple)]
  data = np.array(data,dtype=dtype)
  res = SHistogram(data, NUMBER_OF_R_BINS, [0,1])
  if not res['OK']:
    return S_ERROR(res['Message'])
  
  radius_histogram, edges,  center_data = res['Value']

  # create a background histogram
  factor = sp.comb(600,3)/sp.comb(n,3)
  bkgHistogram = np.loadtxt('600_bg_r.txt')
  bkgHistogram /= factor

  radius = {}
  radius['H'] = radius_histogram - bkgHistogram
  radius['xedges'] = edges
  radius['center_data'] = center_data

  #visualizeRadiusHistogram(radius)

  radiuses, center_data = extractRadius(radius)

  res = []

  # check for each radius if we have a peak in center_data
  for radius, center in zip(radiuses, center_data):
    x,y = zip(*center)
    H, xedges, yedges = np.histogram2d(x,y,NUMBER_OF_S_BINS,[[-0.5, 0.5], [-0.5, 0.5]])
    center_dict = {}
    center_dict['H'] = H
    center_dict['xedges'] = xedges
    center_dict['yedges'] = yedges
    #visualizeCenterHistogram(x,y)
    circle_center = extractCenter(center_dict)
    if len(circle_center) > 0:
      for circle_data in circle_center:
        res.append( { 'radius' : radius, 'center' : circle_data['center'], 'nEntries' : circle_data['nEntries'] } )
  return S_OK( res )

def extractRadius( radius_dict ):
  """ Simple method to find possible radiuses. Find highest entry in histogram, save the value and set bin to 0 
      and then look for the next.


      :param dict radius: radius dicitonary with 'H' histogram, 'xedges' and 'yedges'
      :returns radius, center_data
  """
  radiuses = []
  center_data = []
  H = radius_dict['H']
  edges = radius_dict['xedges']
  center = radius_dict['center_data']
  while True:
    i = np.argmax(H)
    n = NUMBER_OF_R_BINS
    n_entries = sum(H[i-1 if i>0 else i:i+2 if i<n-1 else i+1])
    if n_entries < RADIUS_THRESHOLD:
      # there are less than 200 entries in 3 bins
      break

    radiuses.append(edges[i])
    index_list = range(i-1 if i>0 else i,i+2 if i<n-1 else i+1)
    center_list = []
    for index in index_list:
      if center[index]:
        center_list += center[index]
      H[index] = 0
    center_data.append(center_list)
         
  return radiuses, center_data 

def extractCenter( center_dict ):
  """ Simple method to find possible circle centers. Find highest entry in histogram save the value and set bin to 0 so we can look for the next.
      we do this <x> times.

      :param dict. center: center dictionary obtained by findCircles method.
      :returns list centers: a list with a tuple of x,y coordinates of possible centers

  """

  H = center_dict['H']
  
  xedges = center_dict['xedges']
  yedges = center_dict['yedges']
  centers = []

  n = NUMBER_OF_S_BINS

  #TODO: to be able to set a mask to set values around the maximum index to 0
  # for example
  #                          111
  #                          1x1
  #                          111
  #
  # so x is the maximum we found and we want to set adjacent values to 0 as well
  while True:
    index = np.argmax(H)
    i, j = np.unravel_index( index, (NUMBER_OF_S_BINS, NUMBER_OF_S_BINS) )
    #n_entries =   sum(sum(H[i-1 if i>0 else i:i+2 if i<n else i+1,j-1 if j>0 else j:j+2 if j<n else j+1]))
    

    # this variation only takes the adjacent values of i,j
    #
    #              1
    #             1x1 
    #              1
    n_entries =   sum(H[i-1 if i>0 else i:i+2 if i<n else i+1,j]) + sum(H[i, j-1 if j>0 else j:j+2 if j+2 <= 3 else j+1]) - H[i,j]
    if n_entries < CENTER_THRESHOLD:
      break

    # Set the entries to 0 so they won't contribute twice
    i_index = range(i-1 if i>0 else i,i+2 if i<n else i+1)
    j_index = range(j-1 if j>0 else j,j+2 if j<n else j+1)
    for ii in i_index:
      H[ii][j] = 0  

    for jj in j_index:
      H[i][jj] = 0
    # for i in i_index:
    #   for j in j_index:
    #     H[i][j] = 0
    
    centers.append( {'center' : (xedges[i], yedges[j]), 'nEntries' : n_entries } )

  return centers

def backgroundHistogram( filename ):
  """ Creates a histogram for both radius and center for a given filename. It is used for creating background histograms so we can minimize
      false hits from background and mismatched 3tuples (2 points of one circle and one of another e.g.)

      :param filename: path to the source text file with the background points
      :returns center dict and radius dict.
  """
  data = readFile( filename )
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r = calculateCircleFromPoints( combinationsList )
  bkgHistogram, edges = np.histogram(r, NUMBER_OF_R_BINS, [0,2])
  return bkgHistogram, edges
  
def removeFakes( results ):
  """ 

  """
  res = []
  sorted_results = sorted( results, key=lambda k: k['nEntries'], reverse=True)
  while len(sorted_results):
    circle = sorted_results.pop()
    unique = True

    for dic in sorted_results:
      if (np.linalg.norm(np.array(circle['center']) - np.array(dic['center']))) < MAX_CENTER_DISTANCE and\
         (abs(circle['radius'] - dic['radius']) < MAX_RADIUS_DISTANCE):
        unique = False
    if unique:
      res.append(circle)

  return res

def compareRings( db_rings, results):
  """ Compare rings find by the algorithm with the known results from the pickle database. If any circle from the algorithm is within a certain
  range (radius and center respectively) of a database circle we consider this database circle as found by the algorithm.

  :param list db_rings: list of dictionaries that have the information of the simulated data
  :param list results: list of dictionaries that have the circles found by the algorithm
  :return list found_circles: list of dictionaries of circles found in the algorithm that have a match in the database

  """
  found_circles = []
  missed_circles = []
  fake_circles = []
  # if 
  while len(results):
    result_ring = results.pop()
    if any(abs(np.linalg.norm(np.array(result_ring['center']) - np.array(db_ring['center']))) < MAX_CENTER_DISTANCE and\
           abs(result_ring['radius']-db_ring['radius']) < MAX_RADIUS_DISTANCE for db_ring in db_rings):
      found_circles.append(result_ring)
    else:
      fake_circles.append(result_ring)

  return found_circles, fake_circles


def visualizeCenterHistogram( x,y, bins=NUMBER_OF_S_BINS ):
  if not VISUALISATION:
    pass
  else:
    H, xedges, yedges, _ = plt.hist2d(x,y,bins, [[-0.5,0.5],[-0.5,0.5]])
    plt.colorbar()
    plt.show()
  
  
def visualizeRadiusHistogram( radius ):
  if not VISUALISATION:
    pass
  else:
    step = 1./NUMBER_OF_R_BINS
    edges = np.arange(0,1,step)
    H = radius['H']
    plt.bar(edges,H, width=step)
    plt.xlim(0 , 0.15)
    
    plt.show()


def setUp( path ):
  data = readFile( path )
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  res = main( combinationsList, n=len(data['allPoints']) )
  return res

def openDB():
  """ Open the pickel database. Has all the information of the simulated data

  """
  f = open('db.pkl')
  db = pickle.load(f)
  return db

if __name__ == '__main__': 
  #### read data #####
  if len( sys.argv ) == 2:
    path = sys.argv[1]
    data = readFile( path )  
    EVENTNUMBER = sys.argv[1][-8:-4]
    fileName = sys.argv[1][-12:-4]+".png"
  else:
    data = {}
    res = generateCircle(0.12, (0,0), background=100)
    data['allPoints'] = res
    fileName = 'tttt.png'
  pp = pprint.PrettyPrinter(depth=6)
  x,y = convertTuplesToList(data['allPoints'])
  
  # Pickle dictionary where we store the results 
  pickle_data = {}

  pickle_data['Parameters'] = { "MAX_RADIUS_DISTANCE" : MAX_RADIUS_DISTANCE, "MAX_CENTER_DISTANCE" : MAX_CENTER_DISTANCE,\
                                "MAX_POINT_DISTANCE" : MAX_POINT_DISTANCE, "RADIUS_THRESHOLD" : RADIUS_THRESHOLD,\
                                "CENTER_THRESHOLD" : CENTER_THRESHOLD, "NUMBER_OF_S_BINS" : NUMBER_OF_S_BINS,\
                                "NUMBER_OF_R_BINS" : NUMBER_OF_R_BINS }

  totalTime = 0
  with Timer() as t:
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  print "Time for creating triple list: %ss" % t.secs
  pickle_data['combTime'] = t.secs

  totalTime += t.secs
  
  with Timer() as t:
    res = main( combinationsList, n=len(data['allPoints']) )
  print "Time for main algorithm: %ss" % t.secs
  pickle_data['mainTime'] = t.secs
  
  totalTime += t.secs
  pickle_data['Runtime'] = totalTime
  
  pickle_data['nPoints'] = len(data['allPoints'])

  if res['OK']:
    res = res['Value']

    pickle_data['allRings'] = res

    # removing double entries
    circles = removeFakes(res)

    # now we compare the algorithm results with the real data
    db = openDB()
    entry = db[int(EVENTNUMBER)]['rings']
    found_circles, fake_circles = compareRings(entry, circles)

    pickle_data['foundRings'] = found_circles
    pickle_data['fakeRings'] = fake_circles
    pickle.dump( pickle_data, open(EVENTNUMBER+".pkl", 'wb'))
    print "Algorithm found %s/%s circles" % (len(found_circles),len(entry))
    print "Algorithm found %s fake circles" % len(fake_circles)
    plotData(x,y,found_circles,savePath=EVENTNUMBER+".png")
    if len(fake_circles):
      plotData(x,y,fake_circles,savePath=EVENTNUMBER+"_fakes.png")


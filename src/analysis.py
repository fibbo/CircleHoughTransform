from __future__ import print_function
import pdb
import os
import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from visualizeData import *
from Tools import *
import copy
from parameters import *

basePath = "/home/phi/workspace/CircleHT/analysis/Threshold/split/run01/"
directories = sorted(os.listdir(basePath))
db = pickle.load(open("/home/phi/workspace/CircleHT/src/db.pkl", 'rb'))
filefolder = "/home/phi/workspace/CircleHT/data/lhcb_data/"


def getPKLs():
  pkls = []
  for directory in directories:
    path = os.path.join(basePath, directory)
    files = os.listdir(path)
    for afile in [f for f in files if f.endswith('.pkl')]:
      if afile=='db.pkl':
        try:
          os.remove(os.path.join(path.afile))
          continue
        except Exception, e:
          print(e)
      pkls.append( (afile[:-4],pickle.load( open(os.path.join(path, afile), 'rb')) ) )
  return pkls

def missedRings(rings, db_rings):
# Checks if a ring from the db has a matching circle found with the algorithm
# if there is no match add the db circle to the missed ring list.
# @param list rings: list of rings found by the algorithm
# @param list db_rings: list of true rings
# @returns list missedRings: rings from the db that weren't found by the algorithm
  missedRings = []
  while (len(db_rings)):
    dbring = db_rings.pop()
    if not any(np.linalg.norm(np.array(dbring['center'])-np.array(ring['center'])) < MAX_CENTER_DISTANCE and abs(dbring['radius'] - ring['radius']) < MAX_RADIUS_DISTANCE
                                                                                      for ring in rings['foundRings']):
      missedRings.append(dbring)
  return missedRings

def runtimeVsPoints():
  runtime = []
  nPoints = []
  pkls = getPKLs()
  for pkl in pkls:
    res = pkl[1]
    runtime.append(res['Runtime'])
    nPoints.append(res['nPoints'])
  plt.scatter(nPoints, runtime)
  plt.savefig('runtime_vs_points_run05.png')

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

  results = sorted(results, key=lambda k: k['nEntries'], reverse=True)
  # if 
  while len(results):
    result_ring = results.pop()
    if any(abs(np.linalg.norm(np.array(result_ring['center']) - np.array(db_ring['center']))) < REAL_MAX_CENTER_DISTANCE and\
           abs(result_ring['radius']-db_ring['radius']) < REAL_MAX_RADIUS_DISTANCE for db_ring in db_rings):
      found_circles.append(result_ring)
    else:
      fake_circles.append(result_ring)

  while len(db_rings):
    db_ring = db_rings.pop()
    if not any(abs(np.linalg.norm(np.array(db_ring['center']) - np.array(res['center']))) < REAL_MAX_CENTER_DISTANCE and
           abs(db_ring['radius'] - res['radius'] < REAL_MAX_RADIUS_DISTANCE) for res in found_circles):
      missed_circles.append(db_ring)

  return found_circles, fake_circles, missed_circles


def removeDuplicates( results ):
  """ 

  """
  res = []
  sorted_results = sorted( results, key=lambda k: k['nEntries'], reverse=True)
  while len(sorted_results):
    circle = sorted_results.pop()
    unique = True

    for dic in sorted_results:
      if (np.linalg.norm(np.array(circle['center']) - np.array(dic['center']))) < DUPLICATE_MAX_CENTER_DISTANCE and\
         (abs(circle['radius'] - dic['radius']) < DUPLICATE_MAX_RADIUS_DISTANCE):
        unique = False
        break
    if unique:
      res.append(circle)

  return res


def efficiency():
  pkls = getPKLs()
  found_circles = 0
  missed_circles = 0
  fake_circles = 0
  tot_circles = 0
  counter = 0
  tooManyCircles = []
  for pkl in pkls:
    allRings = pkl[1]['allRings']
    result = removeDuplicates(allRings)
    # pdb.set_trace()
    db_entry = db[int(pkl[0])]['rings']
    rings = copy.deepcopy(db_entry)
    tot_circles += len(db_entry)
    found, fake, missed = compareRings(db_entry, result)
    found_circles += len(found)
    fake_circles += len(fake)
    missed_circles += len(missed)


    if len(found) > len(rings):
      # print(len(found), len(rings))
      tooManyCircles.append(pkl[0])
      

    # counter += 1
    # if not counter%100:
    #   progress = 100*float(counter)/10000
    #   print('Progress: %.2f %%' % progress, end='\r')
    if MAKEPLOTS:
      data = readFile(filefolder+'Event0000'+pkl[0]+'.txt')
      x,y = zip(*data['allPoints'])
      destPathSim = basePath+'img/'+pkl[0]+'withDuplicates.pdf'
      # destPathReal = basePath+'img/'+pkl[0]+'_real.pdf'
      plotData(x,y,found,savePath=destPathSim)
      # plotData(x,y,rings,savePath=destPathReal)

  print("ratio of found circles over total circles: %s" % (found_circles/float(tot_circles)))
  print("wrongly found circles: %s" % fake_circles)
  print("missed circles: %s" % missed_circles)
  print("Events were too many circles were found %s" % len(tooManyCircles))


    


if __name__=='__main__':
  #res = resultLoop(missDuplicates)
  #print res
  if len(sys.argv)==2:
    MAKEPLOTS = sys.argv[1]
  else:
    MAKEPLOTS = False
  efficiency()

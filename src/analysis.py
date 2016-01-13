import os
import pdb
import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from parameters import *
from visualizeData import *

basePath = "/disk/data3/lhcb/phi/circleHT/Threshold/split/run01/"
directories = sorted(os.listdir(basePath))
db = pickle.load(open("/home/hep/phi/CircleHoughTransform/src/db.pkl", 'rb'))

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
          print e
      pkls.append( (int(afile[:-4]),pickle.load( open(os.path.join(path, afile), 'rb')) ) )
  return pkls

def bulk():
  numberOfFakeRings = 0
  numberOfFoundRings = 0
  numberOfTotalRings = 0
  numberOfMissedRings= 0
  numberOfDuplicates = 0
  duplicateRings = []
  ifile = open('fakeEvents.txt','wb')
  pkls = getPKLs()
  for pkl in pkls:
    res = pkl[1]
    eventNumber = int(pkl[0])
    db_rings = db[eventNumber]['rings']
  # if afile == '0001.pkl':
  #   print res
  #   print db[int(afile[:-4])]
  # 
    if len(res['fakeRings']):
     ifile.write(str(eventNumber)+"\n")
    numberOfFakeRings += len(res['fakeRings'])
    numberOfFoundRings += len(res['foundRings'])
    numberOfTotalRings += len(db_rings)
    while (len(db_rings)):
      dbring = db_rings.pop()
      if not any(np.linalg.norm(np.array(dbring['center'])-np.array(ring['center'])) < 0.25 and abs(dbring['radius'] - ring['radius']) < 0.10\
       for ring in res['foundRings']):
        duplicateRings.append(dbring)
  print "number of missed rings: %s" % numberOfMissedRings
  print "number of fake rings: %s" % numberOfFakeRings
  print "number of found rings: %s" % numberOfFoundRings
  print "number of total rings: %s" % numberOfTotalRings
  print "number of rings missed: %s" % (numberOfTotalRings-numberOfFoundRings)
  print "fake efficiency: %s" % (float(numberOfFakeRings)/numberOfTotalRings)
  print "total efficiency: %s" % (float(numberOfFoundRings)/numberOfTotalRings)
  print "duplicates: %s" % numberOfDuplicates

def resultLoop(*args):
  """ Find duplicate rings that were missed because the cut wasn't loose enough
  """
  allResults = []
  for directory in directories:
    path = os.path.join(basePath, directory)
    files = os.listdir(path)
    for afile in [f for f in files if f.endswith('.pkl')]:
      res = pickle.load( open(os.path.join(path,afile), 'rb') )
      eventNumber = int(afile[:-4])
      db_rings = db[eventNumber]['rings']
      for function in args:
        res = function(res, db_rings)
        allResults += res
  return allResults


def missedRings(rings, db_rings):
# Checks if a ring from the db has a matching circle found with the algorithm
# if there is no match add the db circle to the missed ring list.
# @param list rings: list of rings found by the algorithm
# @param list db_rings: list of true rings
# @returns list missedRings: rings from the db that weren't found by the algorithm
  missedRings = []
  while (len(db_rings)):
    dbring = db_rings.pop()
    if not any(np.linalg.norm(np.array(dbring['center'])-np.array(ring['center'])) < 0.010 and abs(dbring['radius'] - ring['radius']) < 0.005
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
  # if 
  while len(results):
    result_ring = results.pop()
    if any(abs(np.linalg.norm(np.array(result_ring['center']) - np.array(db_ring['center']))) < MAX_CENTER_DISTANCE and\
           abs(result_ring['radius']-db_ring['radius']) < MAX_RADIUS_DISTANCE for db_ring in db_rings):
      found_circles.append(result_ring)
    else:
      fake_circles.append(result_ring)

  while len(db_rings):
    db_ring = db_rings.pop()
    if not any(abs(np.linalg.norm(np.array(db_ring['center']) - np.array(res['center']))) < MAX_CENTER_DISTANCE and
           abs(db_ring['radius'] - res['radius'] < MAX_RADIUS_DISTANCE) for res in found_circles):
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
      if (np.linalg.norm(np.array(circle['center']) - np.array(dic['center']))) < MAX_CENTER_DISTANCE and\
         (abs(circle['radius'] - dic['radius']) < MAX_RADIUS_DISTANCE):
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
  for pkl in pkls:
    allRings = pkl[1]['allRings']
    noDuplicates = removeDuplicates(allRings)
    # pdb.set_trace()
    db_entry = db[int(pkl[0])]['rings']
    tot_circles += len(db_entry)
    print pkl
    found, fake, missed = compareRings(db_entry, noDuplicates)
    found_circles += len(found)
    fake_circles += len(fake)
    missed_circles += len(missed)


  print "ratio of found circles over total circles: %s" % (found_circles/float(tot_circles))
  print "wrongly found circles: %s" % fake_circles
  print "missed circles: %s" % missed_circles

    


if __name__=='__main__':
  #res = resultLoop(missDuplicates)
  #print res
  efficiency()

import os
import pdb
import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

basePath = "/disk/data3/lhcb/phi/circleHT/splitData/run05/"
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
      pkls.append( (afile[:-4],pickle.load( open(os.path.join(path, afile), 'rb')) ) )
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


if __name__=='__main__':
  #res = resultLoop(missDuplicates)
  #print res
  runtimeVsPoints()

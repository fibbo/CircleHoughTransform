import os
import pdb
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

basePath = "/home/phi/workspace/CircleHT/analysis/run02"
directories = sorted(os.listdir(basePath))
db = pickle.load(open("/home/phi/workspace/CircleHT/analysis/db.pkl", 'rb'))

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


def missDuplicates(rings, db_rings):
  duplicateRings = []
  while (len(db_rings)):
    dbring = db_rings.pop()
    if not any(np.linalg.norm(np.array(dbring['center'])-np.array(ring['center'])) < 0.20 and abs(dbring['radius'] - ring['radius']) < 0.10 for ring in rings['foundRings']):
      duplicateRings.append(dbring)
  return duplicateRings

def runtimeVsPoints():
  runtime = []
  nPoints = []
  pkls = getPKLs()
  for pkl in pkls:
    res = pkl[1]
    runtime.append(res['Runtime'])
    nPoints.append(res['nPoints'])
  plt.scatter(nPoints, runtime)
  plt.show()


if __name__=='__main__':
  #res = resultLoop(missDuplicates)
  #print res
  runtimeVsPoints()
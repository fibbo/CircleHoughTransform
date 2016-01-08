import os
import pdb
import cPickle as pickle
import numpy as np

basePath = "/disk/data3/lhcb/phi/circleHT/splitData/run01/"
directories = sorted(os.listdir(basePath))
db = pickle.load(open("/home/hep/phi/CircleHoughTransform/src/db.pkl", 'rb'))

def bulk():
  numberOfFakeRings = 0
  numberOfFoundRings = 0
  numberOfTotalRings = 0
  numberOfMissedRings= 0
  numberOfDuplicates = 0
  duplicateRings = []
  ifile = open('fakeEvents.txt','wb')
  for directory in directories:
    path = os.path.join(basePath, directory)
    files = os.listdir(path)
    for afile in [f for f in files if f.endswith('.pkl')]:
      # remove db.pkls if they exist
      if afile == 'db.pkl':
        try:
          os.remove(os.path.join(path, afile))
          continue
        except Exception, e:
          print e.code, e.message, os.path.join(path, afile), directory
      res = pickle.load( open(os.path.join(path,afile), 'rb') )
      db_rings = db[int(afile[:-4])]['rings']
      if len(res['fakeRings']):
       ifile.write(afile[:-4]+"\n")
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


if __name__=='__main__':
  res = resultLoop(missedRings)
  print res
  pdb.set_trace()
  #bulk()


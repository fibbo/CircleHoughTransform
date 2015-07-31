import pickle
import os

import matplotlib.pyplot as plt

f = open('../db.pkl')

db = pickle.load(f)

def findMaxRadius( db ):
  max_radius = 0
  filename = ''
  for entry in db:
    for ring in entry['rings']:
      if ring['radius'] > max_radius:
        max_radius = ring['radius']
        filename = entry['filename']
  return max_radius, filename

def findCircleWithLeastPoints( db ):
  nPe = 99
  for entry in db:
    for ring in entry['rings']:
      if ring['nPe'] < nPe:
        nPe = ring['nPe']
        filename = entry['filename']

  return nPe, filename

def findDataWithXCircles( db, circles ):
  results = []

  for entry in db:
    if len(entry['rings']) == circles:
      results.append( os.path.basename(entry['filename']) )


  return results

def getCircleData( db, eventnumber ):
  for ring in db[eventnumber]['rings']:
    print "nPe: %s" % ring['nPe']
    print "radius: %s" % ring['radius']
    print "center: %s" % (ring['center'],)

def pointsPerCircleDistribution( db ):
  points = []
  for entry in db:
    for rings in entry['rings']:
      points.append(rings['nPe'])

  plt.hist(points, 30)
  plt.show()

def ringsPerEventDistribution( db ):
  rings = []
  for entry in db:
    rings.append( len(entry['rings'] ) )

  print len(rings)
  plt.hist(rings,15)
  plt.show()


#ringsPerEventDistribution( db )
pointsPerCircleDistribution(db)
#print findDataWithXCircles(db, 8)
# getCircleData( db, 2275)
# print findCircleWithLeastPoints(db)
#print findMaxRadius( db )
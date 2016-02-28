import pickle
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import poisson
import pdb

f = open('db.pkl')

db = pickle.load(f)

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)


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
    print "------"
    print "center: %s" % (ring['center'],)
    print "nPe: %s" % ring['nPe']
    print "radius: %s" % ring['radius']

def plotRadiusDistribution():
  radius_list = []
  for entry in db:
    for ring in entry['rings']:
      radius_list.append(ring['radius'])

  plt.hist(radius_list, 40,alpha=0.6,color='g')
  plt.xlabel('radius [m]')
  plt.ylabel('# of entries')
  plt.xlim(0.05,0.13)
  plt.savefig('../img/radiusDist.pdf')

def pointsPerCircleDistribution( ):
  points = []
  for entry in db:
    for ring in entry['rings']:
      points.append(float(ring['nPe']))

  # mu, std = poisson.fit(points)

  n, bins, patches = plt.hist(points,np.arange(5,45)-0.5,normed=False,alpha=0.6,color='g')
  # xmin, xmax = plt.xlim()
  # x = np.linspace(xmin, xmax, 100)
  # p = poisson.pdf(x, mu, std)
  # plt.plot(x, p, 'k', linewidth=2)
  # plt.title('$\mu=%.3f,\ \sigma=%.3f$' % (mu, std))
  plt.ylabel('# of circles')
  plt.xlabel('# of points')
  plt.savefig('../img/ppc.pdf')
  plt.close()

def ratioOfCirclesMoreThanXPoints(x=10):
  s_circle = 0
  b_circle = 0
  for entry in db:
    for ring in entry['rings']:
      if ring['nPe'] > x:
        b_circle += 1
      else:
        s_circle += 1

  #print s_circle
  return b_circle/float(s_circle+b_circle)


def plotRatioOfCirclesWithMoreThanXPoints():
  result = []
  for x in range(1,40):
    result.append(ratioOfCirclesMoreThanXPoints(x))

  plt.plot(result)
  plt.yticks(np.arange(0, 1, 0.1))
  plt.title('circles with more than $x$ points')
  plt.xlabel('number of points per circle')
  plt.ylabel('ratio')
  plt.savefig('../img/ratio_ppc.pdf')


def ringsPerEventDistribution( db ):
  rings = []
  for entry in db:
    rings.append( len(entry['rings'] ) )

  plt.hist(rings,np.arange(1,15)-0.5, alpha=0.6,color='g')
  plt.xticks(range(14))
  plt.xlabel('# of circles')
  plt.ylabel('# of events')
  plt.title('Circle per event distribution')
  plt.savefig('../img/circlePerEventDistribution.pdf')
  plt.close()

def maxPoints():
  maxPoints = 0
  for i,event in enumerate(db):
    nPoints = 0
    nPoints += event['nBkg']
    for ring in event['rings']:
      nPoints += ring['nPe']

    if nPoints > maxPoints:
      maxPoints = nPoints
      eventnumber = i
  print eventnumber

print ratioOfCirclesMoreThanXPoints(13)
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

basePath = "/home/phi/workspace/CircleHT/analysis/Threshold/split/run05/"
run = basePath[-6:-1]
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
  plt.xlabel('points')
  plt.ylabel('time [s]')
  plt.savefig('../img/runtime_vs_points_'+run+'.pdf')

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
  # if not kwargs:
  #   max_center_distance = REAL_MAX_CENTER_DISTANCE
  #   max_radius_distance = REAL_MAX_RADIUS_DISTANCE
  # else:
  #   for key in kwargs:
  #     if key=='radius':
  #       max_radius_distance = kwargs[key]
  #     if key=='center':
  #       max_center_distance = kwargs[key]

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


def removeDuplicates( results, **kwargs ):
  """ 

  """
  res = []
  sorted_results = sorted( results, key=lambda k: k['nEntries'], reverse=True)
  if not kwargs:
    max_center_distance = DUPLICATE_MAX_CENTER_DISTANCE
    max_radius_distance = DUPLICATE_MAX_RADIUS_DISTANCE

  else:
    for key in kwargs:
      if key=='radius':
        max_radius_distance = kwargs[key]
      if key=='center':
        max_center_distance = kwargs[key]
  while len(sorted_results):
    circle = sorted_results.pop()
    unique = True

    if any( (np.linalg.norm(np.array(circle['center']) - np.array(dic['center'])) < max_center_distance) and (abs(circle['radius'] - dic['radius']) < max_radius_distance) for dic in sorted_results ):
      unique=False
    # for dic in sorted_results:
    #   if (np.linalg.norm(np.array(circle['center']) - np.array(dic['center']))) < max_center_distance and\
    #      (abs(circle['radius'] - dic['radius']) < max_radius_distance):
    #     unique = False
    #     break
    if unique:
      res.append(circle)

  return res


def totalEfficiency(**kwargs):
  pkls = getPKLs()
  found_circles = 0
  missed_circles = 0
  fake_circles = 0
  tot_circles = 0
  counter = 0
  containsDuplicates = []
  missedDuplicates = 0
  if not kwargs:
    print pkls[0][1]['Parameters']
  for pkl in pkls:
    allRings = pkl[1]['allRings']
    if kwargs:
      result = removeDuplicates(allRings, radius=kwargs['radius'], center=kwargs['center'])
    else:
      result = removeDuplicates(allRings)
    # pdb.set_trace()
    db_entry = db[int(pkl[0])]['rings']
    rings = copy.deepcopy(db_entry)
    tot_circles += len(db_entry)
    resultrings = copy.deepcopy(result)
    found, fake, missed = compareRings(rings, result)
    found_circles += len(found)
    fake_circles += len(fake)
    if len(fake):
      containsDuplicates.append(pkl[0])
    missed_circles += len(missed)
    # if not len(found)+len(missed)==len(db_entry):
    #   print pkl[0]
    if MAKEPLOTS:
      data = readFile(filefolder+'Event0000'+pkl[0]+'.txt')
      x,y = zip(*data['allPoints'])
      destPathSim = basePath+'img/'+pkl[0]+'_afterCleanup.pdf'
      destPathReal = basePath+'img/'+pkl[0]+'_real.pdf'
      destPathPure = basePath+'img/'+pkl[0]+'_pure.pdf'
      plotData(x,y,found,savePath=destPathSim)
      plotData(x,y,resultrings,savePath=destPathPure)  
      plotData(x,y,rings,savePath=destPathReal)
  if not kwargs:
    print("efficiency: %s" % ((tot_circles-missed_circles)/float(tot_circles)))
    print("fake efficiency: %s" % (fake_circles/float(tot_circles)))
    print("missed: %s" % missed_circles)
    print("fakes: %s" % fake_circles)
    efficiency = ((tot_circles-missed_circles)*100/float(tot_circles))
    ghostrate = (fake_circles*100/float(tot_circles))
    print("%.3f & %.3f & %.2f\%% & %s & %.2f\%% & %s \\\\" % (DUPLICATE_MAX_RADIUS_DISTANCE, DUPLICATE_MAX_CENTER_DISTANCE, efficiency, missed_circles, ghostrate, fake_circles))
    # print containsDuplicates
  else:
    efficiency = ((tot_circles-missed_circles)*100/float(tot_circles))
    ghostrate = (fake_circles*100/float(tot_circles))
    print("%.3f & %.3f & %.2f\%% & %s & %.2f\%% & %s \\\\" % (kwargs['radius'], kwargs['center'], efficiency, missed_circles, ghostrate, fake_circles))

def singleEfficiency(MAKEPLOTS, EVENTNUMBER):
  pkls = getPKLs()
  for pkl in pkls:
    if pkl[0] == EVENTNUMBER:
      allRings = pkl[1]['allRings']
      db_entry = db[int(pkl[0])]['rings']
      rings = copy.deepcopy(db_entry)
      result = removeDuplicates(allRings)
      found, fake, missed = compareRings(db_entry, result)

      if MAKEPLOTS:
        data = readFile(filefolder+'Event0000'+pkl[0]+'.txt')
        x,y = zip(*data['allPoints'])
        destPathSim = basePath+'img/'+pkl[0]+'.pdf'
        destPathReal = basePath+'img/'+pkl[0]+'_real.pdf'
        plotData(x,y,found,savePath=destPathSim)
        plotData(x,y,rings,savePath=destPathReal)

      print("Circles that don't have a match in the event: %s" % len(fake))
      print("Missed circles: %s" % len(missed))
      break

def localSingleEfficiency(MAKEPLOTS, EVENTNUMBER):
  pkl = (EVENTNUMBER, pickle.load(open('../analysis/localPKLs/'+EVENTNUMBER+'.pkl','rb')))
  allRings = pkl[1]['allRings']
  db_entry = db[int(pkl[0])]['rings']
  rings = copy.deepcopy(db_entry)
  result = removeDuplicates(allRings)
  resultrings = copy.deepcopy(result)
  found, fake, missed = compareRings(db_entry, result)

  if MAKEPLOTS:
    data = readFile(filefolder+'Event0000'+pkl[0]+'.txt')
    x,y = zip(*data['allPoints'])
    destPathSim = basePath+'img/'+pkl[0]+'_afterCleanup.pdf'
    destPathReal = basePath+'img/'+pkl[0]+'_real.pdf'
    destPathPure = basePath+'img/'+pkl[0]+'_pure.pdf'
    plotData(x,y,found,savePath=destPathSim)
    plotData(x,y,resultrings,savePath=destPathPure)  
    plotData(x,y,rings,savePath=destPathReal)

  print("Circles that don't have a match in the event: %s" % len(fake))
  print("Missed circles: %s" % len(missed))
  print("Found/Real %s/%s" % (len(found),len(rings)))


if __name__=='__main__':
  #res = resultLoop(missDuplicates)
  #print res
  cuts = np.arange(0.003,0.012,0.003)
  centers = [0.003]
  runtimeVsPoints()
  # if len(sys.argv)>=2:
  #   MAKEPLOTS = sys.argv[1]
  # else:
  #   MAKEPLOTS = False
  # if len(sys.argv)==3:
  #   EVENTNUMBER=sys.argv[2]
  #   localSingleEfficiency(MAKEPLOTS, EVENTNUMBER)
  # else:
  #   if False:
  #     for radius in cuts:
  #       for center in centers:
  #         totalEfficiency(radius=radius, center=center)
  #   else:
  #     totalEfficiency()
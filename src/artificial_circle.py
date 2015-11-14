import math
import matplotlib.pyplot as plt
import numpy as np
import random

def generateCircle(r, c, fakePoint=False, background=False):
  if not type(c) == tuple:
    raise TypeError
  steps = 30
  stepSize = 2*math.pi/steps
  points = []
  for i in range(steps):
    alpha = random.uniform(0,2*math.pi)
    beta = random.uniform(0,2*math.pi)
    points.append( np.array( (c[0]+r*math.cos(alpha), c[1]+r*math.sin(alpha)) ) )
    points.append( np.array( (c[0]+0.9*r*math.cos(beta), c[1]+0.9*r*math.sin(beta)) ) )
  if fakePoint:
    devFactor = 1.05
    alpha = random.uniform(0,2*math.pi)
    points.append( np.array( (c[0]+r*devFactor*math.cos(alpha), c[1]+r*devFactor*math.sin(alpha)) ) )
  if background:
    i = background
    while i > 0:
      points.append( np.array( (random.uniform(-0.5,0.5), random.uniform(-0.5,0.5)) ) )
      i -= 1

  return points


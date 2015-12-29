import numpy as np

a = np.random.randn(100)
b = np.random.randn(100) 

def funcA():
  x,y = np.broadcast_arrays(a[...,np.newaxis], b[np.newaxis,...])
  z = (1-x)**2 + (2-y)**2 - 9
  return z
  
def funcB():
  for x in a:
    for y in b:
      z = (1-x)**2 + (2-y)**2 - 9
  return z

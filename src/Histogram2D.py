from math import ceil

class Histogram2D():
  def __init__( self, xbin_width, ybin_width, xmin, xmax, ymin, ymax, radius ):
    self.xsize = int( ceil( ( xmax - xmin ) / xbin_width ) )
    self.ysize = int( ceil( ( ymax - ymin ) / ybin_width ) )
    self.hist = [[0 for i in range( self.xsize )] for j in range( self.ysize )]
    self.radius = radius
    self.n_entries = 0
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    # print '__init__: binwidth: %d' % bin_width
    self.xbin_width = xbin_width
    self.ybin_width = ybin_width
    # print '__init__: self.bin_width: %d' % self.bin_width

  def printHistogramVariables( self ):
    print '#########################'
    print 'Size: %s x %s' % ( self.xsize, self.ysize )
    print 'Number of entries: %d' % self.n_entries
    print 'Minimum value (xmin, ymax): (%s,%s)' % ( self.xmin, self.ymin )
    print 'Maximum value (xmax, ymax): (%s,%s)' % ( self.xmax, self.ymax )
    print 'Bin width (xwidth, ywidth):(%s,%s)' % ( self.xbin_width, self.ybin_width )
    print '#########################'


  def findBin( self, value ):
    if not type( value ) == tuple:
      return -999
    else:
      x = float( ( value[0] - self.xmin ) / self.xbin_width )
      y = float( ( value[1] - self.ymin ) / self.ybin_width )
      return ( x, y )

  def printHistogram( self ):
    for line in self.hist:
      print line

  def addValue( self, value ):
    if not type( value ) == tuple:
      raise TypeError
    xbin = int( ( value[0] - self.xmin ) / self.xbin_width )
    ybin = int( ( value[1] - self.ymin ) / self.ybin_width )
    if ( xbin >= 0 and xbin < self.xsize ) and ( ybin >= 0 and ybin < self.ysize ):
        self.hist[xbin][ybin] += 1
        self.n_entries += 1

  def getValue( self, hbin ):
    if not type( hbin ) == tuple:
      raise TypeError
      # print 'bin: %d \t self.bin_width: %d \t self.min: %d' % (bin, self.bin_width, self.min)
    xvalue = float( hbin[0] * self.xbin_width ) + self.xbin_width / 2. + self.xmin
    yvalue = float( hbin[1] * self.ybin_width ) + self.ybin_width / 2. + self.ymin
    return ( xvalue, yvalue )

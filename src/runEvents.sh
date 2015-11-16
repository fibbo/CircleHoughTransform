#!/bin/bash

basedir="/disk/data3/lhcb/phi/circleHT"
eventdir="/disk/data1/hep/che/HoughTransform/serie0/files"
mkdir -p $basedir
for event_id in {1..1000..10}; do
  qsub -l cput=02:00:00 -v event=$event_id submitData.pbs
done

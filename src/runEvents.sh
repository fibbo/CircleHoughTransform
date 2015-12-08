#!/bin/bash

for event_id in {0010..9999..10}; do
  qsub -l cput=02:00:00 -v event=$event_id submitData.pbs
done

#!/bin/bash

<<<<<<< HEAD
for event_id in {0..9999..40}; do
=======
for event_id in {0010..9999..10}; do
>>>>>>> origin/split_data
  qsub -l cput=02:00:00 -v event=$event_id submitData.pbs
done

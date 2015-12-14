#!/bin/bash
files_per_job=40
for event_id in {0..9999..40}; do
  qsub -l cput=02:00:00 -v event=$event_id,files_per_job=$files_per_job submitData.pbs
done

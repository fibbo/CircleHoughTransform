#!/bin/bash

# #Strips=('B2KpiX2MuMuDarkBosonLine')
# Strips=('B2KpiX2EEDarkBosonLine')
# selSteps=('L0Hadron' 'L0Electron' 'L0TIS' 'Comb')
# selection="Vetoes"
# Years=('2011' '2012')

# baseDir="/disk/data3/lhcb/rsilvaco/RareDecays/Bd2KpiEE/Ntuples/Data/"
# scriptsDir="/home/hep/rsilvaco/Analysis/RareDecays/Bd2KpiEE/Ntuples/Scripts/"
# # mkdir -p $scriptsDir/$selection/logs/

# for year in "${Years[@]}"; do 
#   echo $year
#   # mkdir -p $baseDir/$year/$selection/
#   for sel in "${selSteps[@]}"; do
#     echo $sel
#     # mkdir -p $baseDir/$year/$selection/$sel/
#     # for config in "${Strips[@]}"
#     # do 
#     #    if [ ${sel} != 'Comb' ] && [ ${config} == 'B2KpiX2MuMuDarkBosonLine' ]
#     #    then
#     #       continue
#     #    else 
#     #       qsub -l cput=10:00:00 -v year=$year,selection=$selection,selTrig=$sel,stripping=$config submitData.pbs
#     #    fi
#     # done
#   done
# done

basedir="/disk/data3/lhcb/phi/circleHT"
mkdir -p $basedir
for event in {0001,..,10, 5}; do
  echo Running: qsub -l cput=02:00:00 -v event=$event submitData.pbs
  qsub -l cput=02:00:00 -v event=$event submitData.pbs
done
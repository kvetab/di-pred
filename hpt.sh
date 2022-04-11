#!/bin/bash
#PBS -N hpt
#PBS -q cheminf
#PBS -l select=1:ncpus=2:ngpus=1:mem=8gb
#PBS -l walltime=200:00:00
#PBS -m ae

DATADIR=/home/$USER/diplomka/project/data/

SCRATCHDIR=/scratch/$USER/$PBS_JOBID
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

mkdir $SCRATCHDIR/
mkdir $SCRATCHDIR/chen 
mkdir $SCRATCHDIR/bin
cp -r $DATADIR/chen/* $SCRATCHDIR/chen || { echo >&2 "Error while copying input file(s)"; exit 2; }
mkdir $SCRATCHDIR/evaluations
mkdir $SCRATCHDIR/evaluations/hyperparameters
cp /home/$USER/diplomka/project/bin/hyperparameter_tuning.py $SCRATCHDIR/bin/

cd $SCRATCHDIR

#module add lich/cuda-10.2


######## job #######

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/$USER/.conda/envs/ml/
python3 bin/hyperparameter_tuning.py --prepro 1 --model 5 || { echo >&2 "Error during job execution"; exit 3; }

mkdir $DATADIR/$PBS_JOBID
######## copy results ####
cp -r evaluations/* $DATADIR/$PBS_JOBID

cd ..
rm -r $SCRATCHDIR/*
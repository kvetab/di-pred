#!/bin/bash
#PBS -N train_round
#PBS -q cheminf
#PBS -l select=1:ncpus=2:ngpus=1:mem=8gb
#PBS -l walltime=200:00:00
#PBS -m ae

DATADIR=/home/$USER/diplomka/project/data/

SCRATCHDIR=/scratch/$USER/$PBS_JOBID
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

mkdir $SCRATCHDIR/
mkdir $SCRATCHDIR/chen 
mkdir $SCRATCHDIR/tap
mkdir $SCRATCHDIR/bin
mkdir $SCRATCHDIR/hyperparameters
cp -r $DATADIR/chen/* $SCRATCHDIR/chen || { echo >&2 "Error while copying input file(s)"; exit 2; }
cp -r $DATADIR/tap/* $SCRATCHDIR/tap || { echo >&2 "Error while copying input file(s)"; exit 2; }
cp -r $DATADIR/evaluations/hyperparameters/* $SCRATCHDIR/hyperparameters || { echo >&2 "Error while copying input file(s)"; exit 2; }
mkdir $SCRATCHDIR/evaluations
cp /home/$USER/diplomka/project/bin/06_high_level_training.py $SCRATCHDIR/bin/

cd $SCRATCHDIR

#module add lich/cuda-10.2


######## job #######

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/$USER/.conda/envs/ml/
# seeds = [4, 18, 27, 36, 42]
python3 bin/06_high_level_training.py --seed 42 --prepro 1 || { echo >&2 "Error during job execution"; exit 3; }


echo "Number of available cpus: $(nproc)" > cpu_info.txt
#nvidia-smi > nvidia_info.txt
#nvcc --version > cuda_info.txt

mkdir $DATADIR/$PBS_JOBID
######## copy results ####
cp -r evaluations/* $DATADIR/$PBS_JOBID #|| { echo >&2 "Result files copying failed (with a code $?)! You can retrieve your files from `hostname -f`:`pwd`"; exit 4 }


# cp *_info.txt $DATADIR/$PBS_JOBID || { echo >&2 "Result files copying failed (with a code $?)! You can retrieve your files from `hostname -f`:`pwd`"; exit 4 }

cd ..
rm -r $SCRATCHDIR/*
#!/bin/bash
#PBS -N sapiens
#PBS -q cheminf
#PBS -l select=1:ncpus=2:ngpus=1:mem=8gb
#PBS -l walltime=200:00:00
#PBS -m ae

DATADIR=/home/$USER/diplomka/project/fairseq/data/

SCRATCHDIR=/scratch/$USER/$PBS_JOBID
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

mkdir $SCRATCHDIR/
mkdir $SCRATCHDIR/models
mkdir $SCRATCHDIR/source

MODEL_NAME=03_pretrained_2000epochs
MODEL_DIR=$SCRATCHDIR/models/heavy/$MODEL_NAME
mkdir -p $MODEL_DIR

cp -r $DATADIR/* $SCRATCHDIR || { echo >&2 "Error while copying input file(s)"; exit 2; }
cp -r /home/brazdilv/diplomka/SW/Sapiens/sapiens $SCRATCHDIR/source

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate $CONDA_ENV_PATH

fairseq-train \
    $DATADIR/processed/heavy/train/  \
    --user-dir /home/brazdilv/diplomka/SW/Sapiens/sapiens \
    --init-token 0 --separator-token 2 \
    --restore-file /home/brazdilv/diplomka/SW/Sapiens/sapiens/models/v1/checkpoint_vh.pt \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir $MODEL_DIR/checkpoints \
    --tensorboard-logdir $MODEL_DIR/tensorboard \
    --arch roberta_small \
    --criterion sentence_prediction \
    --task sentence_prediction \
    --num-classes 2 \
    --optimizer adam \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 \
    --max-positions 144 \
    --shorten-method truncate \
    --batch-size 256 \
    --max-epoch 2000 \
    --log-format simple \
    --log-interval 100 \
    --validate-interval 1 \
    --save-interval 100 \
        2>&1 | tee $MODEL_DIR/log



mkdir $DATADIR/models/heavy/$MODEL_NAME
cp -r $MODEL_DIR $DATADIR/models/heavy/$MODEL_NAME

cd ..
rm -r $SCRATCHDIR/*
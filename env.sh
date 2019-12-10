#!/bin/bash 
source ~/.bashrc
source /esat/spchdisk/scratch/r0797363/venv/ScalableFHVAE/bin/activate
export PYTHONPATH=$PWD/kaldi_python/kaldi-python
export PYTHONPATH=$PYTHONPATH:/users/spraak/spch/prog/spch/tensorflow-1.0.1/lib/python2.7/site-packages:/users/spraak/spch/.local/lib/python2.7/site-packages
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=/users/spraak/spch/prog/spch/cuda-8.0/lib64:$LD_LIBRARY_PATH
echo Using Python from `which python`

nvidia-smi
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
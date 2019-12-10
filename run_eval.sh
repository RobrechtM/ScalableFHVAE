#!/bin/bash
. ./env.sh
python scripts/eval/run_eval.py \
    /esat/spchtemp/scratch/hvanhamm/fhvae_timit/exp/cgn_per_speaker/nogau_reg_fhvae_e99_s5000_p10_a10.0_b10.0_c10.0_e0.01 \
    --dataset=cgn_per_speaker \
    --set_name=train \
    --seqlist=misc/cgn_per_speaker.train.list


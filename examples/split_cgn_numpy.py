"""
split CGN data in smaller chunks for FHVAE
"""
import os
import wave
import argparse
import subprocess
from sphfile import SPHFile
import numpy as np

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--db_dir", type=str, default="./datasets/cgn_o_np_fbank/test",
        help="npy data directory")
args = parser.parse_args()
print args

in_len = "%s/len_long.scp" % args.db_dir
out_len = "%s/len.scp" % args.db_dir
in_feat = "%s/feats_long.scp" % args.db_dir
out_feat = "%s/feats.scp" % args.db_dir

# in_len_f = open(in_len, "r")
# in_feat_f = open(in_feat, "r")
out_len_f = open(out_len, "w")
out_feat_f = open(out_feat, "w")

with open(in_len) as f:
    long_len = dict (line.rstrip().split(None,1) for line in f)
for spk in long_len.keys():
    long_len[spk]=int(long_len[spk])

with open(in_feat) as f:
    long_feat = dict (line.rstrip().split(None,1) for line in f)

for spk in long_len.keys():
    Nblocks = long_len[spk]/500 + 1
    blen = long_len[spk] / Nblocks
    feats = np.load(long_feat[spk])
    start = 0
    for block in range(Nblocks):
        fname = long_feat[spk].replace(".npy","_%d.npy" % block)
        np.save(fname,feats[start:start+blen,:])
        out_len_f.write("%s_%d %d\n" % (spk,block,blen))
        out_feat_f.write("%s_%d %s\n" % (spk,block,fname))
        start+=blen

out_feat_f.close()
out_len_f.close()

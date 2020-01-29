"""
prepare CGN data for FHVAE
"""
import os
import wave
import argparse
import subprocess
from sphfile import SPHFile

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("cgn_dir", type=str, help="CGN raw data directory")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
        help="feature type")
parser.add_argument("--out_dir", type=str, default="./datasets/cgn_o_np_fbank",
        help="output data directory")
parser.add_argument("--dev_spk", type=str, default="./misc/cgn_o_dev_spk.list",
        help="path to list of dev set speakers")
parser.add_argument("--test_spk", type=str, default="./misc/cgn_o_test_spk.list",
        help="path to list of test set speakers")
args = parser.parse_args()
print args

# retrieve partition
with open(args.dev_spk) as f:
    dt_spks = [line.rstrip()[0:-2].lower() for line in f]
with open(args.test_spk) as f:
    tt_spks = [line.rstrip()[0:-2].lower() for line in f]
# dt_spks=[]
# tt_spks=[]

print "making wav.scp"

wav_dir = os.path.abspath("%s/wav" % args.out_dir)
tr_scp = "%s/train/wav.scp" % args.out_dir
dt_scp = "%s/dev/wav.scp" % args.out_dir
tt_scp = "%s/test/wav.scp" % args.out_dir

maybe_makedir(wav_dir)
maybe_makedir(os.path.dirname(tr_scp))
maybe_makedir(os.path.dirname(dt_scp))
maybe_makedir(os.path.dirname(tt_scp))

tr_f = open(tr_scp, "w")
dt_f = open(dt_scp, "w")
tt_f = open(tt_scp, "w")

paths = []
for root, _, fnames in sorted(os.walk(args.cgn_dir)):
    regio = root.split("/")[-1].lower()
    comp = root.split("/")[-2].lower()
    if comp not in ["comp-o"]: #["comp-o","comp-l","comp-m","comp-n"]:
        continue
    for fname in fnames:
        if fname.endswith(".wav") or fname.endswith(".WAV"):
            spk = fname.split(".")[0].lower()
            if spk in dt_spks:
                f = dt_f
            elif spk in tt_spks:
                f = tt_f
            else:
                f = tr_f
            path = "%s/%s/%s/%s" % (args.cgn_dir, comp, regio, fname)
            uttid = "%s_%s" % (spk, comp.split("-")[1])
            f.write("%s %s\n" % (uttid, path))

tr_f.close()
dt_f.close()
tt_f.close()

print "converted to wav and dumped scp files"

# compute feature
feat_dir = os.path.abspath("%s/%s" % (args.out_dir, args.ftype))
maybe_makedir(feat_dir)

def compute_feature(name):
    cmd = ["python", "./scripts/preprocess/prepare_numpy_data.py", "--ftype=%s" % args.ftype]
    cmd += ["%s/%s/wav.scp" % (args.out_dir, name), feat_dir]
    cmd += ["%s/%s/feats_long.scp" % (args.out_dir, name)]
    cmd += ["%s/%s/len_long.scp" % (args.out_dir, name)]
    
    p = subprocess.Popen(cmd)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))

for name in ["train", "dev", "test"]:
    compute_feature(name)

print "computed feature"

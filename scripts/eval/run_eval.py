import os
import sys
import time
import cPickle
import argparse
import tensorflow as tf
from collections import OrderedDict
from eval_loaders import load_data
from fhvae.models import load_model
from fhvae.runners.fhvae_utils import load_prog, get_best_step
from fhvae.runners.test_fhvae import test, visualize, tsne_by_label, shift_reg

print "I am process %s, running on %s: starting (%s)" % (
    os.getpid(), os.uname()[1], time.asctime())

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("exp_dir", type=str,
                    help="experiment directory")
parser.add_argument("--set_name", type=str, default="dev",
                    help="name of dataset partition to evaluate")
parser.add_argument("--seqlist", type=str, default=None,
                    help="specify a list of sequences to evaluate; randomly sample 10 by default")
parser.add_argument("--step", type=int, default=-1,
                    help="step of the model to load. -1 for the best")
parser.add_argument("--dataset", type=str, default=None,
                    help="name of dataset to evaluate, e.g. grabo_np_timit, intra database by default")
parser.add_argument("--write_spec", action='store_true',
                    help="Write spectrograms")
parser.add_argument("--write_gender", action='store_true')
parser.add_argument("--write_phon", action='store_true')
parser.add_argument("--viz_reconstruction", action='store_true')
parser.add_argument("--viz_factorization", action='store_true')
parser.add_argument("--write_neutral", action='store_true')
parser.add_argument("--translate", action='store_true')
parser.add_argument("--tsne", action='store_true')
args = parser.parse_args()

exp_dir, set_name, seqlist, step, dataset = args.exp_dir, args.set_name, args.seqlist, args.step, args.dataset

with open("%s/args.pkl" % exp_dir, "rb") as f:
    print "load hparams from %s/args.pkl" % exp_dir
    hparams = cPickle.load(f)
    print hparams

used_labs = hparams.facs.split(':')
c_n = OrderedDict([(used_labs[0], 5), (used_labs[1], 3)])  # hack
used_talabs = hparams.talab_facs.split(':')
b_n = OrderedDict([(used_talabs[0], 48)])  # hack

if dataset == None:
    dataset = hparams.dataset
dt_iterator, dt_iterator_by_seqs, dt_seqs, dt_seq2lab_d = \
    load_data(dataset, set_name, hparams.is_numpy, seqlist)
FHVAE = load_model(hparams.model)

if hasattr(hparams, "nmu2"):
    print "model trained with hierarchical sampling, nmu2=%s" % hparams.nmu2
    nmu2 = hparams.nmu2
else:
    print "model trained with normal training, nmu2=%s" % hparams.tr_nseqs
    nmu2 = hparams.tr_nseqs

tf.reset_default_graph()

xin = tf.placeholder(tf.float32, shape=(None,)+hparams.tr_shape, name="xin")
xout = tf.placeholder(tf.float32, shape=(None,)+hparams.tr_shape, name="xout")
y = tf.placeholder(tf.int64, shape=(None,), name="y")
n = tf.placeholder(tf.float32, shape=(None,), name="n")
#cReg = tf.placeholder(tf.int64, shape=(None,len(used_labs)), name="cReg")
#model = FHVAE(xin, xout, y, n, nmu2, cReg, nlabs)
#b_n = OrderedDict([(talab,tr_dset.talabseqs_d[talab].nclass) for talab in used_talabs])
bReg = tf.placeholder(tf.int64, shape=(None,) + (len(b_n),), name="bReg")
# c_n = OrderedDict([(lab,tr_dset.labs_d[lab].nclass) for lab in used_labs]) #ordering correct ?
cReg = tf.placeholder(tf.int64, shape=(None,) + (len(c_n),), name="cReg")
model = FHVAE(xin, xout, y, n, nmu2, bReg, cReg, b_n, c_n)
print(model)

prog, _, _, _, _, _ = load_prog("%s/prog.pkl" % exp_dir)
step = get_best_step(prog) if step == -1 else step
# test(exp_dir, step, model, dt_iterator)
shift_reg(exp_dir, step, model, dt_iterator_by_seqs,  dt_seqs, dt_seq2lab_d)
visualize(exp_dir, step, model, dt_iterator_by_seqs, dt_seqs, args)
if dt_seq2lab_d is not None and args.tsne:
    tsne_by_label(exp_dir, step, model, dt_iterator_by_seqs,
                  dt_seqs, dt_seq2lab_d)

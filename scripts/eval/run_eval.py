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
from fhvae.runners.test_fhvae import test, visualize, tsne_by_label

print "I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
args = parser.parse_args()

exp_dir, set_name, seqlist, step, dataset = args.exp_dir, args.set_name, args.seqlist, args.step, args.dataset

with open("%s/args.pkl" % exp_dir, "rb") as f:
    print "load arguments from %s/args.pkl" % exp_dir
    args = cPickle.load(f)
    print args

used_labs = args.facs.split(':')
c_n = OrderedDict([(used_labs[0],5),(used_labs[1],3)]) # hack
used_talabs = args.talab_facs.split(':')
b_n = OrderedDict([(used_talabs[0],48)]) # hack

if dataset==None:
    dataset = args.dataset
dt_iterator, dt_iterator_by_seqs, dt_seqs, dt_seq2lab_d = \
        load_data(dataset, set_name, args.is_numpy, seqlist)
FHVAE = load_model(args.model)

if hasattr(args, "nmu2"):
    print "model trained with hierarchical sampling, nmu2=%s" % args.nmu2
    nmu2 = args.nmu2
else:
    print "model trained with normal training, nmu2=%s" % args.tr_nseqs
    nmu2 = args.tr_nseqs

tf.reset_default_graph()

xin = tf.placeholder(tf.float32, shape=(None,)+args.tr_shape, name="xin")
xout = tf.placeholder(tf.float32, shape=(None,)+args.tr_shape, name="xout")
y = tf.placeholder(tf.int64, shape=(None,), name="y")
n = tf.placeholder(tf.float32, shape=(None,), name="n")
#cReg = tf.placeholder(tf.int64, shape=(None,len(used_labs)), name="cReg")
#model = FHVAE(xin, xout, y, n, nmu2, cReg, nlabs)
#b_n = OrderedDict([(talab,tr_dset.talabseqs_d[talab].nclass) for talab in used_talabs])
bReg = tf.placeholder(tf.int64, shape=(None,) + (len(b_n),) , name="bReg")
#c_n = OrderedDict([(lab,tr_dset.labs_d[lab].nclass) for lab in used_labs]) #ordering correct ?
cReg = tf.placeholder(tf.int64, shape=(None,) + (len(c_n),), name="cReg")
model = FHVAE(xin, xout, y, n, nmu2, bReg, cReg, b_n, c_n)
print(model)

prog, _, _, _, _, _ = load_prog("%s/prog.pkl" % exp_dir)
step = get_best_step(prog) if step == -1 else step

test(exp_dir, step, model, dt_iterator)
visualize(exp_dir, step, model, dt_iterator_by_seqs, dt_seqs)
if dt_seq2lab_d is not None:
    tsne_by_label(exp_dir, step, model, dt_iterator_by_seqs, dt_seqs, dt_seq2lab_d)

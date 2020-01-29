import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from hs_train_loaders import load_data
from fhvae.models import load_model
from fhvae.runners.hs_train_fhvae import hs_train

print "I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="timit",
        help="dataset to use")
parser.add_argument("--is_numpy", action="store_true", dest="is_numpy",
        help="dataset format; kaldi by default")
parser.add_argument("--model", type=str, default="fhvae",
        help="model architecture; {fhvae|simple_fhvae}")
parser.add_argument("--alpha_dis", type=float, default=10.,
        help="discriminative objective weight")
parser.add_argument("--alpha_reg", type=float, default=100.,
        help="regularization weight")
parser.add_argument("--nmu2", type=int, default=5000,
        help="number of sequences for hierarchical sampling")
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_patience", type=int, default=10,
        help="number of maximum consecutive non-improving epochs")
parser.add_argument("--n_steps_per_epoch", type=int, default=5000,
        help="number of training steps per epoch")
parser.add_argument("--fac_root", type=str, default=None,
        help="file name with training factors")
parser.add_argument("--facs", type=str, default=None,
        help="list of regularizing training factors")
parser.add_argument("--adam_eps", type=float, default=1.0e-8,
        help="epsilon parameter of adam")
parser.add_argument("--n_print_steps", type=int, default=200,
        help="number of steps to print statistics")
args = parser.parse_args()
print args

#fac_file = args.fac_root % args.facs
tr_nseqs, tr_shape, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator, tr_dset = \
        load_data(args.dataset, args.is_numpy, args.fac_root, args.facs)

FHVAE = load_model(args.model)

exp_root = "exp/%s" % args.dataset

xin = tf.placeholder(tf.float32, shape=(None,)+tr_shape, name="xin")
xout = tf.placeholder(tf.float32, shape=(None,)+tr_shape, name="xout")
y = tf.placeholder(tf.int64, shape=(None,), name="y")
n = tf.placeholder(tf.float32, shape=(None,), name="n")
used_labs = args.facs.split(':')
c_n = OrderedDict([(lab,tr_dset.labs_d[lab].nclass) for lab in used_labs]) #ordering correct ?
# for lab in c_n.keys():
#         cReg[lab] = tf.placeholder(tf.int64, shape=(None,), name="cReg_%s" % lab)
cReg = tf.placeholder(tf.int64, shape=(None,) + (len(c_n),) , name="cReg")
model = FHVAE(xin, xout, y, n, args.nmu2, cReg, c_n)
print(model)

# keep necessary information in args for restoring model
args.tr_shape = tr_shape

train_conf = [args.n_epochs, args.n_patience, args.n_steps_per_epoch,
        args.n_print_steps, args.alpha_dis, args.nmu2, args.adam_eps, args.alpha_reg]
exp_dir = "%s/%s_hs_e%s_s%s_p%s_a%s_b%s_n%s_e%1.7g" % (exp_root, args.model, args.n_epochs,
        args.n_steps_per_epoch, args.n_patience, args.alpha_dis, args.alpha_reg, args.nmu2, args.adam_eps)

hs_train(exp_dir, model, args, train_conf, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator)

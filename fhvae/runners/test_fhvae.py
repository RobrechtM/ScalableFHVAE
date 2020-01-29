from .plotter import plot_x, scatter_plot
from .fhvae_utils import restore_model, _valid
from ..models.fhvae_fn import z1_mu_fn, z2_mu_fn, x_mu_fn, x_logvar_fn, map_mu2_z2, reg_posteriors_z1, reg_posteriors_z2
import tensorflow as tf
from sklearn.manifold import TSNE
import os
import sys
import time
import cPickle
import numpy as np
from collections import defaultdict
import matplotlib
import numpy as np
import logging
logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
matplotlib.use("Agg")

SESS_CONF = tf.ConfigProto(allow_soft_placement=True)
SESS_CONF.gpu_options.per_process_gpu_memory_fraction = 0.9

np.random.seed(123)


def test(exp_dir, step, model, iterator):
    """
    compute variational lower bound
    """
    xin, xout, y, n = model.xin, model.xout, model.y, model.n
    sum_names = ["lb", "log_px_z", "neg_kld_z1", "neg_kld_z2", "log_pmu2"]
    sum_vars = map(tf.reduce_mean, [model.__dict__[name]
                                    for name in sum_names])

    # util fn
    def _feed_dict(x_val, y_val, n_val):
        return {xin: x_val, xout: x_val, y: y_val, n: n_val}

    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        stime = time.time()
        restore_model(sess, saver, "%s/models" % exp_dir, step)
        print "restore model takes %.2f s" % (time.time() - stime)

        def _print_prog(sum_names, sum_vals):
            msg = " ".join(["%s=%.2f" % p for p in zip(sum_names, sum_vals)])
            print msg
            if np.isnan(sum_vals[0]):
                if_diverged = True
            sys.stdout.flush()

        print("#" * 40)
        vtime = time.time()
        sum_vals = _valid(sess, model, sum_vars, iterator)
        now = time.time()
        print("validation takes %.fs" % (now - vtime,))
        _print_prog(sum_names, sum_vals)
        print("#" * 40)


def visualize(exp_dir, step, model, iterator_by_seqs, seqs, args):
    """
    visualize reconstruction, factorization, sequence-translation
    """
    logging.info("Starting vizualization...")

    if len(seqs) > 10:
        logging.info(">25 seqs. randomly select 25 seqs for visualization")
        seqs = sorted(list(np.random.choice(seqs, 10, replace=False)))

    if not os.path.exists("%s/img" % exp_dir):
        os.makedirs("%s/img" % exp_dir)

    if not os.path.exists("%s/spec" % exp_dir):
        os.makedirs("%s/spec" % exp_dir)

    if not os.path.exists("%s/wav" % exp_dir):
        os.makedirs("%s/wav" % exp_dir)

    if not os.path.exists("%s/txt" % exp_dir):
        os.makedirs("%s/txt" % exp_dir)

    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        stime = time.time()
        restore_model(sess, saver, "%s/models" % exp_dir, step)
        logging.info("restore model takes %.2f s" % (time.time() - stime))
        # infer z1, z2, mu2
        z1_by_seq = defaultdict(list)
        z2_by_seq = defaultdict(list)
        mu2_by_seq = dict()
        regpost_by_seq = dict()
        xin_by_seq = defaultdict(list)
        xout_by_seq = defaultdict(list)
        xoutv_by_seq = defaultdict(list)
        z1reg_by_seq = defaultdict(list)
        for i, seq in enumerate(seqs):
            if i % 10 == 0:
                logging.info("encoding sequence %d/%d" % (i, len(seqs)))
            for x, _, _, _, _ in iterator_by_seqs([seq]):
                z2 = z2_mu_fn(sess, model, x)
                z1 = z1_mu_fn(sess, model, x, z2)
                xout = x_mu_fn(sess, model, z1, z2)
                xoutv = x_logvar_fn(sess, model, z1, z2)
                z1reg = reg_posteriors_z1(sess, model, z1)
                z1_by_seq[seq].append(z1)
                z2_by_seq[seq].append(z2)
                xin_by_seq[seq].append(x)
                xout_by_seq[seq].append(xout)
                xoutv_by_seq[seq].append(xoutv)
                z1reg_by_seq[seq] = z1reg
            z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
            z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)
            xin_by_seq[seq] = np.concatenate(xin_by_seq[seq], axis=0)
            xout_by_seq[seq] = np.concatenate(xout_by_seq[seq], axis=0)
            xoutv_by_seq[seq] = np.concatenate(xoutv_by_seq[seq], axis=0)
            # z1reg_by_seq[seq] = np.concatenate(z1reg_by_seq[seq], axis=0)
            mu2_by_seq[seq] = map_mu2_z2(model, z2_by_seq[seq])
            d2 = mu2_by_seq[seq].shape[0]
            z2 = np.asarray(mu2_by_seq[seq]).reshape([1, d2])
            regpost_by_seq[seq] = reg_posteriors_z2(sess, model, z2)

        # # save the mu2
        # f = open("mu2_by_seq.txt","w")
        # for seq in seqs:
        #     f.write( ' '.join (map(str,mu2_by_seq[seq])) )
        #     f.write('\n')
        # f.close()
        #
        # save the mean mu2
        if not os.path.exists("%s/mu2_by_seq.npy" % exp_dir):
            mumu = np.zeros([mu2_by_seq[seqs[1]].size])
            for seq in seqs:
                mumu += mu2_by_seq[seq]
            mumu /= len(seqs)
            with open("%s/mu2_by_seq.npy" % exp_dir, "wb") as fnp:
                np.save(fnp, mumu)

        if args.write_gender:
            names = ["reg1", "gender"]
            for i, name in enumerate(names):
                with open("%s/txt/%s.scp" % (exp_dir, name), "wb") as f:
                    for seq in seqs:
                        f.write(seq + "  [ ")
                        for e in np.nditer(regpost_by_seq[seq][i]):
                            f.write("%10.3f " % e)
                        f.write("]\n")

        if args.write_phon:
            names = ["pho"]
            for i, name in enumerate(names):
                try:
                    os.makedirs("%s/txt/%s" % (exp_dir, name))
                except OSError:
                    pass
                for seq in seqs:
                    np.save("%s/txt/%s/%s" %
                            (exp_dir, name, seq), z1reg_by_seq[seq][i])

        seq_names = ["%02d_%s" % (i, seq) for i, seq in enumerate(seqs)]

        if args.viz_reconstruction:
            # visualize reconstruction
            logging.info("visualizing reconstruction")
            plot_x([xin_by_seq[seq] for seq in seqs],
                   seq_names, "%s/img/xin.png" % exp_dir)
            plot_x([xout_by_seq[seq] for seq in seqs],
                   seq_names, "%s/img/xout.png" % exp_dir)
            plot_x([xoutv_by_seq[seq] for seq in seqs], seq_names,
                   "%s/img/xout_logvar.png" % exp_dir, clim=(None, None))

        if args.viz_factorization:
            # factorization: use the centered segment from each sequence
            logging.info("visualizing factorization")
            cen_z1 = np.array(
                [z1_by_seq[seq][len(z1_by_seq[seq]) / 2] for seq in seqs])
            cen_z2 = np.array(
                [z2_by_seq[seq][len(z2_by_seq[seq]) / 2] for seq in seqs])
            xfac = []
            for z1 in cen_z1:
                z1 = np.tile(z1, (len(cen_z2), 1))
                xfac.append(x_mu_fn(sess, model, z1, cen_z2))
            plot_x(xfac, seq_names, "%s/img/xfac.png" % exp_dir, sep=True)

        if args.write_spec or args.translate:
            logging.info("saving mel spectrograms to \"%s/spec/\"" % exp_dir)
            # with open( "./datasets/cgn_np_fbank/train/mvn.pkl" ) as f:
            # with open( "./datasets/timit_np_fbank/train/mvn.pkl" ) as f:
            # with open( "./datasets/cgn_per_speaker/train/mvn.pkl" ) as f:
            # with open( "./datasets/grabo_np_fbank/train/mvn.pkl" ) as f:
            with open("./datasets/cgn_per_speaker_afgklno/train/mvn.pkl") as f:
                mvn_params = cPickle.load(f)
            nb_mel = mvn_params["mean"].size

            if args.write_spec:
                for src_seq, src_seq_name in zip(seqs, seq_names):
                    with open("%s/spec/xin_%s.npy" % (exp_dir, src_seq), "wb") as fnp:
                        np.save(fnp, np.reshape(
                            xin_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])
                    with open("%s/spec/xout_%s.npy" % (exp_dir, src_seq), "wb") as fnp:
                        np.save(fnp, np.reshape(
                            xout_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

        if args.write_neutral:
            # sequence neutralisation
            logging.info("visualizing neutral sequences...")
            neu_by_seq = dict()
            with open("%s/mu2_by_seq.npy" % exp_dir, "rb") as fnp:
                mumu = np.load(fnp)
            for src_seq, src_seq_name in zip(seqs, seq_names):
                del_mu2 = mumu - mu2_by_seq[src_seq]
                src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
                neu_by_seq[src_seq] = _seq_translate(
                    sess, model, src_z1, src_z2, del_mu2)
                with open("%s/spec/neu_%s.npy" % (exp_dir, src_seq), "wb") as fnp:
                    np.save(fnp, np.reshape(
                        neu_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

            plot_x([neu_by_seq[seq] for seq in seqs], seq_names,
                   "%s/img/neutral.png" % exp_dir, False)

        if args.translate:
            # sequence translation
            logging.info("visualizing sequence translation...")
            xtra_by_seq = dict()
            for src_seq, src_seq_name in zip(seqs, seq_names):
                xtra_by_seq[src_seq] = dict()
                src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
                for tar_seq in seqs:
                    del_mu2 = mu2_by_seq[tar_seq] - mu2_by_seq[src_seq]
                    xtra_by_seq[src_seq][tar_seq] = _seq_translate(
                        sess, model, src_z1, src_z2, del_mu2)
                    with open("%s/spec/src_%s_tar_%s.npy" % (exp_dir, src_seq, tar_seq), "wb") as fnp:
                        np.save(fnp, np.reshape(
                            xtra_by_seq[src_seq][tar_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

                # plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names,
                #        "%s/img/%s_tra.png" % (exp_dir, src_seq_name), True)

        if False:
            # random z1 z2
            logging.info("visualizing sequence translation...")
            xtra_by_seq = dict()
            for src_seq, src_seq_name in zip(seqs, seq_names):
                xtra_by_seq[src_seq] = dict()
                src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
                np.random.rand(src_z1.shape[0], src_z1.shape[1])*2-1
                np.random.rand(src_z2.shape[0], src_z2.shape[1])*2-1
                del_mu2 = mu2_by_seq[src_seq] - mu2_by_seq[src_seq]
                xtra_by_seq[src_seq][src_seq] = _seq_translate(
                    sess, model, src_z1, src_z2, del_mu2)
                with open("./spec/src_%s.npy" % (src_seq), "wb") as fnp:
                    np.save(fnp, np.reshape(xtra_by_seq[src_seq][src_seq], (-1, nb_mel)) *
                            mvn_params["std"] + mvn_params["mean"])
            plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names,
                   "./spec/%s_tra.png" % (src_seq_name), True)

            # for tar_seq in seqs:  # TODO: don't loop over all n^2 combinations
            #     del_mu2 = mu2_by_seq[tar_seq] - mu2_by_seq[src_seq]
            #     xtra_by_seq[src_seq][tar_seq] = _seq_translate(
            #         sess, model, src_z1, src_z2, del_mu2)
            #     with open("%s/spec/src_%s_tar_%s.npy" % (exp_dir, src_seq, tar_seq), "wb") as fnp:
            #         np.save(fnp, np.reshape(
            #             xtra_by_seq[src_seq][tar_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

            # plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names,
            #        "%s/img/%s_tra.png" % (exp_dir, src_seq_name), True)

        if args.tsne:
            # tsne z1 and z2
            logging.info("t-SNE analysis on latent variables")
            n = [len(z1_by_seq[seq]) for seq in seqs]
            z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
            z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

            p = 30
            logging.info("perplexity = %s" % p)
            tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
            z1_tsne = _unflatten(tsne.fit_transform(z1), n)
            scatter_plot(z1_tsne, seq_names, "z1_tsne_%03d" % p,
                         "%s/img/z1_tsne_%03d.png" % (exp_dir, p))
            z2_tsne = _unflatten(tsne.fit_transform(z2), n)
            scatter_plot(z2_tsne, seq_names, "z2_tsne_%03d" % p,
                         "%s/img/z2_tsne_%03d.png" % (exp_dir, p))
    logging.info("Finished.")


def tsne_by_label(exp_dir, step, model, iterator_by_seqs, seqs, seq2lab_d):
    seqs = sorted(list(np.random.choice(seqs, 25, replace=False)))
    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        restore_model(sess, saver, "%s/models" % exp_dir, step)

        # infer z1, z2
        z1_by_seq = defaultdict(list)
        z2_by_seq = defaultdict(list)
        for seq in seqs:
            for x, _, _, _, _ in iterator_by_seqs([seq]):
                z2 = z2_mu_fn(sess, model, x)
                z1 = z1_mu_fn(sess, model, x, z2)
                z1_by_seq[seq].append(z1)
                z2_by_seq[seq].append(z2)
            z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
            z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)

        # tsne z1 and z2
        print "t-SNE analysis on latent variables by label"
        n = [len(z1_by_seq[seq]) for seq in seqs]
        z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
        z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

        p = 30
        tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
        z1_tsne_by_seq = dict(zip(seqs, _unflatten(tsne.fit_transform(z1), n)))
        for gen_fac, seq2lab in seq2lab_d.items():
            _labs, _z1 = _join(z1_tsne_by_seq, seq2lab)
            scatter_plot(_z1, _labs, gen_fac,
                         "%s/img/tsne_by_label_z1_%s_%03d.png" % (exp_dir, gen_fac, p))

        z2_tsne_by_seq = dict(zip(seqs, _unflatten(tsne.fit_transform(z2), n)))
        for gen_fac, seq2lab in seq2lab_d.items():
            _labs, _z2 = _join(z2_tsne_by_seq, seq2lab)
            scatter_plot(_z2, _labs, gen_fac,
                         "%s/img/tsne_by_label_z2_%s_%03d.png" % (exp_dir, gen_fac, p))


def shift_reg(exp_dir, step, model, iterator_by_seqs, seqs, seq2lab_d):
    if not os.path.exists("%s/spec/shift-label" % exp_dir):
        os.makedirs("%s/spec/shift-label" % exp_dir)

    seqs = sorted(list(np.random.choice(seqs, 3, replace=False)))
    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        restore_model(sess, saver, "%s/models" % exp_dir, step)

        z1_by_seq = defaultdict(list)
        z2_by_seq = defaultdict(list)
        mu2_by_seq = dict()
        for seq in seqs:
            for x, _, _, _, _ in iterator_by_seqs([seq]):
                z2 = z2_mu_fn(sess, model, x)
                z1 = z1_mu_fn(sess, model, x, z2)
                z1_by_seq[seq].append(z1)
                z2_by_seq[seq].append(z2)
            z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
            z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)
            mu2_by_seq[seq] = map_mu2_z2(model, z2_by_seq[seq])

        with open("./datasets/cgn_per_speaker_afgklno/train/mvn.pkl") as f:
            mvn_params = cPickle.load(f)
        nb_mel = mvn_params["mean"].size

        # For each column
        for label, seq2lab in seq2lab_d.items():
            logging.info(label)
            # Determine avg mu2 by label
            mu2s_by_label = defaultdict(list)
            avg_mu2_by_label = dict()
            for seq in mu2_by_seq:
                mu2 = mu2_by_seq[seq]
                mu2s_by_label[seq2lab[seq]].append(mu2)
            for label, mu2s in mu2s_by_label.items():
                avg_mu2_by_label[label] = np.sum(mu2s, axis=0) / len(mu2s)

            # For each sample:
            for src_seq in seqs:
                src_lbl = seq2lab[src_seq]
                src_z1 = z1_by_seq[src_seq]
                src_z2 = z2_by_seq[src_seq]

                # for each label
                for (tar_lbl, tar_mu2) in avg_mu2_by_label.items():
                    # delta mu2 (shift) is vector from sequence mu2 to avg mu2 of label
                    del_mu2 = avg_mu2_by_label[label] - tar_mu2
                    translation = _seq_translate(
                        sess, model, src_z1, src_z2, del_mu2)
                    with open("%s/spec/shift-label/%s_src_%s_tar_%s.npy" % (exp_dir, src_seq, src_lbl, tar_lbl), "wb") as fnp:
                        np.save(fnp, np.reshape(translation, (-1, nb_mel))
                                * mvn_params["std"] + mvn_params["mean"])


def _unflatten(l_flat, n_l):
    """
    unflatten a list
    """
    l = []
    offset = 0
    for n in n_l:
        l.append(l_flat[offset:offset+n])
        offset += n
    assert(offset == len(l_flat))
    return l


def _join(z_by_seqs, seq2lab):
    d = defaultdict(list)
    for seq, z in z_by_seqs.items():
        d[seq2lab[seq]].append(z)
    for lab in d:
        d[lab] = np.concatenate(d[lab], axis=0)
    return d.keys(), d.values()


def _seq_translate(sess, model, src_z1, src_z2, del_mu2):
    mod_z2 = src_z2 + del_mu2[np.newaxis, ...]
    return x_mu_fn(sess, model, src_z1, mod_z2)

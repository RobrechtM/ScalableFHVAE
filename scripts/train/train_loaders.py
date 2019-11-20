import sys
import numpy as np
from fhvae.datasets.seg_dataset import KaldiSegmentDataset, NumpySegmentDataset

def load_data(name, is_numpy, lab_names = None, talab_names=None):
    root = "./datasets/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    seg_len = 20
    seg_shift = 8
    Dataset = NumpySegmentDataset if is_numpy else KaldiSegmentDataset
    
    if lab_names is not None:
        #lab_specs = [(lab, seqlist_path % lab)]
        lab_specs = [(lab, lab_names[0] % lab) for lab in lab_names[1].split(':')]
    else:
        lab_specs = list()
    if talab_names is not None:
        talab = [(talab_names[0], None, fac) for fac in talab_names[1].split(':')]
    else:
        talab=None

    tr_dset = Dataset(
            "%s/train/feats.scp" % root, "%s/train/len.scp" % root,
            lab_specs= lab_specs, talab_specs=talab,
            min_len=seg_len, preload=False, mvn_path=mvn_path,
            seg_len=seg_len, seg_shift=seg_shift, rand_seg=True)
    dt_dset = Dataset(
            "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root,
            lab_specs= lab_specs, talab_specs=talab,
            min_len=seg_len, preload=False, mvn_path=mvn_path,
            seg_len=seg_len, seg_shift=seg_len, rand_seg=False, copy_from = tr_dset)

    return _load(tr_dset, dt_dset) + (tr_dset,)

def _load(tr_dset, dt_dset):
    def _make_batch(seqs, feats, nsegs, labs, talabs, seq2idx, seq2reg):
        x = feats
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(nsegs)
        c = np.asarray([seq2reg[seq] for seq in seqs])
        b = np.asarray(talabs)
        return x, y, n, b, c

    # def _make_batch(seqs, feats, nsegs, seq2idx):
    #     x = feats
    #     y = np.asarray([seq2idx[seq] for seq in seqs])
    #     n = np.asarray(nsegs)
    #     dummy = np.zeros(len(seqs),dtype=np.float32)
    #     return x, y, n, dummy
    
    #tr_nseqs = len(tr_dset.seqlist)
    tr_nseqs = len(tr_dset.labs_d["spk"].lablist) # the number of speakers, really
    tr_shape = tr_dset.get_shape()
    def tr_iterator(bs=256):
        #seq2idx = dict([(seq, i) for i, seq in enumerate(tr_dset.seqlist)])
        s_seqs = tr_dset.seqlist
        spklist = tr_dset.labs_d["spk"].lablist
        seq2lab = tr_dset.labs_d["spk"].seq2lab
        seq2idx = dict([(seq, spklist.index(seq2lab[seq])) for seq in s_seqs])

        lab_names = tr_dset.labs_d.keys()
        lab_names.remove("spk")

        talab_names = tr_dset.talabseqs_d.keys()
        ii = list()
        for k in s_seqs:
            itm = [ tr_dset.labs_d[name].lablist.index(tr_dset.labs_d[name].seq2lab[k]) for name in lab_names]
            ii.append(np.asarray(itm))
        seq2regidx = dict(zip(s_seqs,ii))
        _iterator = tr_dset.iterator(bs, seg_shuffle=True, seg_rem=False, \
                                     seqs=s_seqs, lab_names=lab_names, talab_names=talab_names)
        for seqs, feats, nsegs, labs, talabs in _iterator:
            yield _make_batch(seqs, feats, nsegs, labs, talabs, seq2idx, seq2regidx)
    
    # def dt_iterator(bs=2048):
    #     # THIS IS INCONSISTENT WITH TRAINING !!!
    #     seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
    #     _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True)
    #     for seqs, feats, nsegs, _, _ in _iterator:
    #         yield _make_batch(seqs, feats, nsegs, seq2idx)

    def dt_iterator(bs=2048):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        lab_names = dt_dset.labs_d.keys()
        lab_names.remove("spk")
        talab_names = dt_dset.talabseqs_d.keys()
        ii = list()
        for k in dt_dset.seqlist:
            itm = [ dt_dset.labs_d[name].lablist.index(dt_dset.labs_d[name].seq2lab[k]) for name in lab_names]
            ii.append(np.asarray(itm))
        seq2regidx = dict(zip(dt_dset.seqlist,ii))
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True, lab_names=lab_names, talab_names=talab_names)
        for seqs, feats, nsegs, labs , talabs in _iterator:
            yield _make_batch(seqs, feats, nsegs, labs, talabs, seq2idx, seq2regidx)

    return tr_nseqs, tr_shape, tr_iterator, dt_iterator

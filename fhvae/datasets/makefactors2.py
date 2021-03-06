import os

set = "train"
dir = "/users/spraak/hvanhamm/repos/ScalableFHVAE/"
indir = dir + "datasets/cgn_per_speaker_afgklno/" + set + "/"
outdir = dir + "misc/"
with open(os.path.join(outdir,"speakers_regions.csv")) as f:
    speaker=[l.rstrip('\n').split(';')[0].lower() for l in f]
    f.seek(0)
    gender=[l.rstrip('\n').split(';')[1] for l in f]
    f.seek(0)
    reg1=[l.rstrip('\n').split(';')[2] for l in f]
    f.seek(0)
    reg2=[l.rstrip('\n').split(';')[3] for l in f]
    f.seek(0)
    reg3=[l.rstrip('\n').split(';')[4] for l in f]
    f.seek(0)
    size=[l.rstrip('\n').rstrip('\r').split(';')[5] for l in f]
    f.close()



#with open("/esat/spchtemp/scratch/hvanhamm/fhvae_cgn/cgn_np_fbank/train/len.scp") as f:
with open(os.path.join(indir,"len.scp")) as f:
    #lines=[[l.split()[0],l.split()[0].split('_')[0]] for l in f]
    lines=[l.split()[0] for l in f]
    f.close()

del lines[0]
with open(os.path.join(outdir,"cgn_per_spk_afgklno_%s.fac" % set),"w") as f:
    f.write("#seq speaker comp reg1 reg2 reg3 gender size\n" )
    for l in lines:
        ls = l.split("_")
        k = speaker.index(ls[0])
        f.write("%s %s %s %s %s %s %s %s\n" % (l,ls[0],l[-1],reg1[k],reg2[k],reg3[k],gender[k],size[k]))
    f.close()
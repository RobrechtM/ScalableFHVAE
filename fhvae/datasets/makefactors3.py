import os

set = "test"
dir = "/users/spraak/hvanhamm/repos/ScalableFHVAE/"
indir = dir + "datasets/cogen_np_fbank/" + set + "/"
outdir = dir + "misc/"

infid = open(os.path.join(indir,"len.scp"))
uttlist = [l.split()[0] for l in infid]
infid.close()

xlatfid = open(os.path.join(outdir,"cogen_spk.fac"))
header = xlatfid.readline()
xlat = dict()
for l in xlatfid:
    lspl = l.split(" ",1)
    lspl[1] = lspl[1].replace("M","sex1")
    lspl[1] = lspl[1].replace("V","sex2")
    lspl[1] = lspl[1].replace("A","regV1")
    lspl[1] = lspl[1].replace("B","regV1")
    lspl[1] = lspl[1].replace("O","regV2")
    lspl[1] = lspl[1].replace("W","regV3")
    lspl[1] = lspl[1].replace("L","regV4")
    xlat[lspl[0]]=lspl[1]
xlatfid.close()

outfid = open(os.path.join(outdir,"cogen_%s.fac" % set),"w")
outfid.write(header) # already includes new line
for utt in uttlist:
    spk,_ = utt.split("_")
    outfid.write(utt + " " + xlat[spk] ) # already includes new line
outfid.close()


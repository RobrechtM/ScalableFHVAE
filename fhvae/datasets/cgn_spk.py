import os

dir = "/users/spraak/hvanhamm/repos/ScalableFHVAE/misc"

with open(os.path.join(dir,"recordings.csv")) as f:
    lines=[l.rstrip('\n').split(':')[:2] for l in f]
    #print lines[:10]
    fname2spk = dict(lines)
    f.seek(0)
    lines=[l.rstrip('\n').split(':')[0:3:2] for l in f]
    #print lines[:10]
    fname2role = dict(lines)
    f.close()

#with open("/esat/spchtemp/scratch/hvanhamm/fhvae_cgn/cgn_np_fbank/train/len.scp") as f:
with open(os.path.join(dir,"cgn_o_test.txt")) as f:
    #lines=[[l.split()[0],l.split()[0].split('_')[0]] for l in f]
    lines=[l.split()[0] for l in f]
    print lines[:10]
    f.close()

with open(os.path.join(dir,"cgn_o_test2.txt"),"w") as f:
    for l in lines:
        lsplit = l.split('_')
        k = lsplit[0]
        comp = lsplit[-2]
        #if comp is not 'o':
        #   print "%s %s %s %s\n" % (l,comp,fname2spk[k],fname2role[k])
        f.write("%s %s\n" % (l,fname2spk[k]))
    f.close()
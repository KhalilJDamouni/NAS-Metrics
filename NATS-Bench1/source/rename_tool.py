import glob
import sys
import os

pickles = glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/NATS-tss-v1_0-3ffb9-full/*')
for pickle in pickles:
    try:
        print(str(int((pickle.split(os.path.sep)[-1]).split('.')[0])))
        os.rename(pickle,("NATS-Bench1/fake_torch_dir/NATS-tss-v1_0-3ffb9-full/"+str(int((pickle.split(os.path.sep)[-1]).split('.')[0]))+".pickle.pbz2"))
    except:
        print("skipping")

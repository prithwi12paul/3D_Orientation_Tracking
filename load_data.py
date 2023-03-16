import pickle
import sys
import time
import numpy as np
import jax.numpy as jnp
import transforms3d as t
import matplotlib.pyplot as plt

jnp.set_printoptions(threshold=np.inf)

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

dataset="3"
cfile = "/home/prithwiraj/Desktop/ECE276A_PR1/ECE276A_PR1/data/cam" + dataset + ".p"
#ifile = "../data/imu/imuRaw" + dataset + ".p"
ifile = "/home/prithwiraj/Desktop/ECE276A_PR1/ECE276A_PR1/data/imu/imuRaw" + dataset + ".p"
#vfile = "/home/prithwiraj/Desktop/ECE276A_PR1/ECE276A_PR1/data/vicon/viconRot" + dataset + ".p"

ts = tic()
#camd = read_data(cfile)
imud = read_data(ifile)
#vicd = read_data(vfile)
#print(imud)
#a=jnp.float_(vicd['rots'])
#print(a.shape[2])
toc(ts,"Data import")







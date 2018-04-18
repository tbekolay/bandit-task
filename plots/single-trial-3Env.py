import math
import zipfile
import numpy
import random
from copy import copy
from matplotlib import rc,use

rc('text',usetex=True)
rc('font',family='serif')
use('Agg')

import pylab

save = True
arm = True
env_file = 'data/3Env.zip'
arm_file = 'data/3Arm.zip'

##################
#Functions called#
##################
def sample(data):
    for _ in data:
        yield random.choice(data)
def bootstrapci(data,func,n,p):
    index = int(n*(1-p)/2)
    r = [func(list(sample(data))) for _ in range(n)]
    r.sort()
    return r[index],r[-index]
def read_zip(zip_path,skip_rows=3):
    z_trial = None
    z_choice = None
    z_rewarded = None
    
    try:
        zf = zipfile.ZipFile(zip_path, 'r')
        for f in zf.namelist():
            csv_f = zf.open(f)
            rows = csv_f.readlines()
            
            if z_trial is None:
                z_trial = range(1,len(rows)-skip_rows+1)
                z_choice = [[] for _ in range(len(rows)-skip_rows)]
                z_rewarded = [[] for _ in range(len(rows)-skip_rows)]
            
            for ix,row in enumerate(rows):
                if ix >= skip_rows:
                    row_l = row[:-1].split(', ')
                    z_choice[ix-skip_rows].append(float(row_l[1]))
                    z_rewarded[ix-skip_rows].append(float(row_l[2]))
        zf.close()
    except:
        raise Exception("Error reading %s" % zip_path)
    
    return (z_trial,z_choice,z_rewarded)
def moving_average(data,start=0.5,window=10):
    extended_data = numpy.hstack([[start] * (window-1), data])
    weightings = numpy.repeat(1.0, window) / window
    return numpy.convolve(extended_data, weightings)[window-1:-(window-1)]
def square_error(d1,d2):
    se = []
    for i in range(len(d1)):
        se.append((d1[i]-d2[i])**2)
    return se

#################
#Data processing#
#################
def get_data(z_file):
  _,choice,rewarded = read_zip(z_file)
  choice = numpy.array(choice)
  rewarded = numpy.array(rewarded)

  left = copy(choice)
  left = left - 1
  left[left >= 0] = 0
  left[left < 0] = 1

  center = copy(choice)
  center[center == 2] = 0

  right = copy(choice)
  right[right == 1] = 0
  right[right == 2] = 1

  def process(a):
    mean = []; cil = []; cih = []
    for x in a:
        mean.append(numpy.mean(x))
        l,h = bootstrapci(x,numpy.mean,1000,0.95)
        cil.append(l)
        cih.append(h)
    return mean,cil,cih

  lm,lil,lih = process(left)
  cm,cil,cih = process(center)
  rm,ril,rih = process(right)
  return lm,lil,lih,cm,cil,cih,rm,ril,rih

env_lm,env_lil,env_lih,env_cm,env_cil,env_cih,env_rm,env_ril,env_rih = get_data(env_file)
if arm:
  arm_lm,arm_lil,arm_lih,arm_cm,arm_cil,arm_cih,arm_rm,arm_ril,arm_rih = get_data(arm_file)

trial = range(1,len(env_lm)+1)

##########
#Plotting#
##########
pylab.figure(1,figsize=(15,7))
pylab.axes((0.06,0.08,0.89,0.84))
pylab.title(r"L:0.21 C:0.21 R:0.63 \hspace{0.3em} L:0.21 C:0.63 R:0.21 \hspace{0.3em} L:0.63 C:0.21 R:0.21 \hspace{0.3em} L:0.12 C:0.12 R:0.72 \hspace{0.3em} L:0.21 C:0.72 R:0.12 \hspace{0.3em} L:0.72 C:0.12 R:0.12",fontsize=16)
pylab.fill_between(trial,y1=env_lil,y2=env_lih,color='k',alpha=0.4)
pylab.plot(trial,env_lm,linewidth=2,linestyle='-',color='k',label='Left')
pylab.fill_between(trial,y1=env_cil,y2=env_cih,color='k',alpha=0.4)
pylab.plot(trial,env_cm,linewidth=2,linestyle='--',color='k',label='Center')
pylab.fill_between(trial,y1=env_ril,y2=env_rih,color='k',alpha=0.4)
pylab.plot(trial,env_rm,linewidth=2,linestyle=':',color='k',label='Right')
if arm:
  pylab.fill_between(trial,y1=arm_lil,y2=arm_lih,color='0.6',alpha=0.4)
  pylab.plot(trial,arm_lm,linewidth=2,linestyle='-',color='0.6',label='Same environment')
  pylab.fill_between(trial,y1=arm_cil,y2=arm_cih,color='0.6',alpha=0.4)
  pylab.plot(trial,arm_cm,linewidth=2,linestyle='--',color='0.6')
  pylab.fill_between(trial,y1=arm_ril,y2=arm_rih,color='0.6',alpha=0.4)
  pylab.plot(trial,arm_rm,linewidth=2,linestyle=':',color='0.6')
pylab.axhline(0.5,linestyle='--',linewidth=1,color='k')
pylab.text(9,0.52,'Environment 1',fontsize=16)
pylab.axvline(40,linestyle='--',linewidth=1,color='0.5')
pylab.text(49,0.52,'Environment 2',fontsize=16)
pylab.axvline(80,linestyle='--',linewidth=1,color='0.5')
pylab.text(89,0.52,'Environment 3',fontsize=16)
pylab.axvline(120,linestyle='--',linewidth=1,color='0.5')
pylab.text(129,0.52,'Environment 1',fontsize=16)
pylab.axvline(160,linestyle='--',linewidth=1,color='0.5')
pylab.text(169,0.52,'Environment 2',fontsize=16)
pylab.axvline(200,linestyle='--',linewidth=1,color='0.5')
pylab.text(209,0.52,'Environment 3',fontsize=16)
pylab.legend(loc=2)
pylab.axis([min(trial),max(trial),-0.02,1.02])
pylab.xlabel(r"Trial number",fontsize=20)
pylab.ylabel(r"Proportion of trials moving that direction",fontsize=20)
pylab.xticks((1,40,80,120,160,200,240))

if save:
  pylab.savefig('single-trial-3Env.pdf')
else:
  pylab.show()


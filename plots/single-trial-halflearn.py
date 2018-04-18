import math
import zipfile
import numpy
import pylab
import random

from matplotlib import rc

#rc('text',usetex=True)
#rc('font',family='serif')
rc('legend',fontsize='small')

save = True
z_file = 'data/halflearn.zip'

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
                    z_choice[ix-skip_rows].append(1.0 - float(row_l[1])) # 1 - value to match paper
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
######
#Data#
######
trial = range(1,161)

#################
#Data processing#
#################
_,choice,rewarded = read_zip(z_file)
choice = numpy.array(choice)
rewarded = numpy.array(rewarded)

min_mse = 1e5

mean = []; cil = []; cih = []
for x in choice:
    mean.append(numpy.mean(x))
    l,h = bootstrapci(x,numpy.mean,1000,0.95)
    cil.append(l)
    cih.append(h)

##########
#Plotting#
##########
pylab.figure(1,figsize=(8,5))
#pylab.axes((0.06,0.08,0.89,0.84))
fsize = 12
pylab.text(20,1.04, "E1  L:0.12 R:0.72",fontsize=fsize,ha='center')
pylab.text(60,1.04, "E2  L:0.72 R:0.12",fontsize=fsize,ha='center')
pylab.text(100,1.03, "E1  No learning",fontsize=fsize,ha='center')
pylab.text(140,1.03, "E2  No learning",fontsize=fsize,ha='center')
pylab.fill_between(trial,y1=cil,y2=cih,color='k',alpha=0.5)
pylab.plot(trial,mean,color='k',linewidth=2)
pylab.axhline(0.5,linestyle='--',linewidth=1,color='k')
pylab.axvline(40,linestyle='--',linewidth=1,color='0.5')
pylab.axvline(80,linestyle='--',linewidth=1,color='0.5')
pylab.axvline(120,linestyle='--',linewidth=1,color='0.5')
pylab.axis([min(trial),max(trial),-0.02,1.02])
pylab.xlabel(r"trial number",fontsize=fsize)
pylab.ylabel(r"probability of moving left",fontsize=fsize)
pylab.xticks((40,80,120,160))

if save:
  pylab.savefig('../2011-BehaviourLearning-Fig11-Stop40.pdf')
  pylab.savefig('../2011-BehaviourLearning-Fig11-Stop40.png',dpi=600)
else:
  pylab.show()


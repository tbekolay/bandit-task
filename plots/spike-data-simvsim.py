import numpy
import random
import pylab
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import rc

rc('text',usetex=True)
rc('font',family='serif')

twoarm_file = 'data/vStrSpikes-2arm.csv'
threearm_file = 'data/vStrSpikes-3arm.csv'

save = True

##################
#Functions called#
##################
def read_file(csv_file):
    z_spikes = None
    
    try:
        f = open(csv_file, 'r')
        rows = f.readlines()
        z_spikes = [[] for _ in range(len(rows))]
        
        for i,row in enumerate(rows):
            if len(row) > 2:
                z_spikes[i] = map(float,row[:-1].split(', '))
        f.close()
    except:
        raise Exception("Error reading %s" % csv_file)
    
    return z_spikes

######
#Data#
######
def randindexes(data,n):
    random.shuffle(data)
    r = data[:n]
    r.sort()
    return r

twoarm_spikes = numpy.array(read_file(twoarm_file))
threearm_spikes = numpy.array(read_file(threearm_file))
ix = randindexes(range(len(threearm_spikes)),len(twoarm_spikes))
print ix
threearm_spikes = threearm_spikes[ix]
#################
#Data processing#
#################
my_start = 0.6
my_end = 1.1
for i in range(len(twoarm_spikes)):
    twoarm_spikes[i] = filter(lambda t:t >= my_start and t < my_end,twoarm_spikes[i])
    twoarm_spikes[i] = map(lambda t:t-my_start,twoarm_spikes[i])
for i in range(len(threearm_spikes)):
    threearm_spikes[i] = filter(lambda t:t >= my_start and t < my_end,threearm_spikes[i])
    threearm_spikes[i] = map(lambda t:t-my_start,threearm_spikes[i])
my_approach = 0.2
my_reward = 0.3
my_delay = 0.4
my_end = 0.5

bins = 500
x = numpy.linspace(0.0,my_end,num=bins)
bin_size = my_end / bins

sigma = 15.0

twoarm_bins = [0.0]*bins
for i in range(len(twoarm_spikes)):
    for s in twoarm_spikes[i]:
        twoarm_bins[int(s/bin_size)-1] += 1
twoarm_filtered = gaussian_filter1d(twoarm_bins,sigma)

threearm_bins = [0.0]*bins
for i in range(len(threearm_spikes)):
    for s in threearm_spikes[i]:
        threearm_bins[int(s/bin_size)-1] += 1
threearm_filtered = gaussian_filter1d(threearm_bins,sigma)

dx = x[1]-x[0]
print "sigma = "+str(sigma*dx)

##########
#Plotting#
##########
pylab.figure(1,figsize=(8,6))
fontsize=14

bottom_axes = (0.08,0.07,0.88,0.29)
pylab.axes((bottom_axes[0],bottom_axes[1]+bottom_axes[3]*2,bottom_axes[2],bottom_axes[3]))
pylab.plot(x,twoarm_filtered,linewidth=3,color='0.5',label='2-arm simulation')
pylab.plot(x,threearm_filtered,linewidth=3,color='k',label='3-arm simulation')
pylab.axvline(my_approach,linestyle='--',linewidth=1,color='k')
pylab.axvline(my_reward,linestyle='--',linewidth=1,color='k')
pylab.axvline(my_delay,linestyle='--',linewidth=1,color='k')
pylab.legend(loc=2)
pylab.title(r"Delay \hspace{10em} Approach \hspace{2.5em} Reward \hspace{3em} Delay",fontsize=fontsize)
pylab.ylabel(r"Filtered \# of spikes",fontsize=fontsize)
pylab.xticks(())

pylab.axes((bottom_axes[0],bottom_axes[1]+bottom_axes[3],bottom_axes[2],bottom_axes[3]))
for i,s in enumerate(twoarm_spikes):
    pylab.plot(s,[i]*len(s),linestyle='None',markerfacecolor='0.5',marker='o',markersize=3.0)
pylab.axvline(my_approach,linestyle='--',linewidth=1,color='k')
pylab.axvline(my_reward,linestyle='--',linewidth=1,color='k')
pylab.axvline(my_delay,linestyle='--',linewidth=1,color='k')
pylab.ylabel(r"Neuron number",fontsize=fontsize)
pylab.xticks(())

pylab.axes(bottom_axes)
for i,s in enumerate(threearm_spikes):
    pylab.plot(s,[i]*len(s),linestyle='None',markerfacecolor='k',marker='o',markersize=3.0)
pylab.axvline(my_approach,linestyle='--',linewidth=1,color='k')
pylab.axvline(my_reward,linestyle='--',linewidth=1,color='k')
pylab.axvline(my_delay,linestyle='--',linewidth=1,color='k')
pylab.ylabel(r"Neuron number",fontsize=fontsize)
pylab.xlabel(r"Time (seconds)",fontsize=fontsize)

if save:
  pylab.savefig('spike-data-simvsim.pdf')
else:
  pylab.show()


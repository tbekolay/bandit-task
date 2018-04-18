import numpy
import itertools
import png

r = png.Reader(filename="st.png")
w,h,p,_ = r.asFloat()
pixels = numpy.vstack(itertools.imap(numpy.float32, p))

spike_w = 7
spike_h = 20
threshold = 0.8

rows = range(spike_h/2, h, spike_h)
cols = range(spike_w/2, w, spike_w)
spikes = [[] for _ in range(len(rows))]

for i,r in enumerate(rows):
    for j,c in enumerate(cols):
        if pixels[r,c] < threshold:
            spikes[i].append(j)

fname = 'kim_spikes.csv'
f = open(fname,'w')
for st in spikes:
    f.write("%s\n" % ', '.join(map(str,st)))


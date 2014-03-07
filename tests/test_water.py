import matplotlib.pyplot as plt
import numpy as np

from pyrdf.rdf import rdf

pairs = None
trajectory_file = 'water.lammpstrj'
r, g_r = rdf(trajectory_file, pairs=pairs,
        n_bins=200, max_frames=np.inf, verbose=True)

fig = plt.figure()
plt.plot(r, g_r, 'ro-')
plt.plot([0, 8], [1, 1], 'k-')
plt.xlabel(u'r (\u00c5)')
plt.ylabel('g(r)')

if pairs:
    name = '{0}_{1}'.format(pairs[0], pairs[1])
else:
    name = 'all_all'
fig.savefig('g_r-{0}.pdf'.format(name), bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np

from pyrdf.rdf import rdf

types = {1: 'O', 2: 'H', 3: 'H'}
#pairs = [None]
pairs = [[1, 2]]

for pair in pairs:
    trajectory_file = 'water_216.lammpstrj'
    r, g_r = rdf(trajectory_file, pairs=pair, r_range=(0.0, 8.0), n_bins=200,
            max_frames=np.inf, opencl=True, verbose=True)

    fig = plt.figure()
    plt.plot(r, g_r, 'bo-')
    plt.plot([0, 8], [1, 1], 'k-')
    plt.xlabel(u'r (\u00c5)', fontsize=18)
    plt.ylabel('g(r)', fontsize=18)

    if pair:
        name = '{0}-{1}'.format(types[pair[0]], types[pair[1]])
    else:
        name = 'all-all'
    fig_name = 'img/{0}_pyopencl.pdf'.format(name)
    fig.savefig(fig_name, bbox_inches='tight')
    print "Wrote: {0}".format(fig_name)

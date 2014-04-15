import matplotlib.pyplot as plt
import numpy as np

from pyrdf.rdf import rdf

types = {1: 'O', 2: 'H', 3: 'H'}

pairs = [None, [1, 1], [1, 2]]
pairs = [[1, 1]]
for pair in pairs:
    #trajectory_file = 'water_216.lammpstrj'
    trajectory_file = 'water_267098.lammpstrj'
    r, g_r = rdf(trajectory_file, pairs=pair, r_range=(0.0, 8.0), n_bins=200,
            max_frames=1, opencl=False, verbose=True)

    fig = plt.figure()
    plt.plot(r, g_r, 'bo-')
    plt.plot([0, 8], [1, 1], 'k-')
    plt.xlabel(u'r (\u00c5)', fontsize=18)
    plt.ylabel('g(r)', fontsize=18)

    if pair:
        name = '{0}-{1}'.format(types[pair[0]], types[pair[1]])
    else:
        name = 'all-all'
    fig.savefig('img/{0}.pdf'.format(name), bbox_inches='tight')

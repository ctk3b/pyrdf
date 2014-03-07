import matplotlib.pyplot as plt

from pyrdf.rdf import rdf

trajectory_file = 'water.lammpstrj'
r, g_r = rdf(trajectory_file, n_bins=200, max_frames=50, verbose=True)

plt.plot(r, g_r, 'ro-')
plt.plot([1, 8], [1, 1], 'k-')
plt.xlabel(u'r (\u212B)')
plt.ylabel('g(r)')
plt.show()

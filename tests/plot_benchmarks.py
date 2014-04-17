import matplotlib.pyplot as plt
import numpy as np
import pdb

n_water = 3 * np.array([216, 4142, 33226], dtype=int)
numpy = np.array([0.255, 33.3])
cython = np.array([0.0161, 4.23])
pyopencl = np.array([0.0104, 1.42, 90.2])

fig = plt.figure()
ax = plt.subplot(111)

ax.set_xlabel('System size (# atoms)')
ax.set_ylabel('Time per frame (s)')
ax.set_ylim([5e-3, 1e2])

ax.loglog(n_water, numpy, 'o--', alpha = 0.7, label='NumPy')
ax.loglog(n_water, cython, 'D--', alpha = 0.7, label='Cython')
ax.loglog(n_water, pyopencl, 's--', alpha = 0.7, label='PyOpenCL (CPU, 4 threads)')

for i in range(2):
    ax.text(0.6 * n_water[i], cython[i] * 1.2, "{0:.1f}x".format(cython[i] / pyopencl[i]))
    ax.text(0.6 * n_water[i], numpy[i] * 1.2, "{0:.1f}x".format(numpy[i] / pyopencl[i]))

plt.legend(numpoints=1, loc='upper left')
fig_name = 'img/benchmarks.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print "Wrote: {0}".format(fig_name)



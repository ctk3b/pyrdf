#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Various implementations of radial distribution function calculations.
"""

import pdb

import numpy as np

from pyrdf.mdio import read_frame_lammpstrj, distance_pbc


def rdf(file_name, pairs=None, n_bins=100, max_frames=1, verbose=False):
    """Calculate radial distribution functions.

    :param file_name: name of trajectory file.
    :type file_name: str.
    :param pairs: pairs of atomtypes to consider.
    :type pairs: none, list
    :param n_bins: number of bins.
    :type n_bins: int.
    :returns: r -- radii values corresponding to bins.
    :returns: g_r -- values at r.
    :rtype: tuple of numpy arrays.
    """
    r_range = (0.0, 8.0)
    g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
    g_r[0] = 0
    g_r = g_r.astype(np.float64)
    n_frames = 0
    rho = 0

    with open(file_name, 'r') as trj:
        while n_frames < max_frames:
            try:
                xyz, types, _, box = read_frame_lammpstrj(trj)
            except ValueError:
                print("Reached end of '{0}' or "
                    "file contains unexpected line.".format(file_name))
                break
            if verbose:
                print "Read " + str(n_frames)
            n_frames += 1
            box_lengths = [dim_max - dim_min for dim_min, dim_max in box]
            box_lengths = np.array(box_lengths)
            box_volume = np.prod(box_lengths)

            # all-all
            if not pairs:
                for i, xyz_i in enumerate(xyz):
                    xyz_j = np.vstack([xyz[:i], xyz[i+1:]])
                    d = distance_pbc(xyz_i, xyz_j, box_lengths)
                    temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
                    g_r += temp_g_r

            # type_i-type_i
            elif pairs[0] == pairs[1]:
                xyz_0 = xyz[types == pairs[0]]
                for i, xyz_i in enumerate(xyz_0):
                    xyz_j = np.vstack([xyz_0[:i], xyz_0[i+1:]])
                    d = distance_pbc(xyz_i, xyz_j, box_lengths)
                    temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
                    g_r += temp_g_r

            # type_i-type_j
            else:
                for i, xyz_i in enumerate(xyz[types == [pairs[0]]]):
                    xyz_j = xyz[types == pairs[1]]
                    d = distance_pbc(xyz_i, xyz_j, box_lengths)
                    temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
                    g_r += temp_g_r

            rho += (i + 1) / box_volume

        r = 0.5 * (edges[1:] + edges[:-1])
        V = 4./3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        norm = rho * i
        g_r /= norm * V
    return r, g_r

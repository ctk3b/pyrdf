"""Various implementations of radial distribution function calculations."""

import os
import pdb

import numpy as np
import pyopencl as cl

from pyrdf.mdio import read_frame_lammpstrj, distance_pbc


def rdf(file_name, pairs=None, r_range=np.array([0.0, 8.0], dtype=np.float32),
        n_bins=np.uint32(100), max_frames=1, opencl=True, local_size= None,
        verbose=False):
    """Calculate radial distribution functions.

    Args:
        file_name (str): name of trajectory file.
        pairs (None or list): pairs of atomtypes to consider.
        r_range (np.ndarray): min and max radii.
        n_bins (np.uint32): number of bins.
        max_frames (int): maximum number of frames to read.
    Returns:
        r (np.ndarray): radii values corresponding to bins.
        g_r (np.ndarray): values at r.
    """
    n_bins = np.uint32(n_bins)
    r_range = np.array(r_range)
    r_range = r_range.astype(np.float32)
    g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
    g_r[0] = 0
    g_r = g_r.astype(np.uint32)
    temp_g_r = np.zeros_like(g_r)
    n_frames = 0
    rho = 0

    if opencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        kernel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'rdf.cl')
        with open(kernel_path, 'r') as f:
            source = "".join(f.readlines())
        program = cl.Program(ctx, source).build()

    with open(file_name, 'r') as trj:
        while n_frames < max_frames:
            try:
                xyz, types, _, box = read_frame_lammpstrj(trj)
            except ValueError:
                if verbose:
                    print("Reached end of '{0}' or "
                        "file contains unexpected line.".format(file_name))
                break
            if verbose:
                print "Read " + str(n_frames)
            n_frames += 1
            n_atoms = np.int64(xyz.shape[0])
            box_lengths = [dim_max - dim_min for dim_min, dim_max in box]
            box_lengths = np.array(box_lengths)
            box_volume = np.prod(box_lengths)

            if opencl:
                if not pairs:
                    pass
                elif pairs[0] == pairs[1]:
                    xyz = xyz[types == pairs[0]]
                    n_atoms = np.int64(xyz.shape[0])
                else:
                    raise Exception("Not yet implemented!")
                i = n_atoms - 1
                global_size = (n_atoms,)

                xyz_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=xyz)
                box_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=box_lengths)
                result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, temp_g_r.nbytes)

                kernel = program.rdf(queue, global_size, local_size,
                        n_atoms, n_bins, r_range[0], r_range[1],
                        xyz_buf, box_buf, result_buf)

                cl.enqueue_read_buffer(queue, result_buf, temp_g_r).wait()
                g_r += temp_g_r

            else:
                # all-all
                if not pairs:
                    for i, xyz_i in enumerate(xyz):
                        xyz_j = np.vstack([xyz[:i], xyz[i+1:]])
                        d = distance_pbc(xyz_i, xyz_j, box_lengths)
                        temp_g_r, _ = np.histogram(d, n_bins, r_range)
                        g_r += temp_g_r

                # type_i-type_i
                elif pairs[0] == pairs[1]:
                    xyz_0 = xyz[types == pairs[0]]
                    for i, xyz_i in enumerate(xyz_0):
                        xyz_j = np.vstack([xyz_0[:i], xyz_0[i+1:]])
                        d = distance_pbc(xyz_i, xyz_j, box_lengths)
                        temp_g_r, _ = np.histogram(d, n_bins, r_range)
                        g_r += temp_g_r

                # type_i-type_j
                else:
                    for i, xyz_i in enumerate(xyz[types == [pairs[0]]]):
                        xyz_j = xyz[types == pairs[1]]
                        d = distance_pbc(xyz_i, xyz_j, box_lengths)
                        temp_g_r, _ = np.histogram(d, n_bins, r_range)
                        g_r += temp_g_r

            rho += (i + 1) / box_volume

    # normalization
    r = 0.5 * (edges[1:] + edges[:-1])
    V = 4./3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = rho * i
    g_r = g_r.astype(np.float32)  # from uint32
    g_r /= norm * V

    return r, g_r

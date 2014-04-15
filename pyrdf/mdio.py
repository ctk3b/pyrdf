"""Utilities to read and analyze molecular dynamics trajectories."""

import numpy as np

def read_frame_lammpstrj(trj):
    """Load a frame from a LAMMPS dump file.

    Args:
        trj (file): LAMMPS dump file of format 'ID type x y z'
    Returns:
        xyz (np.ndarray): coordinates of all atoms
        types (np.ndarray): types of all atoms
        step (int): current timestep
        box (np.ndarray): box dimensions
    """
    box = np.empty(shape=(3, 2), dtype=np.float32)

    # --- begin header ---
    trj.readline()  # text
    step = int(trj.readline())  # timestep
    trj.readline()  # text
    n_atoms = int(trj.readline())  # num atoms
    trj.readline()  # text
    box[0] = trj.readline().split()  # x-dim of box
    box[1] = trj.readline().split()  # y-dim of box
    box[2] = trj.readline().split()  # z-dim of box
    trj.readline()  # text
    # --- end header ---

    xyz = np.empty(shape=(n_atoms, 3), dtype=np.float32)
    xyz[:] = np.NAN
    types = np.empty(shape=(n_atoms), dtype=np.int32)

    # --- begin body ---
    for i in range(n_atoms):
        temp = trj.readline().split()
        a_ID = int(temp[0])  # atom ID
        types[a_ID - 1] = int(temp[1])  # atom type
        xyz[a_ID - 1] = map(float, temp[2:5])  # coordinates
    # --- end body ---

    return xyz, types, step, box

def distance_pbc(x0, x1, dimensions):
    """Vectorized distance calculation considering minimum image."""
    d = np.abs(x0 - x1)
    d = np.where(d > 0.5 * dimensions, dimensions - d, d)
    return np.sqrt((d ** 2).sum(axis=-1))

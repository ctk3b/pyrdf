#define ANINT(x) ((x >= 0.5) ? (1.0) : (x < -0.5) ? (-1.0) : (0.0))

__kernel void rdf(const int n_atoms,
                const unsigned int n_bins,
                float r_min,
                float r_max,
                __global float* xyz,
                __global float* box,
                __global unsigned int* g_r)
{
    // assign the kernel a central atom
    unsigned int center = get_global_id(0);
    float center_coord[3] = {xyz[3 * center],
                             xyz[3 * center + 1],
                             xyz[3 * center + 2]};

    unsigned int atom;
    unsigned int i;
    unsigned int added = 0;
    float coord[3];
    float distances[n_atoms-1];
    for (atom = 0; atom < n_atoms; atom++) {
        // for every other atom
        if (atom != center) {
            coord[0] = xyz[3 * atom];
            coord[1] = xyz[3 * atom + 1];
            coord[2] = xyz[3 * atom + 2];

            // calculate distance to nearest image of atom
            float dist = 0;
            for (i = 0; i < 3; i++) {
                float temp = center_coord[i] - coord[i];
                temp -= box[i] * ANINT(temp / box[i]);
                dist += temp * temp;
            }
            // NOTE: maybe do sqrt on host in DP; return g(r^2) from here
            distances[added] = sqrt(dist);
            added++;
        }
    }

    // do local distance histogram
    float binsize = (r_max - r_min) / n_bins;
    unsigned int dist_histogram[n_bins];
    for (i = 0; i < n_bins; i++) {
        dist_histogram[i] = 0;
    }

    for (i = 0; i < n_atoms-1; i ++) {
        unsigned int bin = floor((distances[i] - r_min) / binsize);
        if (center == 0 && bin == 0) {
            printf("Atom %d to atom %d has distance: %.2f.\n", center, i, distances[i]);
        }
        dist_histogram[bin]++;
    }

    // atomic add to global histogram
    for (i = 0; i < n_bins; i++) {
        atomic_add(&g_r[i], dist_histogram[i]);
    }
}


import os
import time
import socket

import numpy as np
from mpi4py import MPI

from sbi_bmode import so_utils

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
host = socket.gethostname()


def get_rss_mb():
    """Resident memory of this process, in MB (Linux only)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1


if __name__ == "__main__":
    obsmat_dir = os.environ.get("TEST_OBSMAT_DIR", "/u/bing/so-data/mss2")
    freq = os.environ.get("TEST_OBSMAT_FREQ", "f090")
    tag = os.environ.get("TEST_OBSMAT_TAG", "RC1.r01")
    release = os.environ.get("TEST_OBSMAT_RELEASE", "mss2")

    rss_before = get_rss_mb()
    t0 = time.time()

    obsmats = so_utils.load_obs_matrix_mpi_shared(
        freqs=[freq], obsmat_dir=obsmat_dir, comm=comm, tag=tag, release=release,
    )
    obsmat = obsmats[freq]

    comm.Barrier()
    dt = time.time() - t0
    rss_after = get_rss_mb()

    # Node-local rank (to see who actually touched disk).
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()

    # Correctness check: hash the data array and make sure every rank agrees.
    local_checksum = np.float64(np.sum(obsmat.matrix.data[:1000].astype(np.float64)))
    all_checksums = comm.allgather(local_checksum)
    checksums_match = all(abs(c - all_checksums[0]) < 1e-6 for c in all_checksums)

    # Address of the underlying data buffer (same node -> should be identical
    # only for ranks that attached the SAME shared window, not literally
    # equal across nodes, but consistent within a node run this is a good sniff test).
    buf_ptr = obsmat.matrix.data.ctypes.data

    print(
        f"[rank {rank:4d} host {host} node_rank {node_rank:3d}] "
        f"load_time={dt:6.2f}s  rss_before={rss_before:8.1f}MB  "
        f"rss_after={rss_after:8.1f}MB  delta={rss_after - rss_before:8.1f}MB  "
        f"nnz={obsmat.nnz}  buf_ptr={buf_ptr:#x}  checksum_ok={checksums_match}",
        flush=True,
    )

    comm.Barrier()
    if rank == 0:
        print("\n--- summary ---")
        print(f"world size: {size}")
        print(f"checksums match across ALL ranks: {checksums_match}")
        print("Expect: only node_rank==0 processes show a large rss delta;")
        print("        node_rank!=0 processes should show ~0 MB delta,")
        print("        since they only attach to shared memory, not copy it.")
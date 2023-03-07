## The gist

If all of the following are true, I get a hang on the second of four `MPI_Irecv` calls on Frontier right now:
1. I use a memory pool type approach of allocating data up front and using offsets of that pool
2. I use GPU-aware MPI by using GPU pointers for data buffers
3. I `hipMalloc` anything more than 16*2^30 bytes (16 GB) for the pool.

## Instructions

```bash
git clone git@github.com:mrnorman/frontier-gpu-aware-hang-reproduce.git
cd frontier-gpu-aware-hang-reproduce
git submodule update --init
cd build
# salloc -A ABC123 -J debug_job -t 1:00:00 -N 8 -p batch
# The following should succeed
source crusher_gpu.env            && ./cmakescript.sh && make -j4 && srun -N8 -n64 --gpus-per-node=8 --ntasks-per-gpu=1 -c 1 --gpu-bind=closest ./driver_without_yakl
# The following should hang on line 124
source crusher_gpu_expose_bug.env && ./cmakescript.sh && make -j4 && srun -N8 -n64 --gpus-per-node=8 --ntasks-per-gpu=1 -c 1 --gpu-bind=closest ./driver_without_yakl
```

There is also a reproducer with a driver that uses the YAKL library (as in the original app), but I figured you wouldn't want to use that when there's a simpler reproducer without that library.

## Output files

This writes a one-file-per-MPI-task record of debugging for convenience to see that all tasks are stalling at the same line with the above conditions are met.

## My modules for this reproducer

```bash
[imn@login05:/gpfs/alpine/proj-shared/stf006/imn/frontier/frontier-gpu-aware-hang-reproduce/build] 8-) module -t list
craype-x86-trento
libfabric/1.15.2.0
craype-network-ofi
perftools-base/22.12.0
xpmem/2.5.2-2.4_3.20__gd0f7936.shasta
cray-pmi/6.1.8
amd/5.3.0
craype/2.7.19
cray-dsmml/0.2.2
cray-mpich/8.1.23
cray-libsci/22.12.1.1
PrgEnv-amd/8.3.3
hsi/default
DefApps/default
cray-parallel-netcdf/1.12.3.1
cmake/3.23.2
craype-accel-amd-gfx90a
cray-hdf5/1.12.2.1
cray-netcdf/4.9.0.1
```

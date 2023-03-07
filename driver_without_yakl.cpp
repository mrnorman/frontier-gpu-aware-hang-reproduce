
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "hip/hip_runtime.h"

inline void verbose_inform(std::string prefix, std::string label = "", std::string suffix = "") {
  // Form the output
  std::string output = std::string("*** [YAKL_VERBOSE] ") + prefix;
  if (label != "") output += std::string(" (label: \"") + label + std::string("\")");
  if (suffix != "") output += std::string(";  ") + suffix;

  // Get MPI rank
  int rank = 0;
  int is_initialized;
  MPI_Initialized(&is_initialized);
  if (is_initialized) { MPI_Comm_rank(MPI_COMM_WORLD, &rank); }

  // Write to file
  std::ofstream myfile;
  std::string fname = std::string("yakl_verbose_output_task_") + std::to_string(rank) + std::string(".log");
  myfile.open(fname , std::ofstream::out | std::ofstream::app);
  myfile << output << std::endl;
  myfile.close();

  // Write to stdout for task 0
  if (rank == 0) std::cout << output << std::endl;
}

#define DEBUG_PRINT() verbose_inform("DEBUG GOT HERE: " , std::string(__FILE__)+std::string(": ")+std::to_string(__LINE__) );


int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  {
    // Simulation parameters
    int nx_glob       = 1024;
    int ny_glob       = 1024;
    int nz            = 100;
    int num_tracers   = 3;
    int constexpr ord = 3;
    int constexpr hs  = 1;
    int constexpr num_state = 5;
    int npack = num_state + num_tracers;
    // Pool
    void *ptr;
    size_t bytes = 16;
    bytes *= 1024;
    bytes *= 1024;
    bytes *= 1024;
    #ifdef EXPOSE_THE_BUG
      bytes += 16*sizeof(size_t);
    #endif
    hipMalloc(&ptr,bytes);
    double *offset = (double *) ptr;

    ////////////////////////////////////////////////////
    // Setup MPI
    ////////////////////////////////////////////////////
    int nx, ny, nproc_x, nproc_y, nranks, px, py, i_beg, i_end, j_beg, j_end, myrank;
    bool mainproc;
    double neigh[3][3];
    MPI_Comm_size( MPI_COMM_WORLD , &nranks );
    MPI_Comm_rank( MPI_COMM_WORLD , &myrank );

    mainproc = (myrank == 0);

    // Find integer nproc_y * nproc_x == nranks such that nproc_y and nproc_x are as close as possible
    nproc_y = (int) std::ceil( std::sqrt((double) nranks) );
    while (nproc_y >= 1) {
      if (nranks % nproc_y == 0) { break; }
      nproc_y--;
    }
    nproc_x = nranks / nproc_y;

    // Get my ID within each dimension's number of ranks
    py = myrank / nproc_x;
    px = myrank % nproc_x;

    // Get my beginning and ending indices in the x- and y- directions
    double nper;
    nper = ((double) nx_glob)/nproc_x;
    i_beg = static_cast<size_t>( round( nper* px    )   );
    i_end = static_cast<size_t>( round( nper*(px+1) )-1 );
    nper = ((double) ny_glob)/nproc_y;
    j_beg = static_cast<size_t>( round( nper* py    )   );
    j_end = static_cast<size_t>( round( nper*(py+1) )-1 );

    //Determine my number of grid cells
    nx = i_end - i_beg + 1;
    ny = j_end - j_beg + 1;
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 3; i++) {
        int pxloc = px+i-1;
        while (pxloc < 0        ) { pxloc = pxloc + nproc_x; }
        while (pxloc > nproc_x-1) { pxloc = pxloc - nproc_x; }
        int pyloc = py+j-1;
        while (pyloc < 0        ) { pyloc = pyloc + nproc_y; }
        while (pyloc > nproc_y-1) { pyloc = pyloc - nproc_y; }
        neigh[j][i] = pyloc * nproc_x + pxloc;
      }
    }

    ///////////////////////////////////////////////////
    // Mimic time step MPI workflow
    ///////////////////////////////////////////////////
    double * halo_send_buf_W = offset;   offset += npack*nz*ny*hs;
    double * halo_send_buf_E = offset;   offset += npack*nz*ny*hs;
    double * halo_send_buf_S = offset;   offset += npack*nz*hs*nx;
    double * halo_send_buf_N = offset;   offset += npack*nz*hs*nx;
    double * halo_recv_buf_W = offset;   offset += npack*nz*ny*hs;
    double * halo_recv_buf_E = offset;   offset += npack*nz*ny*hs;
    double * halo_recv_buf_S = offset;   offset += npack*nz*hs*nx;
    double * halo_recv_buf_N = offset;   offset += npack*nz*hs*nx;

    MPI_Request sReq[4];
    MPI_Request rReq[4];

    hipDeviceSynchronize(); DEBUG_PRINT();
    
    auto type = MPI_DOUBLE;
    auto comm = MPI_COMM_WORLD;
    MPI_Irecv( halo_recv_buf_W , npack*nz*ny*hs*sizeof(double) , type , neigh[1][0] , 0 , comm , &rReq[0] ); DEBUG_PRINT();
    MPI_Irecv( halo_recv_buf_E , npack*nz*ny*hs*sizeof(double) , type , neigh[1][2] , 1 , comm , &rReq[1] ); DEBUG_PRINT();
    MPI_Irecv( halo_recv_buf_S , npack*nz*hs*nx*sizeof(double) , type , neigh[0][1] , 2 , comm , &rReq[2] ); DEBUG_PRINT();
    MPI_Irecv( halo_recv_buf_N , npack*nz*hs*nx*sizeof(double) , type , neigh[2][1] , 3 , comm , &rReq[3] ); DEBUG_PRINT();

    MPI_Isend( halo_send_buf_W , npack*nz*ny*hs*sizeof(double) , type , neigh[1][0] , 1 , comm , &sReq[0] ); DEBUG_PRINT();
    MPI_Isend( halo_send_buf_E , npack*nz*ny*hs*sizeof(double) , type , neigh[1][2] , 0 , comm , &sReq[1] ); DEBUG_PRINT();
    MPI_Isend( halo_send_buf_S , npack*nz*hs*nx*sizeof(double) , type , neigh[0][1] , 3 , comm , &sReq[2] ); DEBUG_PRINT();
    MPI_Isend( halo_send_buf_N , npack*nz*hs*nx*sizeof(double) , type , neigh[2][1] , 2 , comm , &sReq[3] ); DEBUG_PRINT();

    MPI_Status  sStat[4];
    MPI_Status  rStat[4];

    MPI_Waitall(4, sReq, sStat);
    MPI_Waitall(4, rReq, rStat);

    hipDeviceSynchronize();

  }
  MPI_Finalize();
}



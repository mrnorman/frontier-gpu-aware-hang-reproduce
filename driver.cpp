
#include "YAKL.h"

typedef double real;
typedef yakl::Array<real,1,yakl::memDevice> real1d;
typedef yakl::Array<real,2,yakl::memDevice> real2d;
typedef yakl::Array<real,3,yakl::memDevice> real3d;
typedef yakl::Array<real,4,yakl::memDevice> real4d;
typedef yakl::Array<real,4,yakl::memDevice> real5d;
typedef yakl::Array<real,5,yakl::memDevice> real6d;

typedef yakl::Array<real,1,yakl::memHost> realHost1d;
typedef yakl::Array<real,2,yakl::memHost> realHost2d;
typedef yakl::Array<real,3,yakl::memHost> realHost3d;
typedef yakl::Array<real,4,yakl::memHost> realHost4d;
typedef yakl::Array<real,5,yakl::memHost> realHost5d;
typedef yakl::Array<real,6,yakl::memHost> realHost6d;

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    int nx_glob = 1024;
    int ny_glob = 1024;
    int nz = 100;
    int num_tracers = 3;
    int constexpr ord = 3;
    int constexpr hs  = 1;
    int constexpr num_state = 5;
    int npack = num_state + num_tracers;
    int nx, ny, nproc_x, nproc_y, nranks, px, py, i_beg, i_end, j_beg, j_end, myrank;
    bool mainproc;
    yakl::SArray<real,2,3,3> neigh;

    MPI_Comm_size( MPI_COMM_WORLD , &nranks );
    MPI_Comm_rank( MPI_COMM_WORLD , &myrank );

    mainproc = (myrank == 0);

    bool sim2d = ny_glob == 1;

    if (sim2d) {
      nproc_x = nranks;
      nproc_y = 1;
    } else {
      // Find integer nproc_y * nproc_x == nranks such that nproc_y and nproc_x are as close as possible
      nproc_y = (int) std::ceil( std::sqrt((double) nranks) );
      while (nproc_y >= 1) {
        if (nranks % nproc_y == 0) { break; }
        nproc_y--;
      }
      nproc_x = nranks / nproc_y;
    }

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
        neigh(j,i) = pyloc * nproc_x + pxloc;
      }
    }

    real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
    real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
    real1d hy_dens_cells      ("hy_dens_cells"      ,nz  );
    real1d hy_dens_theta_cells("hy_dens_theta_cells",nz  );
    real1d hy_dens_edges      ("hy_dens_edges"      ,nz+1);
    real1d hy_dens_theta_edges("hy_dens_theta_edges",nz+1);
    real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
    real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
    real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
    real4d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     );

    realHost4d halo_send_buf_W_host("halo_send_buf_W_host",npack,nz,ny,hs);
    realHost4d halo_send_buf_E_host("halo_send_buf_E_host",npack,nz,ny,hs);
    realHost4d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx);
    realHost4d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx);
    realHost4d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx);
    realHost4d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx);
    realHost4d halo_recv_buf_W_host("halo_recv_buf_W_host",npack,nz,ny,hs);
    realHost4d halo_recv_buf_E_host("halo_recv_buf_E_host",npack,nz,ny,hs);

    real4d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs);
    real4d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs);

    parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,ny,hs) , YAKL_LAMBDA (int v, int k, int j, int ii) {
      if (v < num_state) {
        halo_send_buf_W(v,k,j,ii) = state  (v          ,hs+k,hs+j,hs+ii);
        halo_send_buf_E(v,k,j,ii) = state  (v          ,hs+k,hs+j,nx+ii);
      } else {
        halo_send_buf_W(v,k,j,ii) = tracers(v-num_state,hs+k,hs+j,hs+ii);
        halo_send_buf_E(v,k,j,ii) = tracers(v-num_state,hs+k,hs+j,nx+ii);
      }
    });

    real4d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx);
    real4d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx);

    if (!sim2d) {
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,hs,nx) , YAKL_LAMBDA (int v, int k, int jj, int i) {
        if (v < num_state) {
          halo_send_buf_S(v,k,jj,i) = state  (v          ,hs+k,hs+jj,hs+i);
          halo_send_buf_N(v,k,jj,i) = state  (v          ,hs+k,ny+jj,hs+i);
        } else {
          halo_send_buf_S(v,k,jj,i) = tracers(v-num_state,hs+k,hs+jj,hs+i);
          halo_send_buf_N(v,k,jj,i) = tracers(v-num_state,hs+k,ny+jj,hs+i);
        }
      });
    }

    yakl::fence();
    yakl::timer_start("halo_exchange_mpi");

    real4d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
    real4d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
    real4d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx);
    real4d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx);

    MPI_Request sReq[4];
    MPI_Request rReq[4];

    MPI_Datatype mpi_data_type = MPI_DOUBLE;

    yakl::fence();

    //Pre-post the receives
    MPI_Irecv( halo_recv_buf_W.data() , halo_recv_buf_W.size() , mpi_data_type , neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
    MPI_Irecv( halo_recv_buf_E.data() , halo_recv_buf_E.size() , mpi_data_type , neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
    if (!sim2d) {
      MPI_Irecv( halo_recv_buf_S.data() , halo_recv_buf_S.size() , mpi_data_type , neigh(0,1) , 2 , MPI_COMM_WORLD , &rReq[2] );
      MPI_Irecv( halo_recv_buf_N.data() , halo_recv_buf_N.size() , mpi_data_type , neigh(2,1) , 3 , MPI_COMM_WORLD , &rReq[3] );
    }

    //Send the data
    MPI_Isend( halo_send_buf_W.data() , halo_send_buf_W.size() , mpi_data_type , neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
    MPI_Isend( halo_send_buf_E.data() , halo_send_buf_E.size() , mpi_data_type , neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );
    if (!sim2d) {
      MPI_Isend( halo_send_buf_S.data() , halo_send_buf_S.size() , mpi_data_type , neigh(0,1) , 3 , MPI_COMM_WORLD , &sReq[2] );
      MPI_Isend( halo_send_buf_N.data() , halo_send_buf_N.size() , mpi_data_type , neigh(2,1) , 2 , MPI_COMM_WORLD , &sReq[3] );
    }

    MPI_Status  sStat[4];
    MPI_Status  rStat[4];

    //Wait for the sends and receives to finish
    if (sim2d) {
      MPI_Waitall(2, sReq, sStat);
      MPI_Waitall(2, rReq, rStat);
    } else {
      MPI_Waitall(4, sReq, sStat);
      MPI_Waitall(4, rReq, rStat);
    }
    yakl::fence();
    yakl::timer_stop("halo_exchange_mpi");

  }
  yakl::finalize();
  MPI_Finalize();
}



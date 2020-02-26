/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildModExtInterp
 *  Comment:
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildModExtInterp(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                              hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
                              HYPRE_Int num_functions, HYPRE_Int *dof_func, // first version for num_functions = 1 only!
                              HYPRE_Int debug_flag,
                              HYPRE_Real strong_th,
                              HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                              HYPRE_Int *col_offd_S_to_A,
                              hypre_ParCSRMatrix  **P_ptr)
{
   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);


   HYPRE_Int              my_id, num_procs;

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   //HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   //HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   //HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   //HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   //HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
   HYPRE_BigInt     total_global_cpts;
   //HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   //hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   //HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   //HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   //hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   //HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   //HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /* Intermediate matrices */
   hypre_ParCSRMatrix *As_FF, *As_FC, *W;
   HYPRE_Real *D_f, *D_q, *D_w;
   hypre_CSRMatrix *As_FF_diag;
   hypre_CSRMatrix *As_FF_offd;
   hypre_CSRMatrix *As_FC_diag;
   hypre_CSRMatrix *As_FC_offd;
   hypre_CSRMatrix *W_diag;
   hypre_CSRMatrix *W_offd;

   HYPRE_Int *As_FF_diag_i;
   HYPRE_Int *As_FF_offd_i;
   HYPRE_Int *As_FC_diag_i;
   HYPRE_Int *As_FC_offd_i;
   HYPRE_Int *W_diag_i;
   HYPRE_Int *W_offd_i;
   HYPRE_Int *W_diag_j;
   HYPRE_Int *W_offd_j;

   HYPRE_Real *As_FF_diag_data;
   HYPRE_Real *As_FF_offd_data;
   HYPRE_Real *As_FC_diag_data;
   HYPRE_Real *As_FC_offd_data;
   HYPRE_Real *W_diag_data;
   HYPRE_Real *W_offd_data;

   HYPRE_BigInt    *col_map_offd_P = NULL;
   HYPRE_BigInt    *new_col_map_offd = NULL;
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int        new_ncols_P_offd;
   HYPRE_Int        num_cols_P_offd;
   HYPRE_Int       *P_marker = NULL;

   /* Loop variables */
   HYPRE_Int        index;
   HYPRE_Int        i, j;
   //HYPRE_Int        jd, jo, col_S, col_A;
   HYPRE_Int       *cpt_array;
   HYPRE_Int       *start_array;
   HYPRE_Int       *startf_array;
   HYPRE_Int start, stop, startf, stopf;
   HYPRE_Int cnt_diag, cnt_offd, row, c_pt;

   /* Definitions */
   //HYPRE_Real       wall_time;
   HYPRE_Int n_Cpts, n_Fpts;
   HYPRE_Int num_threads = hypre_NumThreads();

   //if (debug_flag==4) wall_time = time_getWallclockSeconds();

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
   n_Cpts = num_cpts_global[1]-num_cpts_global[0];
#else
   //my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
   n_Cpts = num_cpts_global[my_id+1]-num_cpts_global[my_id];
#endif

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   // start code here >>>>>
   hypre_ParCSRMatrixGenerateFFFC(A, CF_marker, num_cpts_global, S, &As_FC, &As_FF);
   //hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "FC", &As_FC, strong_th);
   //hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "FF", &As_FF, strong_th);

   As_FC_diag = hypre_ParCSRMatrixDiag(As_FC);
   As_FC_diag_i = hypre_CSRMatrixI(As_FC_diag);
   As_FC_diag_data = hypre_CSRMatrixData(As_FC_diag);
   As_FC_offd = hypre_ParCSRMatrixOffd(As_FC);
   As_FC_offd_i = hypre_CSRMatrixI(As_FC_offd);
   As_FC_offd_data = hypre_CSRMatrixData(As_FC_offd);
   As_FF_diag = hypre_ParCSRMatrixDiag(As_FF);
   As_FF_diag_i = hypre_CSRMatrixI(As_FF_diag);
   As_FF_diag_data = hypre_CSRMatrixData(As_FF_diag);
   As_FF_offd = hypre_ParCSRMatrixOffd(As_FF);
   As_FF_offd_i = hypre_CSRMatrixI(As_FF_offd);
   As_FF_offd_data = hypre_CSRMatrixData(As_FF_offd);
   n_Fpts = hypre_CSRMatrixNumRows(As_FF_diag);

   D_q = hypre_CTAlloc(HYPRE_Real, n_Fpts, HYPRE_MEMORY_HOST);
   D_f = hypre_CTAlloc(HYPRE_Real, n_Fpts, HYPRE_MEMORY_HOST);
   D_w = hypre_CTAlloc(HYPRE_Real, n_Fpts, HYPRE_MEMORY_HOST);
   cpt_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   start_array = hypre_CTAlloc(HYPRE_Int, num_threads+1, HYPRE_MEMORY_HOST);
   startf_array = hypre_CTAlloc(HYPRE_Int, num_threads+1, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,j,start,stop,startf,stopf,row)
#endif
   {
      HYPRE_Int my_thread_num = hypre_GetThreadNum();

      start = (n_fine/num_threads)*my_thread_num;
      if (my_thread_num == num_threads-1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine/num_threads)*(my_thread_num+1);
      }
      start_array[my_thread_num+1] = stop;
      for (i=start; i < stop; i++)
      {
         if (CF_marker[i] > 0) 
         {
            cpt_array[my_thread_num]++;
         }
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         for (i=1; i < num_threads; i++)
         {
            cpt_array[i] += cpt_array[i-1];
         }
      }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
      if (my_thread_num > 0)
         startf = start - cpt_array[my_thread_num-1];
      else
         startf = 0;

      if (my_thread_num < num_threads-1)
         stopf = stop - cpt_array[my_thread_num];
      else
         stopf = n_Fpts;

      startf_array[my_thread_num+1] = stopf;

      for (i=startf; i < stopf; i++)
      {
         for (j=As_FC_diag_i[i]; j < As_FC_diag_i[i+1]; j++)
         {
            D_q[i] += As_FC_diag_data[j];
         }
         for (j=As_FC_offd_i[i]; j < As_FC_offd_i[i+1]; j++)
         {
            D_q[i] += As_FC_offd_data[j];
         }
         // Assumes diagonal element first element in row of As_FF_diag
         D_f[i] += As_FF_diag_data[As_FF_diag_i[i]];
      }

      row = startf;
      for (i=start; i < stop; i++)
      {
         if (CF_marker[i] < 0)
         {
            for (j=A_diag_i[i]; j < A_diag_i[i+1]; j++)
            {
               D_w[row] += A_diag_data[j];
            }
            for (j=A_offd_i[i]; j < A_offd_i[i+1]; j++)
            {
               D_w[row] += A_offd_data[j];
            }
            for (j=As_FF_diag_i[row]; j < As_FF_diag_i[row+1]; j++)
            {
               D_w[row] -= As_FF_diag_data[j];
            }
            for (j=As_FF_offd_i[row]; j < As_FF_offd_i[row+1]; j++)
            {
               D_w[row] -= As_FF_offd_data[j];
            }
            D_w[row] -= D_q[row];
            row++;
         }
      }

      for (i=startf; i<stopf; i++)
      {
         D_w[i] = 1.0/(D_f[i]+D_w[i]);
         j = As_FF_diag_i[i];      
         As_FF_diag_data[j] = D_w[i]*D_q[i];
         D_q[i] = -1.0/D_q[i]; 
         for (j=As_FF_diag_i[i]+1; j < As_FF_diag_i[i+1]; j++)
            As_FF_diag_data[j] *= D_w[i];
         for (j=As_FF_offd_i[i]; j < As_FF_offd_i[i+1]; j++)
            As_FF_offd_data[j] *= D_w[i];
         for (j=As_FC_diag_i[i]; j < As_FC_diag_i[i+1]; j++)
            As_FC_diag_data[j] *= D_q[i];
         for (j=As_FC_offd_i[i]; j < As_FC_offd_i[i+1]; j++)
            As_FC_offd_data[j] *= D_q[i];
      }

   }   /* end parallel region */ 

   W = hypre_ParMatmul(As_FF, As_FC);
   W_diag = hypre_ParCSRMatrixDiag(W);
   W_offd = hypre_ParCSRMatrixOffd(W);
   W_diag_i = hypre_CSRMatrixI(W_diag); 
   W_diag_j = hypre_CSRMatrixJ(W_diag); 
   W_diag_data = hypre_CSRMatrixData(W_diag); 
   W_offd_i = hypre_CSRMatrixI(W_offd); 
   W_offd_j = hypre_CSRMatrixJ(W_offd); 
   W_offd_data = hypre_CSRMatrixData(W_offd); 
   /*-----------------------------------------------------------------------
    *  Intialize data for P
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_HOST);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_HOST);

   P_diag_size = n_Cpts + hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W))[n_Fpts];
   P_offd_size = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W))[n_Fpts];

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
      P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, HYPRE_MEMORY_HOST);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
      P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, HYPRE_MEMORY_HOST);
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,j,start,stop,startf,stopf,c_pt,row,cnt_diag,cnt_offd)
#endif
   {
      HYPRE_Int my_thread_num = hypre_GetThreadNum();
      startf = startf_array[my_thread_num];
      stopf = startf_array[my_thread_num+1];
      start = start_array[my_thread_num];
      stop = start_array[my_thread_num+1];

      if (my_thread_num > 0)
         c_pt = cpt_array[my_thread_num-1];
      else
         c_pt = 0;
      cnt_diag = W_diag_i[startf]+c_pt;
      cnt_offd = W_offd_i[startf];
      row = startf;
      for (i=start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            P_diag_j[cnt_diag] = c_pt++;
            P_diag_data[cnt_diag++] = 1.0;
         }
         else
         {
            for (j=W_diag_i[row]; j < W_diag_i[row+1]; j++)
            {
               P_diag_j[cnt_diag] = W_diag_j[j];
               P_diag_data[cnt_diag++] = W_diag_data[j];
            }
            for (j=W_offd_i[row]; j < W_offd_i[row+1]; j++)
            {
               P_offd_j[cnt_offd] = W_offd_j[j];
               P_offd_data[cnt_offd++] = W_offd_data[j];
            }
            row++;
         }
         P_diag_i[i+1] = cnt_diag;
         P_offd_i[i+1] = cnt_offd;
      }

   }   /* end parallel region */ 
   
   /*-----------------------------------------------------------------------
    *  Create matrix
    *-----------------------------------------------------------------------*/

   num_cols_P_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W));
   P = hypre_ParCSRMatrixCreate(comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         total_global_cpts,
         hypre_ParCSRMatrixColStarts(A),
         num_cpts_global,
         num_cols_P_offd,
         P_diag_i[n_fine],
         P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
   hypre_ParCSRMatrixColMapOffd(P) = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(W) = NULL;

   hypre_CSRMatrixMemoryLocation(P_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(P_offd) = HYPRE_MEMORY_HOST;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      HYPRE_Int *map;
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
      //replace col_map_offd
      col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);
      P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      for (i=0; i < P_offd_size; i++)
         P_marker[P_offd_j[i]] = 1;
      
      new_ncols_P_offd = 0;
      for (i=0; i < num_cols_P_offd; i++)
         if (P_marker[i]) new_ncols_P_offd++;

      new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt, new_ncols_P_offd, HYPRE_MEMORY_HOST);
      map = hypre_CTAlloc(HYPRE_Int, new_ncols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
         if (P_marker[i])
         {
             new_col_map_offd[index] = col_map_offd_P[i];
             map[index++] = i;
         }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);


#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i=0; i < P_offd_size; i++)
      {
         P_offd_j[i] = hypre_BigBinarySearch(map, P_offd_j[i],
               new_ncols_P_offd);
      }
      hypre_TFree(col_map_offd_P, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColMapOffd(P) = new_col_map_offd; 
      hypre_CSRMatrixNumCols(P_offd) = new_ncols_P_offd; 
      hypre_TFree(map, HYPRE_MEMORY_HOST);
   }

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(D_q, HYPRE_MEMORY_HOST);
   hypre_TFree(D_w, HYPRE_MEMORY_HOST);
   hypre_TFree(D_f, HYPRE_MEMORY_HOST);
   hypre_TFree(cpt_array, HYPRE_MEMORY_HOST);
   hypre_TFree(start_array, HYPRE_MEMORY_HOST);
   hypre_TFree(startf_array, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixDestroy(As_FF);
   hypre_ParCSRMatrixDestroy(As_FC);
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}


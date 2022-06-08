
#include "HYPRE.h"
#include "seq_mv.h"

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   /* Initialize hypre (sets up sycl queue, etc.) */
   HYPRE_Init();

   /* Read in matrices and move to shared memory */
   hypre_CSRMatrix *A = hypre_CSRMatrixRead("A");
   hypre_CSRMatrix *B = hypre_CSRMatrixRead("B");
   A = hypre_CSRMatrixClone_v2(A, 1, HYPRE_MEMORY_DEVICE);
   B = hypre_CSRMatrixClone_v2(B, 1, HYPRE_MEMORY_DEVICE);

   /* Do the matmats on GPU and CPU */
   hypre_printf("Do matmats\n");
   hypre_CSRMatrix *C_device = hypre_CSRMatrixMultiplyDevice(A, B);
   hypre_CSRMatrix *C_host = hypre_CSRMatrixMultiplyHost(A, B);

   /* Compare the results */
   hypre_printf("Compare results\n");
   C_device = hypre_CSRMatrixClone_v2(C_device, 1, HYPRE_MEMORY_HOST);
   hypre_CSRMatrix *C_error = hypre_CSRMatrixAdd(1.0, C_host, -1.0, C_device);
   hypre_printf("done with add\n");
   hypre_CSRMatrix *C_error_remove_zeros = hypre_CSRMatrixDeleteZeros(C_error, 1.e-15);
   hypre_printf("done with remove\n");
   hypre_printf("C_error_remove_zeros = %p\n", C_error_remove_zeros);
   if (C_error_remove_zeros)
   {
      hypre_printf("WM: getting fnorm\n");
      HYPRE_Real fnorm = hypre_CSRMatrixFnorm(C_error_remove_zeros);
      hypre_printf("WM: getting fnorm\n");
      HYPRE_Real fnorm0 = hypre_CSRMatrixFnorm(C_host);
      hypre_printf("WM: done getting fnorm\n");
      HYPRE_Real rfnorm = fnorm0 > 0 ? fnorm / fnorm0 : fnorm;

      hypre_printf("Relative norm error between host and device matmat: %e\n", rfnorm);
      if (rfnorm > 0.01)
      {
         hypre_CSRMatrixPrint(A, "A");
         hypre_CSRMatrixPrint(B, "B");
         hypre_CSRMatrixPrint(C_device, "C_device");
         hypre_CSRMatrixPrint(C_host, "C_host");
         hypre_CSRMatrixPrint(C_error_remove_zeros, "C_error");
         hypre_printf("WM: done printing\n");
      }
      hypre_CSRMatrixDestroy(C_error_remove_zeros);
   }
   else
   {
      hypre_printf("No error between host and device matmat\n");
   }

   hypre_CSRMatrixDestroy(C_device);
   hypre_CSRMatrixDestroy(C_host);
   hypre_CSRMatrixDestroy(C_error);
}

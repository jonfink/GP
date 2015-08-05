#include "armadillo_backsub.h"

#include <clapack.h>

namespace arma
{
  extern "C"
  {

    void arma_fortran_prefix(spotrs)(char* uplo, blas_int* n, blas_int *nrhs, float* a, blas_int* lda, float* b, blas_int* ldb, blas_int* info)
    {
      arma_fortran_noprefix(spotrs)(uplo, n, nrhs, a, lda, b, ldb, info);
    }

    void arma_fortran_prefix(dpotrs)(char* uplo, blas_int* n, blas_int *nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* info)
    {
      arma_fortran_noprefix(dpotrs)(uplo, n, nrhs, a, lda, b, ldb, info);
    }

  }

}

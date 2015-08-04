#ifndef ARMADILLO_BACKSUB_H_
#define ARMADILLO_BACKSUB_H_

#include <armadillo>
using namespace arma;

namespace arma
{

  extern "C"
  {
// Cholesky decomposition Solve
    void arma_fortran(spotrs)(char* uplo, blas_int* n, blas_int *nrhs, float* a, blas_int* lda, float* b, blas_int* ldb, blas_int* info);
    void arma_fortran(dpotrs)(char* uplo, blas_int* n, blas_int *nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* info);
  }

  namespace lapack
  {

    template<typename eT>
      inline
      void
      potrs(char* uplo, blas_int* n, blas_int *nrhs, const eT* a, blas_int* lda, eT* b, blas_int* ldb, blas_int* info)
    {
      arma_type_check(( is_supported_blas_type<eT>::value == false ));

      if(is_float<eT>::value == true)
      {
        typedef float T;
        arma_fortran(spotrs)(uplo, n, nrhs, (T*)a, lda, (T*)b, ldb, info);
      }
      else
        if(is_double<eT>::value == true)
        {
          typedef double T;
          arma_fortran(dpotrs)(uplo, n, nrhs, (T*)a, lda, (T*)b, ldb, info);
        }
        else {
          
          /* if(is_supported_complex_float<eT>::value == true) */
          /* { */
          /*   typedef std::complex<float> T; */
          /*   arma_fortran(cpotrs)(uplo, n, nrhs, (T*)a, lda, (T*)b, ldb, info); */
          /* } */
          /* else */
          /*   if(is_supported_complex_double<eT>::value == true) */
          /*   { */
          /*     typedef std::complex<double> T; */
          /*     arma_fortran(zpotrs)(uplo, n, nrhs, (T*)a, lda, (T*)b, ldb, info); */
          /*   } */
        }
        
    }

  }

  template<typename eT>
    inline
    bool
    cholbacksub(Mat<eT>& out, const Mat<eT>& U, const Mat<eT>& B)
  {
    arma_extra_debug_sigprint();
    
#if defined(ARMA_USE_LAPACK)
    {
      char uplo = 'U';
      blas_int  n    = blas_int(U.n_rows);
      blas_int nrhs = blas_int(B.n_cols);
      blas_int info = 0;
                        
      Mat<eT> U_copy = U;
      
      out = B;
      lapack::potrs(&uplo, &n, &nrhs, U_copy.memptr(), &n, out.memptr(), &n, &info);
      
      return (info == 0);
    }
#else
    {
      arma_stop("auxlib::cholbacksub(): need LAPACK library");
      return false;
    }
#endif
  }

  template<typename eT, typename T1, typename T2>
    inline
    bool
    cholbacksub(Mat<eT>& out, const Base<eT,T1>& U, const Base<eT,T2>&b)
  {
    arma_extra_debug_sigprint();
    const unwrap<T1> tmp_A(U.get_ref());
    const unwrap<T2> tmp_B(b.get_ref());
          
    const Mat<eT>& A = tmp_A.M;
    const Mat<eT>& B = tmp_B.M;
              
    arma_debug_check( !A.is_square(), "cholbacksub(): given matrix is not square");
    arma_debug_check( A.n_rows != B.n_rows, "cholbacksub(): A and B must have same number of rows");
              
    return cholbacksub(out, A, B);
  }


//! Solve a system of linear equations
//! where A is triangular
//! Assumes that A.n_rows = A.n_cols
//! and B.n_rows = A.n_rows
  template<typename eT>
    inline
    bool
    solve_tri(Mat<eT>& out, const Mat<eT>& A, const Mat<eT>& B, const bool upper)
  {
    arma_extra_debug_sigprint();
    
#if defined(ARMA_USE_LAPACK)
    {
      char uplo = (upper ? 'U' : 'L');
      char trans = 'N';
      char diag = 'N';
      blas_int n    = blas_int(A.n_rows);
      blas_int lda  = blas_int(A.n_rows);
      blas_int ldb  = blas_int(A.n_rows);
      blas_int nrhs = blas_int(B.n_cols);
      blas_int info = 0;
      out = B;
      Mat<eT> A_copy = A;
          
      lapack::trtrs<eT>(&uplo, &trans, &diag, &n, &nrhs,
                        A_copy.memptr(), &lda,
                        out.memptr(), &ldb, &info);
      return (info == 0);
    }
#else
    {
      arma_stop("auxlib::solve_tri(): need LAPACK library");
      return false;
    }
#endif
  }  


//! Solve a system of linear equations, i.e., A*X = B, where X is unknown.
//! and X is upper or lower triangular
  template<typename eT, typename T1, typename T2>
    inline
    bool
    solve_tri(Mat<eT>& X, const Base<eT,T1>& A_in, const Base<eT,T2>& B_in, const bool upper=true)
  {
    arma_extra_debug_sigprint();
 
    const unwrap<T1> tmp1(A_in.get_ref());
    const unwrap<T2> tmp2(B_in.get_ref());

    const Mat<eT>& A = tmp1.M;
    const Mat<eT>& B = tmp2.M;

    arma_debug_check( (A.n_rows != B.n_rows), "solve_tri(): number of rows in A and B must be the same" );

    arma_debug_check( (A.n_rows != A.n_cols), "solve_tri(): A must be square" );

    return solve_tri(X, A, B, upper);

  }

  template<typename eT, typename T1, typename T2>
    inline
    Mat<eT>
    solve_tri(const Base<eT,T1>& A_in, const Base<eT,T2>& B_in, const bool upper=true)
  {
    arma_extra_debug_sigprint();

    Mat<eT> X;
    bool info = solve_tri(X, A_in, B_in, upper);

    if(info == false)
    {
      arma_print("solve_tri(): solution not found");
    }

    return X;
  }

}

#endif

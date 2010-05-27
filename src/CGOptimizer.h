#ifndef _COGOPTIMIZER_H_
#define _COGOPTIMIZER_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>

class GP;

class CGOptimizer {
 public:
  CGOptimizer(void *gp);
  
  void Initialize(const Col<REAL> &init, 
		  double (*f_eval)(const gsl_vector*, void*), 
		  void (*df_eval)(const gsl_vector*, void*, gsl_vector*),
		  void (*fdf_eval)(const gsl_vector*, void*, double*, gsl_vector*),
		  double step=10.0, double eps=0.1);

  int Optimize(Col<REAL> &soln, int max_iterations=10);

 private:
  void *gp;
  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;
  gsl_multimin_function_fdf my_func;

  gsl_vector *x;
};

#endif

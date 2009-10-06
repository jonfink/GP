#ifndef _COGOPTIMIZER_H_
#define _COGOPTIMIZER_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>

class GP;

double f_eval_mean(const gsl_vector *x, void *param);
void df_eval_mean(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_mean(const gsl_vector *x, void *param, double *f, gsl_vector *g);

double f_eval_kernel(const gsl_vector *x, void *param);
void df_eval_kernel(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_kernel(const gsl_vector *x, void *param, double *f, gsl_vector *g);

double f_eval_noise(const gsl_vector *x, void *param);
void df_eval_noise(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_noise(const gsl_vector *x, void *param, double *f, gsl_vector *g);

class CGOptimizer {
 public:
  CGOptimizer(GP *gp);
  
  void InitializeNoise(const REAL &init, double step=0.5);
  void InitializeMean(const Col<REAL> &init, double step=10.0);
  void InitializeKernel(const Col<REAL> &init, double step=10.0);

  int Optimize(Col<REAL> &soln, int max_iterations=10);

 private:
  GP *gp;
  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;
  gsl_multimin_function_fdf my_func;

  gsl_vector *x;
};

#endif

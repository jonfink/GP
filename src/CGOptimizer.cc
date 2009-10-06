#include <gsl/gsl_multimin.h>
#include "GP.h"

#include <iostream>
using namespace std;

#include "CGOptimizer.h"

double f_eval_mean(const gsl_vector *x, void *param)
{
  GP *gp_obj = reinterpret_cast<GP*>(param);
  
  Col<REAL> mean_param(gp_obj->GetMeanFunction()->GetParamDim());
  for(int i=0; i < mean_param.n_elem; ++i) {
    mean_param(i) = gsl_vector_get(x, i);
  }

  gp_obj->SetMeanFuncParams(mean_param);

  double ret = -gp_obj->ComputeLikelihood();
  
  return ret;
}

void df_eval_mean(const gsl_vector *x, void *param, gsl_vector *g)
{
  GP *gp_obj = reinterpret_cast<GP*>(param);
  
  Col<REAL> mean_param(gp_obj->GetMeanFunction()->GetParamDim());
  for(int i=0; i < mean_param.n_elem; ++i) {
    mean_param(i) = gsl_vector_get(x, i);
  }
  gp_obj->SetMeanFuncParams(mean_param);

  Col<REAL> grad;
  gp_obj->GradLikelihoodMeanParams(grad);

  for(int i=0; i < grad.n_elem; ++i) {
    gsl_vector_set(g, i, -grad(i));
  }
}

void fdf_eval_mean(const gsl_vector *x, void *param, double *f, gsl_vector *g)
{
  *f = f_eval_mean(x, param);
  df_eval_mean(x, param, g);
}

double f_eval_kernel(const gsl_vector *x, void *param)
{
  GP *gp_obj = reinterpret_cast<GP*>(param);
  
  Col<REAL> kernel_param(gp_obj->GetKernelFunction()->GetParamDim());
  for(int i=0; i < kernel_param.n_elem; ++i) {
    kernel_param(i) = gsl_vector_get(x, i);
    if(kernel_param(i) < 1e-6) {
      return 1e6;
    }
  }
  
  gp_obj->SetKernelFuncParams(kernel_param);

  double ret = -gp_obj->ComputeLikelihood();
  
  return ret;
}

void df_eval_kernel(const gsl_vector *x, void *param, gsl_vector *g)
{
  GP *gp_obj = reinterpret_cast<GP*>(param);
  
  Col<REAL> kernel_param(gp_obj->GetKernelFunction()->GetParamDim());
  for(int i=0; i < kernel_param.n_elem; ++i) {
    kernel_param(i) = gsl_vector_get(x, i);
  }
  gp_obj->SetKernelFuncParams(kernel_param);

  Col<REAL> grad;
  gp_obj->GradLikelihoodKernelParams(grad);

  for(int i=0; i < grad.n_elem; ++i) {
    gsl_vector_set(g, i, -grad(i));
  }
}

void fdf_eval_kernel(const gsl_vector *x, void *param, double *f, gsl_vector *g)
{
  *f = f_eval_kernel(x, param);
  df_eval_kernel(x, param, g);
}

double f_eval_noise(const gsl_vector *x, void *param)
{
  GP *gp_obj = reinterpret_cast<GP*>(param);
  
  REAL noise_param;
  noise_param = gsl_vector_get(x, 0);
  if(noise_param < 1e-6) {
    return 1e6;
  }
  
  gp_obj->SetNoise(noise_param);

  double ret = -gp_obj->ComputeLikelihood();
  
  return ret;
}

void df_eval_noise(const gsl_vector *x, void *param, gsl_vector *g)
{
  GP *gp_obj = reinterpret_cast<GP*>(param);

  REAL noise_param;
  noise_param = gsl_vector_get(x, 0);

  gp_obj->SetNoise(noise_param);

  Col<REAL> grad;
  gp_obj->GradLikelihoodNoise(grad);

  for(int i=0; i < grad.n_elem; ++i) {
    gsl_vector_set(g, i, -grad(i));
  }
}

void fdf_eval_noise(const gsl_vector *x, void *param, double *f, gsl_vector *g)
{
  *f = f_eval_noise(x, param);
  df_eval_noise(x, param, g);
}


CGOptimizer::CGOptimizer(GP *gp)
{
  this->gp = gp;
}

void CGOptimizer::InitializeNoise(const REAL &init, double step)
{
  my_func.n = 1;
  my_func.f = &f_eval_noise;
  my_func.df = &df_eval_noise;
  my_func.fdf = &fdf_eval_noise;
  my_func.params = (void*)this->gp;

  x = gsl_vector_alloc(1);
  gsl_vector_set(x, 0, init);

  T = gsl_multimin_fdfminimizer_conjugate_fr;
  s = gsl_multimin_fdfminimizer_alloc (T, 1);

  gsl_multimin_fdfminimizer_set (s, &my_func, x, step, 0.1);
  //gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 1e-2);
}


void CGOptimizer::InitializeMean(const Col<REAL> &init, double step)
{
  my_func.n = init.n_elem;
  my_func.f = &f_eval_mean;
  my_func.df = &df_eval_mean;
  my_func.fdf = &fdf_eval_mean;
  my_func.params = (void*)this->gp;

  x = gsl_vector_alloc(init.n_elem);
  for(int i=0; i < init.n_elem; ++i)
    gsl_vector_set(x, i, init(i));

  T = gsl_multimin_fdfminimizer_conjugate_fr;
  s = gsl_multimin_fdfminimizer_alloc (T, init.n_elem);

  gsl_multimin_fdfminimizer_set (s, &my_func, x, step, 0.01);
  //gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 1e-2);
}

void CGOptimizer::InitializeKernel(const Col<REAL> &init, double step)
{
  my_func.n = init.n_elem;
  my_func.f = &f_eval_kernel;
  my_func.df = &df_eval_kernel;
  my_func.fdf = &fdf_eval_kernel;
  my_func.params = (void*)this->gp;

  x = gsl_vector_alloc(init.n_elem);
  for(int i=0; i < init.n_elem; ++i)
    gsl_vector_set(x, i, init(i));

  T = gsl_multimin_fdfminimizer_conjugate_fr;
  s = gsl_multimin_fdfminimizer_alloc (T, init.n_elem);

  gsl_multimin_fdfminimizer_set (s, &my_func, x, step, 0.1);
  //gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 1e-2);
}

int CGOptimizer::Optimize(Col<REAL> &soln, int max_iterations)
{
  int iter = 0;
  int status;

  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate (s);
     
    if (status) {
      printf("Dropping out of optimization _before_ gradient test\n");
      break;
    }
     
    status = gsl_multimin_test_gradient (s->gradient, 1e-3);
     
    if (status == GSL_SUCCESS)
      printf ("Minimum found at:\n");

    printf("%5d ", iter);
    for(int i=0; i < my_func.n; ++i)
      printf("%.2f ", gsl_vector_get(s->x, i));
    printf("%10.5f\n", s->f);
  }
  while (status == GSL_CONTINUE && iter < max_iterations);

  soln.set_size(my_func.n);
  for(int i=0; i < my_func.n; ++i) {
    soln(i) = gsl_vector_get(s->x, i);
  }

  printf("status = %d\n", status);
  switch(status) {
  case GSL_SUCCESS:
    printf("GSL_SUCCESS\n");
    break;
  case GSL_CONTINUE:
    printf("GSL_CONTINUE (failed on iterations)\n");
    break;
  };

  if((status == GSL_SUCCESS) || (status == GSL_CONTINUE))
    return 0;

  return -1;
}

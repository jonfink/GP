#include <gsl/gsl_multimin.h>
#include "GP.h"

#include <iostream>
using namespace std;

#include "CGOptimizer.h"

CGOptimizer::CGOptimizer(void *gp)
{
  this->gp = gp;
}

void CGOptimizer::Initialize(const Col<REAL> &init, 
			     double (*f_eval)(const gsl_vector*, void*), 
			     void (*df_eval)(const gsl_vector*, void*, gsl_vector*),
			     void (*fdf_eval)(const gsl_vector*, void*, double*, gsl_vector*),
			     double step, double eps)
{
  my_func.n = init.n_elem;
  my_func.f = f_eval;
  my_func.df = df_eval;
  my_func.fdf = fdf_eval;
  my_func.params = (void*)this->gp;

  x = gsl_vector_alloc(init.n_elem);
  for(unsigned int i=0; i < init.n_elem; ++i)
    gsl_vector_set(x, i, init(i));

  T = gsl_multimin_fdfminimizer_conjugate_fr;
  s = gsl_multimin_fdfminimizer_alloc (T, init.n_elem);

  gsl_multimin_fdfminimizer_set (s, &my_func, x, step, eps);
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
    for(unsigned int i=0; i < my_func.n; ++i)
      printf("%.2f ", gsl_vector_get(s->x, i));
    printf("%10.5f\n", s->f);
  }
  while (status == GSL_CONTINUE && iter < max_iterations);

  soln.set_size(my_func.n);
  for(unsigned int i=0; i < my_func.n; ++i) {
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

#ifndef _SPGP_H_
#define _SPGP_H_

#include "GP.h"

double f_eval_pi(const gsl_vector *x, void *param);
void df_eval_pi(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_pi(const gsl_vector *x, void *param, double *f, gsl_vector *g);

class SPGP : public GP
{
 public:
  SPGP(REAL s2_n, KernelFunction *kernel, MeanFunction *mean);
  virtual ~SPGP();

void SetDel(REAL del);
  void SetKernelFuncParams(const Col<REAL>& param);
  void SetMeanFuncParams(const Col<REAL>& param);

  void SetTraining(const Mat<REAL>& X, const Row<REAL> &y);
  void SetPseudoInputs(const Mat<REAL>& Xb);

  void Predict(const Mat<REAL> &Xs, Row<REAL> &mu);
  void Predict(const Mat<REAL> &Xs, Row<REAL> &mu, Row<REAL> &var);
  void Predict(const Col<REAL> &Xs, REAL &mu);
  void Predict(const Col<REAL> &Xs, REAL &mu, REAL &var);

  void ComputeLm();
  void ComputeBet();

  void GradLikelihoodPseudoInputs(Mat<REAL> &grad);
void OptimizePseudoInputs(Mat<REAL> &pseudoinputs, int max_iterations, double step, double eps);
  
  REAL ComputeLikelihood();

void GetPseudoInputs(Mat<REAL> &Xb);

 private:
  // State variables
  Mat<REAL> Xb; //! Pseudo input locations
  
  // Misc
  Mat<REAL> Lm, V, ep, bet;

  Mat<REAL> kxbxb, kxbx;

  bool new_training, new_mean, new_kernel, new_pi;

  unsigned int N, n, dim;
  REAL del;

};

#endif

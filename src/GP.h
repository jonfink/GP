#ifndef _GP_H_
#define _GP_H_

#include "armadillo"
using namespace arma;
#include "myTypes.h"

#include "KernelFunction.h"
#include "MeanFunction.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>

double f_eval_mean(const gsl_vector *x, void *param);
void df_eval_mean(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_mean(const gsl_vector *x, void *param, double *f, gsl_vector *g);

double f_eval_kernel(const gsl_vector *x, void *param);
void df_eval_kernel(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_kernel(const gsl_vector *x, void *param, double *f, gsl_vector *g);

double f_eval_noise(const gsl_vector *x, void *param);
void df_eval_noise(const gsl_vector *x, void *param, gsl_vector *g);
void fdf_eval_noise(const gsl_vector *x, void *param, double *f, gsl_vector *g);

class GP {
public:
  GP(REAL s2_n, KernelFunction *kernel, MeanFunction *mean);
  
  void SetTraining(const Mat<REAL>& X, const Row<REAL> &y);
  void AddTraining(const Mat<REAL>& X, const Row<REAL> &y);
  void Predict(const Mat<REAL> &Xs, Row<REAL> &mu);
  void Predict(const Mat<REAL> &Xs, Row<REAL> &mu, Row<REAL> &var);
  void Predict(const Col<REAL> &Xs, REAL &mu);
  void Predict(const Col<REAL> &Xs, REAL &mu, REAL &var);

  void PredictGradient(const Col<REAL> &Xs, Col<REAL> &grad);
  void PredictGradient(const Col<REAL> &Xs, Col<REAL> &grad, Col<REAL> &vargrad);

  void ComputeAlpha();
  void ComputeChol();
  void ComputeW();
  REAL ComputeLikelihood();

  void SetKernelFuncParams(const Col<REAL>& param);
  void SetMeanFuncParams(const Col<REAL>& param);
  void SetNoise(const REAL &s2_n);

  void GradLikelihoodMeanParams(Col<REAL> &grad);
  void GradLikelihoodKernelParams(Col<REAL> &grad);
  void GradLikelihoodNoise(Col<REAL> &grad);

  void HessianLikelihoodMeanParams(Mat<REAL> &hessian);

  void OptimizeNoiseParam(REAL &noise_param, int max_iterations=10);
  void OptimizeMeanParam(Col<REAL> &mean_param, int max_iterations=10);
  void OptimizeKernelParam(Col<REAL> &kernel_param, int max_iterations=10);

  inline Mat<REAL> &GetTrainingData() { return X; }
  inline KernelFunction *GetKernelFunction() { return this->kernel; }
  inline MeanFunction *GetMeanFunction() { return this->mean; }
  
protected:
  KernelFunction *kernel;
  MeanFunction *mean;
  Mat<REAL> K;
  Mat<REAL> W;
  Mat<REAL> X;
  Mat<REAL> L;
  Row<REAL> y;
  Row<REAL> meanvals;
  Col<REAL> alpha;
  REAL s2_n;
  REAL loglikelihood;

  bool need_to_compute_alpha;
  bool need_to_compute_chol;
  bool need_to_compute_w;

  void MatrixMap(Mat<REAL> &matrix, const Mat<REAL> &a, const Mat<REAL> &b);
};

#endif

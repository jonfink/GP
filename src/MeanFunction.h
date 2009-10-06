#ifndef _MEAN_FUNCTION_
#define _MEAN_FUNCTION_

#include "armadillo"
using namespace arma;

#include "myTypes.h"

class MeanFunction {
 public:
  virtual ~MeanFunction() {}
  virtual REAL Eval(const Col<REAL>& x)=0;
  virtual void Grad(Col<REAL> &grad, const Col<REAL>& x)=0;
  virtual void Hessian(Mat<REAL> &hessian, const Col<REAL>& x)=0;
  virtual void SetParams(const Col<REAL>& param);
  virtual Col<REAL> GetParams();
  virtual void GradX(Col<REAL> &grad, const Col<REAL>& x)=0;
  int GetParamDim();
  void SetGradMult(const Col<REAL>& grad_mult);
  
 protected:
  int param_dim;
  Col<REAL> param;
  Col<REAL> grad_mult;
};

class ConstantMean : public MeanFunction {
 public:
  ConstantMean();
  ConstantMean(const Col<REAL>& param);
  REAL Eval(const Col<REAL>& x);
  void Grad(Col<REAL> &grad, const Col<REAL>& x);
  void Hessian(Mat<REAL> &hessian, const Col<REAL>& x);
  void GradX(Col<REAL> &grad, const Col<REAL>& x);
};

class LogMean : public MeanFunction {
 public:
  LogMean();
  LogMean(const Col<REAL>& param);
  void SetParams(const Col<REAL>& param);
  REAL Eval(const Col<REAL>& x);
  void Grad(Col<REAL> &grad, const Col<REAL>& x);
  void Hessian(Mat<REAL> &hessian, const Col<REAL>& x);
  void GradX(Col<REAL> &grad, const Col<REAL>& x);
 private:
  Col<REAL> source;
};

#endif

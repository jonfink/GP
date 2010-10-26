#ifndef _KERNEL_FUNCTION_
#define _KERNEL_FUNCTION_

#include "armadillo"
using namespace arma;

#include "myTypes.h"

class KernelFunction {
 public:
  virtual ~KernelFunction() {}
  virtual REAL Eval(const Col<REAL>& a, const Col<REAL> &b)=0;
  virtual void Grad(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b)=0;
  virtual void SetParams(const Col<REAL>& param) {
    this->param = param;
  }

  virtual Col<REAL> GetParams() {
    return this->param;
  }

  virtual void GradX(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b)=0;
  int GetParamDim() { return this->param_dim; }

 protected:
  int param_dim;
  Col<REAL> param;
};

class SqExpKernel : public KernelFunction {
 public:
  SqExpKernel();
  SqExpKernel(const Col<REAL>& param);
  REAL Eval(const Col<REAL>& a, const Col<REAL> &b);
  void Grad(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b);
  void GradX(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b);
};

class SqExpCircleKernel : public KernelFunction {
 public:
  SqExpCircleKernel();
  SqExpCircleKernel(const Col<REAL>& param);
  REAL Eval(const Col<REAL>& a, const Col<REAL> &b);
  void Grad(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b);
  void GradX(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b);
};

#endif

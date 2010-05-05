#include "KernelFunction.h"

SqExpKernel::SqExpKernel()
{
  this->SetParams("1 1");
  this->param_dim = param.n_elem;
}

SqExpKernel::SqExpKernel(const Col<REAL>& param)
{
  this->SetParams(param);
  this->param_dim = param.n_elem;
}

REAL SqExpKernel::Eval(const Col<REAL>& a, const Col<REAL> &b)
{
  REAL dist = norm(a-b, 2);
  return param(0)*exp(-(dist*dist)/(2*param(1)*param(1)));
}

void SqExpKernel::Grad(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b)
{
  REAL dist = norm(a-b, 2);
  grad.set_size(param.n_elem);
  grad(0) = exp(-(dist*dist)/(2*param(1)*param(1)));
  grad(1) = dist*dist*param(0)*(1/pow(param(1),3))*exp(-(dist*dist)/(2*param(1)*param(1)));
}

void SqExpKernel::GradX(Col<REAL> &grad, const Col<REAL>& a, const Col<REAL>& b)
{
  REAL dist = norm(a-b, 2);
  grad.set_size(a.n_elem);

  if(dist < 1e-6) {
    grad.fill(0);
  }
  else {
    grad(0) = -((a(0) - b(0))*param(0))*exp(-(dist*dist)/(2*param(1)*param(1)))/
      (2*param(1)*param(1)*dist);
    grad(1) = -((a(1) - b(1))*param(0))*exp(-(dist*dist)/(2*param(1)*param(1)))/
      (2*param(1)*param(1)*dist);
  }
}

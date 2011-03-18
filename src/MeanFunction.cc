#include "MeanFunction.h"

void MeanFunction::SetParams(const Col<REAL>& param)
{
  this->param = param;
}

Col<REAL> MeanFunction::GetParams()
{
  return this->param;
}


void MeanFunction::SetGradMult(const Col<REAL>& grad_mult)
{
  this->grad_mult = grad_mult;
}

int MeanFunction::GetParamDim() { return this->param_dim; }

ConstantMean::ConstantMean()
{
  this->param.set_size(1);
  param(0) = 0;
  this->param_dim = param.n_elem;
  this->grad_mult.set_size(param.n_elem);
  this->grad_mult.fill(1.0);
}

ConstantMean::ConstantMean(const Col<REAL>& param)
{
  this->SetParams(param);
  this->param_dim = param.n_elem;
  this->grad_mult.set_size(param.n_elem);
  this->grad_mult.fill(1.0);
}

REAL ConstantMean::Eval(const Col<REAL>& x)
{
  return param(0);
}

void ConstantMean::Grad(Col<REAL> &grad, const Col<REAL>& x)
{
  grad.set_size(param.n_elem);
  grad(0) = 1;

  grad *= this->grad_mult;
}

void ConstantMean::Hessian(Mat<REAL> &hessian, const Col<REAL>& x)
{
  hessian.set_size(param.n_elem, param.n_elem);
  hessian.fill(0);
}

void ConstantMean::GradX(Col<REAL> &grad, const Col<REAL>& x)
{
  grad.set_size(x.n_elem);
  grad.fill(0);
}

LogMean::LogMean()
{
  this->SetParams("49 3 0 0");
  this->param_dim = param.n_elem;
  this->grad_mult.set_size(param.n_elem);
  this->grad_mult.fill(1.0);
}

LogMean::LogMean(const Col<REAL>& param)
{
  this->SetParams(param);
  this->param_dim = param.n_elem;
  this->grad_mult.set_size(param.n_elem);
  this->grad_mult.fill(1.0);
}

void LogMean::SetParams(const Col<REAL>& param)
{

  this->param = param;
  source.set_size(2);
  source(0) = param(2);
  source(1) = param(3);
}

REAL LogMean::Eval(const Col<REAL>& x)
{
  REAL dist = norm(source-x, 2);
  if(dist < 1e-6)
    return -param(0);
  else
    return -param(0) - 10*param(1)*log10(dist);
}

void LogMean::Grad(Col<REAL> &grad, const Col<REAL>& x)
{
  grad.set_size(param.n_elem);
  /*
  grad(0) = -1;
  grad(1) = -10*log10(norm(source-x, 2));
  grad(2) = 10*param(1)*(source(0)-x(0))/(log(10)*
					  ((source(0)-x(0))*(source(0)-x(0)) +
					   (source(1)-x(1))*(source(1)-x(1))));
  grad(3) = 10*param(1)*(source(1)-x(1))/(log(10)*
					  ((source(0)-x(0))*(source(0)-x(0)) +
					   (source(1)-x(1))*(source(1)-x(1))));
  */

  grad(0)=1;

  grad(1)=(-10*log(sqrt(pow(source(0) - x(0),2) + pow(source(1) - x(1),2))))/log(10);

  grad(2)=(-10*param(1)*(source(0) - x(0)))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  grad(3)=(-10*param(1)*(source(1) - x(1)))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  grad = grad % this->grad_mult;
}

void LogMean::Hessian(Mat<REAL> &H, const Col<REAL>& x)
{
  H.set_size(param.n_elem, param.n_elem);


  /*
  // d[grad(0)]/d[param(i)]
  H(0,0) = 0;
  H(0,1) = 0;
  H(0,2) = 0;
  H(0,3) = 0;

  // d[grad(1)]/d[param(i)]
  H(1,0) = 0;
  H(1,1) = 0;
  H(1,2) = -10*(param(2)-x(0))/
    (((param(2)-x(0))*(param(2)-x(0)) + (param(3)-x(1))*(param(3)-x(1)))*log(10));
  H(1,3) = -10*(param(3)-x(1))/
    (((param(2)-x(0))*(param(2)-x(0)) + (param(3)-x(1))*(param(3)-x(1)))*log(10));

  // d[grad(2)]/d[param(i)]
  H(2,0) = 0;
  H(2,1) = 10*(source(0)-x(0))/(log(10)*
				((source(0)-x(0))*(source(0)-x(0)) +
				 (source(1)-x(1))*(source(1)-x(1))));
  H(2,2) = 10*param(1)*
    ((param(3)-x(1))*(param(3)-x(1)) - (param(2)-x(0))*(param(2)-x(0)))/
    pow(((param(2)-x(0))*(param(2)-x(0)) + (param(3)-x(1))*(param(3)-x(1))),2);
  H(2,3) = -20*param(1)*(param(2)-x(0))*(param(3)-x(1))/
    pow(((param(2)-x(0))*(param(2)-x(0)) + (param(3)-x(1))*(param(3)-x(1))),2);

  // d[grad(3)]/d[param(i)]
  H(3,0) = 0;
  H(3,1) = 10*(source(1)-x(1))/(log(10)*
				((source(0)-x(0))*(source(0)-x(0)) +
				 (source(1)-x(1))*(source(1)-x(1))));
  H(3,2) = -20*param(1)*(param(2)-x(0))*(param(3)-x(1))/
    pow(((param(2)-x(0))*(param(2)-x(0)) + (param(3)-x(1))*(param(3)-x(1))),2);
  H(3,3) = 10*param(1)*(pow(param(2)-x(0),2) - pow(param(3)-x(1),2))/
    pow(((param(2)-x(0))*(param(2)-x(0)) + (param(3)-x(1))*(param(3)-x(1))),2);

  */

  H(0,0)=0;

  H(0,1)=0;

  H(0,2)=0;

  H(0,3)=0;

  H(1,0)=0;

  H(1,1)=0;

  H(1,2)=(-10*(source(0) - x(0)))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  H(1,3)=(-10*(source(1) - x(1)))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  H(2,0)=0;

  H(2,1)=(-10*(source(0) - x(0)))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  H(2,2)=(20*param(1)*pow(source(0) - x(0),2))/(pow(pow(source(0) - x(0),2) + pow(source(1) - x(1),2),2)*log(10)) - (10*param(1))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  H(2,3)=(20*param(1)*(source(0) - x(0))*(source(1) - x(1)))/(pow(pow(source(0) - x(0),2) + pow(source(1) - x(1),2),2)*log(10));

  H(3,0)=0;

  H(3,1)=(-10*(source(1) - x(1)))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10));

  H(3,2)=(20*param(1)*(source(0) - x(0))*(source(1) - x(1)))/(pow(pow(source(0) - x(0),2) + pow(source(1) - x(1),2),2)*log(10));

  H(3,3)=(-10*param(1))/((pow(source(0) - x(0),2) + pow(source(1) - x(1),2))*log(10)) + (20*param(1)*pow(source(1) - x(1),2))/(pow(pow(source(0) - x(0),2) + pow(source(1) - x(1),2),2)*log(10));


}


void LogMean::GradX(Col<REAL> &grad, const Col<REAL>& x)
{
  grad.set_size(x.n_elem);
  REAL dist = norm(source-x, 2);
  if(dist < 1e-6)
    grad.fill(0);
  else {
    grad(0) = -10*param(1)*(x(0)-source(0))/(dist*log(10));
    grad(1) = -10*param(1)*(x(1)-source(1))/(dist*log(10));
  }
}

#include <iostream>
#include "GP.h"

using namespace std;

int main(int argc, char **argv)
{
  Col<REAL> kernel_param = "1.0 4.0";
  SqExpKernel kernel(kernel_param);
  ConstantMean mean("10");
  GP gp(0.1, &kernel, &mean);

  Mat<REAL> X(1,10);
  Row<REAL> y;
  y.set_size(X.n_cols);

  for(int i=0; i < X.n_cols; ++i) {
    X(0,i) = i;
    y(i) = 10+sin(i/(2*Math<REAL>::pi()*X.n_cols));
  }

  Mat<REAL> Xs = X;
  Row<REAL> mu, var;
  REAL mu1, var1;

  cout << "Setting training data\n";
  gp.SetTraining(X, y);
  cout << "Making predictions\n";
  gp.Predict(Xs, mu, var);
  // gp.Predict(Xs.col(0), mu1, var1);

  Col<REAL> grad;
  gp.GradLikelihoodKernelParams(grad);
  grad.print("Likelihood Gradient (kernel)");

  cout << "X:\n" << X << endl;
  cout << "y:\n" << y << endl;
  cout << "Xs:\n" << Xs << endl;
  cout << "mu:\n" << mu << endl;

  Row<REAL> error = y - mu;
  cout << "error:\n" << norm(error,2) << endl;

  cout << "Log-likelihood: " << gp.ComputeLikelihood() << endl;

  return 0;
}

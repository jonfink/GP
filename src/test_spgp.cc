#include <iostream>
#include <fstream>

#include "SPGP.h"

using namespace std;

int main(int argc, char **argv)
{
  Col<REAL> kernel_param = "1.0 2.0";
  SqExpKernel kernel(kernel_param);
  ConstantMean mean("0");
  SPGP gp(0.1, &kernel, &mean);
  
  Mat<REAL> X(1,200);
  Row<REAL> y;
  y.set_size(X.n_cols);
  
  for(int i=0; i < X.n_cols; ++i) {
    X(0,i) = (2.0*Math<REAL>::pi())*(REAL)i/(REAL)X.n_cols;
    y(i) = 1*sin(X(0,i));
  }
  
  Mat<REAL> Xb(1,X.n_cols/10);
  for(int i=0; i < Xb.n_cols; ++i) {
    Xb(0,i) = (0.25*Math<REAL>::pi())*(REAL)i/(REAL)Xb.n_cols;
  }

  gp.SetPseudoInputs(Xb);

  Mat<REAL> Xs(1,400);
  for(int i=0; i < Xs.n_cols; ++i) {
    Xs(0,i) = (2.0*Math<REAL>::pi())*(REAL)i/(REAL)X.n_cols;
  }

  Row<REAL> mu, var;
  REAL mu1, var1;

  cout << "Setting training data\n";
  gp.SetTraining(X, y);
  cout << "Making predictions\n";
  gp.Predict(Xs, mu, var);
  //gp.Predict(Xs.col(0), mu1, var1);

  /*
  Col<REAL> grad;
  gp.GradLikelihoodKernelParams(grad);
  grad.print("Likelihood Gradient (kernel)");
  */

  cout << "X:\n" << X << endl;
  cout << "y:\n" << y << endl;
  cout << "Xb:\n" << Xb << endl;
  cout << "Xs:\n" << Xs << endl;
  cout << "mu:\n" << mu << endl;

  // Output csv file for plotting


  ofstream out("test.txt");
  for(int i=0; i < Xs.n_cols; ++i) {
    out << Xs(0, i) << ", " << mu(i) << ", " << var(i) << "\n";
  }
  out.close();

  out.open("training.txt");
  for(int i=0; i < X.n_cols; ++i) {
    out << X(0, i) << ", " << y(i) << "\n";
  }
  out.close();

  out.open("pseudoinputs.txt");
  for(int i=0; i < Xb.n_cols; ++i) {
    out << Xb(0, i) << ", 0\n";
  }
  out.close();

  /*
  Row<REAL> error = y - mu;
  cout << "error:\n" << norm(error,2) << endl;
  */
  cout << "Log-likelihood: " << gp.ComputeLikelihood() << endl;

  gp.OptimizePseudoInputs(Xb, 20, 10, 0.001);
  gp.SetPseudoInputs(Xb);  
  cout << "Xb*:\n" << Xb << endl;

  cout << "Log-likelihood: " << gp.ComputeLikelihood() << endl;  

  gp.Predict(Xs, mu, var);
  
  out.open("test2.txt");
  for(int i=0; i < Xs.n_cols; ++i) {
    out << Xs(0, i) << ", " << mu(i) << ", " << var(i) << "\n";
  }
  out.close();

  out.open("pseudoinputs2.txt");
  for(int i=0; i < Xb.n_cols; ++i) {
    out << Xb(0, i) << ", 0\n";
  }
  out.close();

  for(int i=0; i < Xb.n_cols; ++i) {
    Xb(0,i) = (2.0*Math<REAL>::pi())*(REAL)i/(REAL)Xb.n_cols;
  }

  gp.SetPseudoInputs(Xb);
  gp.Predict(Xs, mu, var);
  cout << "Log-likelihood: " << gp.ComputeLikelihood() << endl;  

  out.open("test3.txt");
  for(int i=0; i < Xs.n_cols; ++i) {
    out << Xs(0, i) << ", " << mu(i) << ", " << var(i) << "\n";
  }
  out.close();

  out.open("pseudoinputs3.txt");
  for(int i=0; i < Xb.n_cols; ++i) {
    out << Xb(0, i) << ", 0\n";
  }
  out.close();

  gp.OptimizePseudoInputs(Xb, 20, 10, 0.001);
  gp.SetPseudoInputs(Xb);  

  gp.Predict(Xs, mu, var);
  cout << "Log-likelihood: " << gp.ComputeLikelihood() << endl;  

  out.open("test4.txt");
  for(int i=0; i < Xs.n_cols; ++i) {
    out << Xs(0, i) << ", " << mu(i) << ", " << var(i) << "\n";
  }
  out.close();

  out.open("pseudoinputs4.txt");
  for(int i=0; i < Xb.n_cols; ++i) {
    out << Xb(0, i) << ", 0\n";
  }
  out.close();

  return 0;
}

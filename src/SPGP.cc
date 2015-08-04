#include <iostream>
using namespace std;

#include "SPGP.h"
#include "CGOptimizer.h"
#include "armadillo_backsub.h"

SPGP::SPGP(REAL s2_n, KernelFunction *kernel, MeanFunction *mean)
  : GP(s2_n, kernel, mean)
{
  // Nothing special to do here

  del = 1e-6;
}

SPGP::~SPGP()
{

}

void SPGP::
SetDel(REAL del)
{
  this->del = del;
}

void SPGP::
SetKernelFuncParams(const Col<REAL>& param)
{

  GP::SetKernelFuncParams(param);
  this->new_kernel = true;
}

void SPGP::
SetMeanFuncParams(const Col<REAL>& param)
{
  GP::SetMeanFuncParams(param);
  this->new_mean = true;
}

void SPGP::
SetPseudoInputs(const Mat<REAL>& Xb)
{
  this->Xb = Xb;
  MatrixMap(kxbxb, Xb, Xb);

  this->n = Xb.n_cols;

  this->new_pi = true;
}

void SPGP::
SetTraining(const Mat<REAL>& X, const Row<REAL> &y)
{
  this->X = X;
  this->y = y;

  this->N = X.n_cols;
  this->dim = X.n_rows;

  this->meanvals.set_size(X.n_cols);

  this->new_training = true;
}

void SPGP::
ComputeLm()
{
  if(new_training || new_pi || new_kernel) {
    MatrixMap(kxbx, Xb, X);
  }

  if(new_pi || new_kernel) {
    MatrixMap(kxbxb, Xb, Xb);
  }

  //kxbx.print("kxbx: ");
  //kxbxb.print("kxbxb: ");

  //printf("Computing L\n");
  L = trans(chol(kxbxb + del*eye<Mat<REAL> >(kxbxb.n_cols, kxbxb.n_cols)));
  //L.print("L: ");

  solve_tri(V, L, kxbx, false);
  
  //V.print("V: ");
  //V = trans(V);

  Col<REAL> tmp(this->dim);
  tmp.zeros();

  ep = 1 + (this->kernel->Eval(tmp, tmp) - trans(sum(V%V)))/s2_n;

  //ep.print("ep:");

  V = V / (repmat(trans(sqrt(ep)), n, 1));

  //printf("Computing Lm\n");
  Lm = trans(chol(s2_n*eye<Mat<REAL> >(n, n) + V*trans(V)));

  //Lm.print("Lm");
}

void SPGP::
ComputeBet()
{
  if(new_mean || new_training) {
    for(unsigned int i=0; i < this->N; ++i)
      this->meanvals(i) = this->mean->Eval(X.col(i));
  }

  Col<REAL> ytmp = trans(y-meanvals)/sqrt(ep);

  bet.zeros(Lm.n_cols, ytmp.n_cols);

  //bet.print("bet (init)");
  Mat<REAL> tmp = V*ytmp;
  //tmp.print("V*ytmp");

  solve_tri(bet, Lm, V*ytmp, false);

  //bet.print("bet");
}

REAL SPGP::ComputeLikelihood()
{
  ComputeLm();
  ComputeBet();

  Mat<REAL> tmp;
  loglikelihood = 0;

  tmp = sum(log(Lm.diag()));
  //tmp.print("sum(log(Lm.diag()))");

  loglikelihood += tmp(0,0);

  tmp = (dot(y,y) - dot(bet, bet) + sum(log(ep)))/2;
  //tmp.print("(dot(y,y) - dot(bet, bet) + sum(log(ep)))/2");

  loglikelihood += tmp(0,0);

  /*
  loglikelihood = sum(log(Lm.diag())) +
    (X.n_cols - Xb.n_cols)/2*log(s2_n) +
    (dot(y,y) - dot(bet, bet) + sum(log(ep)))/2 +
    0.5*X.n_cols*log(2*Math<REAL>::pi());
  */

  loglikelihood += (X.n_cols - Xb.n_cols)/2*log(s2_n) +
    0.5*X.n_cols*log(2*Math<REAL>::pi());

  return loglikelihood;
}

void SPGP::Predict(const Mat<REAL> &Xs, Row<REAL> &mu)
{
  ComputeLm();
  ComputeBet();

  Mat<REAL> kstar(Xb.n_cols, Xs.n_cols);
  //kstar.print("kstar(init)");
  MatrixMap(kstar, Xb, Xs);
  //kstar.print("kstar");
  Mat<REAL> lst;
  solve(lst, trimatu(L), kstar);
  //lst.print("lst");
  Mat<REAL> lmst;
  solve(lmst, trimatu(Lm), lst);
  //lmst.print("lmst");

  Row<REAL> meanxs(Xs.n_cols);
  for(unsigned int i=0; i < Xs.n_cols; ++i)
    meanxs(i) = this->mean->Eval(Xs.col(i));

  mu = trans(bet*lmst) + meanxs;
}

void SPGP::Predict(const Mat<REAL> &Xs, Row<REAL> &mu, Row<REAL> &var)
{
  ComputeLm();
  ComputeBet();

  Mat<REAL> kstar(Xb.n_cols, Xs.n_cols);
  //kstar.print("kstar(init)");
  MatrixMap(kstar, Xb, Xs);
  //kstar.print("kstar");
  Mat<REAL> lst;
  solve_tri(lst, L, kstar, false);
  //lst.print("lst");
  Mat<REAL> lmst;
  solve_tri(lmst, Lm, lst, false);
  //lmst.print("lmst");

  Row<REAL> meanxs(Xs.n_cols);
  for(unsigned int i=0; i < Xs.n_cols; ++i)
    meanxs(i) = this->mean->Eval(Xs.col(i));

    //printf("trans(bet): %d x %d, lmst: %d x %d \n", bet.n_cols, bet.n_rows, lmst.n_rows, lmst.n_cols);
  mu = trans(bet)*lmst + meanxs;

    //mu.print("mu");

  Col<REAL> tmp(this->dim);
  tmp.zeros();
  var = this->kernel->Eval(tmp, tmp) -
    sum(lst % lst) +
    s2_n*sum(lmst % lmst);
}


void SPGP::Predict(const Col<REAL> &Xs, REAL &mu)
{
  Mat<REAL> XsMat(Xs.n_rows, 1);
  XsMat.col(0) = Xs;

  Predict(XsMat, mu);
}

void SPGP::Predict(const Col<REAL> &Xs, REAL &mu, REAL &var)
{
  Mat<REAL> XsMat(Xs.n_rows, 1);
  XsMat.col(0) = Xs;

  Predict(XsMat, mu, var);
}

void SPGP::
GradLikelihoodPseudoInputs(Mat<REAL> &grad)
{
  // note: we make some assumptions here about the kernel function being sq-exp
  ComputeLm();
  ComputeBet();

  Mat<REAL> Q = kxbxb + del*eye<Mat<REAL> >(kxbxb.n_cols, kxbxb.n_cols);

  Mat<REAL> K = kxbx;
  K = K / (repmat(trans(sqrt(ep)), n, 1));

  Mat<REAL> invLmV;
  solve(invLmV, trimatu(Lm), V);

  Mat<REAL> Lt = L*Lm;
  Mat<REAL> B1; solve(B1, trimatl(trans(Lt)), invLmV);
  Col<REAL> b1; solve(b1, trimatl(trans(Lt)), bet);
  Mat<REAL> invLV; solve(invLV, trimatl(trans(L)), V);
  Mat<REAL> invL = inv(L);
  Mat<REAL> invQ = trans(invL)*invL;
  Mat<REAL> invLt = inv(Lt);
  Mat<REAL> invA = trans(invLt)*invLt;

  Col<REAL> mu; solve(mu, trimatl(trans(Lm)), bet); mu = trans(trans(mu)*V);
  Col<REAL> sumVsq = trans(sum(V%V));

  Col<REAL> bigsum = trans(y) % trans((trans(bet)*invLmV))/s2_n -
    trans(sum(invLmV % invLmV))/2 -
    (trans(y % y) + mu % mu)/2/s2_n + 0.5;

  Mat<REAL> TT = invLV*(trans(invLV)%repmat(bigsum, 1, n));

  Mat<REAL> dnnQ, dNnK, epdot;
  Col<REAL> epPmod;
  Mat<REAL> dfxb(n, dim);

  Col<REAL> kernel_param = this->kernel->GetParams();

  for(unsigned int i=0; i < this->dim; ++i) {
    dnnQ = (repmat(trans(Xb.row(i)), 1, n) - repmat(Xb.row(i), n, 1)) % Q;
    dNnK = (repmat(X.row(i), n, 1) - repmat(trans(Xb.row(i)), 1, N)) % K;

    epdot = -2/s2_n*(dNnK % invLV);
    epPmod = trans(-sum(epdot));

    dfxb.col(i) = -b1 % (dNnK*(trans(y)-mu)/s2_n + dnnQ*b1)
      + sum((invQ - invA*s2_n)%dnnQ, 1)
      + epdot*bigsum - 2/s2_n*sum(dnnQ % TT, 1);

    dfxb.col(i) = dfxb.col(i)/sqrt(kernel_param(1));
  }

  grad = dfxb;
}

void SPGP::GetPseudoInputs(Mat<REAL> &Xb)
{
  Xb.set_size(this->Xb.n_rows, this->Xb.n_cols);
  //Xb = Mat<REAL>(this->Xb);
  Xb = this->Xb;
}

void SPGP::
OptimizePseudoInputs(Mat<REAL> &pseudoinputs, int max_iterations, double step, double eps)
{
  CGOptimizer opt(reinterpret_cast<void*>(this));

  Col<REAL> opt_state(pseudoinputs.n_rows*pseudoinputs.n_cols);
  int idx=0;
  for(unsigned int i=0; i < pseudoinputs.n_rows; ++i) {
    for(unsigned int j=0; j < pseudoinputs.n_cols; ++j) {
      opt_state(idx++) = pseudoinputs(i, j);
    }
  }

  opt.Initialize(opt_state, &f_eval_pi, &df_eval_pi, &fdf_eval_pi, step, eps);
  opt.Optimize(opt_state, max_iterations);

  idx=0;
  for(unsigned int i=0; i < pseudoinputs.n_rows; ++i) {
    for(unsigned int j=0; j < pseudoinputs.n_cols; ++j) {
      pseudoinputs(i, j) = opt_state(idx++);
    }
  }
}

double f_eval_pi(const gsl_vector *x, void *param)
{
  SPGP *gp_obj = reinterpret_cast<SPGP*>(param);

  Mat<REAL> pinput; gp_obj->GetPseudoInputs(pinput);
  unsigned int dim = pinput.n_cols;
  unsigned int n = pinput.n_rows;

  int k=0;
  for(unsigned int i=0; i < n; ++i) {
    for(unsigned int j=0; j < dim; ++j) {
      pinput(i, j) = gsl_vector_get(x, k++);
    }
  }

  gp_obj->SetPseudoInputs(pinput);

  double ret = gp_obj->ComputeLikelihood();

  return ret;
}

void df_eval_pi(const gsl_vector *x, void *param, gsl_vector *g)
{

  SPGP *gp_obj = reinterpret_cast<SPGP*>(param);

  Mat<REAL> pinput; gp_obj->GetPseudoInputs(pinput);
  unsigned int dim = pinput.n_cols;
  unsigned int n = pinput.n_rows;

  int k=0;
  for(unsigned int i=0; i < n; ++i) {
    for(unsigned int j=0; j < dim; ++j) {
      pinput(i, j) = gsl_vector_get(x, k++);
    }
  }

  gp_obj->SetPseudoInputs(pinput);

  Mat<REAL> grad;
  gp_obj->GradLikelihoodPseudoInputs(grad);

  k=0;
  for(unsigned int i=0; i < n; ++i) {
    for(unsigned int j=0; j < dim; ++j) {

      gsl_vector_set(g, k++, -grad(j, i));

    }
  }
}

void fdf_eval_pi(const gsl_vector *x, void *param, double *f, gsl_vector *g)
{
  *f = f_eval_pi(x, param);
  df_eval_pi(x, param, g);
}

Gaussian Process Libary
=======================

This library provides a C++ implementation of Gaussian process regression as described in ["Gaussian Processes for Machine Learning"](http://www.gaussianprocess.org/gpml/) by Carl Edward Rasmussen and Christopher K. I. Williams.  The design goal of the software is to provide an easy interface with fast performance by using efficient wrappers around low-level LAPACK code.

Dependencies
------------

* [Armadillo](http://arma.sourceforge.net/) is a C++ linear algebra library with LAPACK integration.
* [Gnu Scientific Library (GSL)](http://www.gnu.org/software/gsl/) provides some non-linear optimization routines for doing hyper-parameter estimation.

Building
--------
Standard cmake build..

     $ mkdir build && cd build
     $ cmake ..
     $ make

Basic Usage
-----------
An example is given in 'src/test.cc'

     #include <iostream>
     #include "GP.h"

     using namespace std;

     int main(int argc, char **argv)
     {
       Col<REAL> kernel_param = "1.0 4.0";
       SqExpKernel kernel(kernel_param);
       ConstantMean mean("10");
       GP gp(0.1, &kernel, &mean);

       Mat<REAL> X(1,1000);
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

License
-------
Code is licensed under the BSD license.

Copyright (c) 2011, Jon Fink
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

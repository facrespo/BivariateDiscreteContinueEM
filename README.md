# BivariateProbitContinueEM

This repository has the function and examples to run the Bivariate Model with one variable discrete and the other as continue variable using 
the EM-Algorihm. For the Univariate Probit using the EM-Algorithm applied in Bioinformatics, and for the continue variable use the classical linear regression.

We use the libraries: Numpy, Scipy, Sympy, Math, statsmodels.api, and Python 3.5 with Anaconda. To down statsmodels, you should visit: http://statsmodels.sourceforge.net/stable/ and use the suggestions.

The emoneProbitf.py is the function to univariate Probit calculated with EM algorithm with multivariate X predictor.

The embiprobitcontinue.py is the function to estimate the bivariate discrete and continue outputs calculated with EM algorithm with multivariate X predictor.

The classicmodel.py is the function to Linear Clasiccal Model with multivariate X predictor.

The cargar_archivo_sim_discont_02.py is one ejecution using the files generate for generar_aleatorios2.R (This file use R), the result is write in resultado_cont_dist_02.txt. The constant is agregated in X with function X = sm.add_constant(X, prepend=False).


# BivariateProbitContinueEM

This repository has the function and examples to run the Bivariate Model with one variable discrete and the other as continue variable using 
the EM-Algorihm. For the Univariate Probit using the EM-Algorithm applied in Bioinformatics, and for the continue variable use the classical linear regression.

We use the libraries: Numpy, Scipy, Sympy, Math, statsmodels.api, and Python 3.5 with Anaconda. To down statsmodels, you should visit: http://statsmodels.sourceforge.net/stable/ and use the suggestions.

The emoneProbitf.py is the function to univariate Probit calculated with EM algorithm with multivariate X predictor.

The embiprobitf.py is the function to bivariate Probir calculated with EM algorithm with multivariate X predictor.

The classicmodel.py is the function to Linear Clasiccal Model with multivariate X predictor.

The sample_green.py is one ejecution using the file ejemplo_greenvf.csv, the result is show in screen. The constant is agregated in X with function X = sm.add_constant(X, prepend=False).

cargar_archivo_prueba_bicontinuo4.py is an example to calculate models with SNPs, it use the file simulation.txt. simulation.txt has the variables: total.status, total.tgres, total-rs3115860_2. We get total.status as Y1, total.tgres as Y2, and X with total.rs3115860_2. For this cases cargar_archivo_prueba_3.py make 9 models with SNps and one model without SNPs. The idea is proof if the SNPs generate a better models that without theirs. The output of cargar_archivo_prueba_bicontinuo4.py are resultado_continue4.txt and timestage_continue4.txt. In resultado_continue4.txt appear the model we get its. In timestage_continue4.txt appear the running times.

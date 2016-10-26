"This file is working to read the simulation files and running 
"the model to discrete and continue outputs, and the result
" is writing in the resultado_cont_dist_02.txt

import numpy as np;
import pickle;
import csv;
import io;
import six;
#import pandas;
#from pandas.io.parsers import read_csv;
#from io import StringIO;

#import sys;
#import builtins;

from scipy import stats;
import matplotlib as plt;
import statsmodels.api as sm;
import statsmodels.formula.api as smf;
from statsmodels.formula.api import logit, probit, poisson, ols;
import emoneProbitf;
import embiprobitcontinue;
import classicmodel;
from time import time;
#from emoneProbit import emProbit;

n=400;
r=1;
p=1;
error=0.01;
maxiter=80;
perturbation=0.000001;
alpha=0.05;
delta_grid=0.1;
ipower=0; #is 1 if you like to get the power function
abseps=1e-6; #Tolerance to normal bivariate distribution
p_special_test=1;
lb=1000;

fich=open("resultado_cont_dist_02.txt","w");
fich.write("Model,Correlation,number_iteration_bivariate,B1[1],B10_constant,B2[1],B20_constant,ErrorVariable1,ErrorVariable2,CorrelationEstimateReal1,CorrelationEstimateReal2,logwconstant,logmodelcomplet,logwSNP,pseudorsquared,test_log_likehood,p-value,Desesv_B1[1],Desesv_B2[1],Desv_rho,Test_B1[1],Test_B2[1],pvalue_B1[1],pvalue_B2[1],testcortebivariateconstantH01,estadisticobivariateconstantH01,pvaluebivariateconstantH01, testcortebivariateSNPH01,estadisticobivariateSNPH01, pvaluebivariateSNPH01,  logverotsy1, tcortepspecialsy1, testpspecialsy1, pvaluepspecialsy1, logverotsy2, tcortepspecialsy2, testpspecialsy2, pvaluepspecialsy2, number_iteration_modelY1,B1[1],B1_constant,ErrorVariable1,CorrelationEstimateRealY1,lognull,logverwsnpm1,logmodel,pseudorsquared,test_likehood,p-value,Desesv_B1[1],Test_B1[1],pvalue_B1[1],testcorteY1constantH01,estadisticoY1constantH01,pvalueY1constantH01,testcorteY1SNPH01,estadisticoY1SNPH01,pvalueY1SNPH01,number_iteration_modelY2,B2[1],B2_constant,ErrorVariable2,CorrelationEstimateRealY2,lognull2,logverwsnpm2,logmodel2,pseudorsquared2,test_likehood2,p-value2,Desesv_B2[1],Test_B2[1],pvalue_B2[1],testcorteY2constantH01,estadisticoY2constantH01,pvalueY2constantH01,testcorteY2SNPH01,estadisticoY2SNPH01,pvalueY2SNPH01\n");
fich.close();

for l in range(0,lb):
        print(l);
        if (l!=1001):
           datos= csv.reader(open("sim02b" + str(l) +".txt"),delimiter=",");
           datosn=np.zeros((n,p+2));

           for i,fila in enumerate(datos):
               for j in range(0,p+2):
                   datosn[i-1,j]=float(fila[j]);

           datosn=np.asmatrix(datosn);

           Y1=np.zeros((n,r));
           Y2=np.zeros((n,r));
           X=np.zeros((n,p));
           Y1=datosn[0:n,0];
           Y2=datosn[0:n,1];
           X[0:n,]=datosn[0:n,(p+1)];
           Y1=np.matrix(Y1);
           Y2=np.matrix(Y2);
           Xcont = sm.add_constant(X, prepend=False);
           #Xcont=X;
           X=np.asmatrix(Xcont);

           rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logmodelocompleto, logmwSNP, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2 = embiprobitcontinue.embiprobitcontinue(Y1, Y2, X, maxiter, error, perturbation, alpha, abseps, p_special_test);

           iteracionesm1, Bponderadorm1, vprediccionm1, error_prediccionm1, Rsquarem1, lognullm1, logverwsnpm1, logverofm1, scorefm1, pseudorsquaredm1, llrfm1, llrf_pvaluem1, effect_marginalm1, Desv_finalm1, Tfinalm1, P_valuem1, tcortem1, testm1, pvaluem1, tcortepspecialm1, testpspecialm1, pvaluepspecialm1 = emoneProbitf.emProbit(Y1, X, maxiter, error, perturbation, alpha, p_special_test);

           Bponderadorm2, vprediccionm2, error_prediccionm2, Rsquarem2, lognullm2, logverwsnpm2, logverofm2, scorefm2, pseudorsquaredm2, llrfm2, llrf_pvaluem2, effect_marginalm2, Desv_finalm2, Tfinalm2, P_valuem2, tcortem2, testm2, pvaluem2, tcortepspecialm2, testpspecialm2, pvaluepspecialm2 = classicmodel.linealmodel(Y2, X, maxiter, error, perturbation, alpha, p_special_test);

           fich=open("resultado_cont_dist_02.txt","a");
           fich.write("sim02b" + str(l) +".txt" + "," + str(rho) + "," + str(iteracion) + "," + str(Btt1[0,0]) + ","  + str(Btt1[1,0]) + ",");
           fich.write(str(Btt2[0,0]) + "," + str(Btt2[1,0]) + "," + str(error_predicciont1[0,0]) + ",");
           fich.write(str(error_predicciont2[0,0]) + "," + str(Rsquare1) + "," + str(Rsquare2) + "," + str(lognull) + "," + str(logmodelocompleto) + "," + str(logmwSNP) + ",");
           fich.write(str(pseudorsquared) + "," + str(llrf) + "," + str(llrf_pvalue) + "," +  str(Desv_b1[0,0]) + ",");
           fich.write(str(Desv_b2[0,0]) + ",");
           fich.write(str(Desv_rho[0]) + ",");
           fich.write(str(Tfinal1[0,0]) + "," + str(Tfinal2[0,0]) + ",");
           fich.write(str(P_value1[0,0]) + "," + str(P_value2[0,0]) + ",");
           fich.write(str(tcortemb) + "," + str(testmb) + "," + str(pvaluemb) + "," +  str(tcortepspecial) + "," + str(testpspecial) + "," + str(pvaluepspecial) + ",");
           fich.write(str(logverotppy1) + "," + str(tcortepspecialy1) + "," + str(testpspecialy1) + "," + str(pvaluepspecialy1) + "," + str(logverotppy2) + "," + str(tcortepspecialy2) + "," + str(testpspecialy2) + "," + str(pvaluepspecialy2) + ",");              
           fich.write(str(iteracionesm1) + "," + str(Bponderadorm1[0,0]) + "," + str(Bponderadorm1[1,0]) + "," + str(error_prediccionm1[0,0]) + ",");
           fich.write(str(Rsquarem1) + "," + str(lognullm1) + "," + str(logverwsnpm1) + "," + str(logverofm1) + "," + str(pseudorsquaredm1) + "," + str(llrfm1) + "," + str(llrf_pvaluem1) + ",");
           fich.write(str(Desv_finalm1[0,0]) + "," + str(Tfinalm1[0,0]) + ",");
           fich.write(str(P_valuem1[0,0]) + ",");
           fich.write(str(tcortem1) + "," + str(testm1) + "," + str(pvaluem1) + "," +  str(tcortepspecialm1) + "," + str(testpspecialm1) + "," + str(pvaluepspecialm1) + ",");
           fich.write(str(0) + "," + str(Bponderadorm2[0,0]) + ","  + str(Bponderadorm2[1,0]) + "," + str(error_prediccionm2[0,0]) + ",");
           fich.write(str(Rsquarem2) + "," + str(lognullm2) + "," + str(logverwsnpm2) + ","  + str(logverofm2) + "," + str(pseudorsquaredm2) + "," + str(llrfm2) + "," + str(llrf_pvaluem2) + ",");
           fich.write(str(Desv_finalm2[0,0]) + ","  + str(Tfinalm2[0,0]) + ",");
           fich.write(str(P_valuem2[0,0]) + ",");
           fich.write(str(tcortem2) + "," + str(testm2) + "," + str(pvaluem2) + "," +  str(tcortepspecialm2) + "," + str(testpspecialm2) + "," + str(pvaluepspecialm2) + "\n");                         
           fich.close();
        else:
           print("No va"); 


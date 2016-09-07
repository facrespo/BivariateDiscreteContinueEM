''' emProbit(Y, X, maxiter=100, error=0.01, perturbation=0.001, alpha=0.05, p_special_test=1):
Parameters Y an Bivariate Vector with 0 or 1 with the same raws with X
X is the explicative variables, the constant is colocated in the Execution Problems.
maxiter is the maximal iteration number for default is 100.
Error is the error of tolerance to stop the look for the loglikehood in the EM Algorithm. For default is  0.01.
perturbation is a variable to perturb the Matrix X^tX if you like invert its. The default value is 0.001.
alpha is the value to look for the cut point of each test, for default is 0.05.
p_special_test is the variable p of the matrix X that we like to get the multivariate p-value, for default is 1.

Results:

iteracion is the number of iteration that Em Algorithm use to look for the optimal.
Bt is the ponderators of X.
vt is the predicción to Y, if vt[i]>0 then Y[i] estimate is 1.
error_predicciont the cuadratic difference of error of Y respect the Y estimate,
Rsquare the correlation of Y estimate with Y.
lognull is the loglikehood only the constant or last variable.
logmreducp is the loglikehood without the variable p_special_test. 
logverot is the loglikehood using the complete X.
scoret is the variation of ponderators, this function appear in (Green, 2008).
pseudorsquared is the value calculed as proportion of loglikehood (Green, 2008).  
llrf is the test of -2 diference of likelihood complete model less model with constant only
llrf_pvalue is the p-value of llrf
marginalt is the marginal value that appear in (Green, 2008).
Desv_final is the standard desviation of each ponderator.
Tfinal is the value ponderator/standard desviation for each ponderator
P_value is the p-value of each test using the t-student distribution as linear standard model,  
tcorte1 is the cut point of multilinneal test to the model without constant,
testm1 is the test value of multilinneal test to the model without constant
pvaluem1 is the p-value of testm1
tcortespecial is the cut point of multilinneal test to the model without p_special_test variable.
testspecial is the test value of multilinneal test to the model without p_special_test variable.
pvalpspecial is the p-value of testspecial value.


Greene, W. (1998). Econometric Analysis, Third edition. Prentice-Hall,
New Jersey.

'''

import numpy as np;
import numpy.linalg as la;
import scipy.stats;
from scipy import stats, special, optimize;
from scipy.stats import norm, chisqprob;
import scipy.linalg as scl;
import math;
#import statsmodels.api as sm;

FLOAT_EPS = np.finfo(float).eps;

def calculo_B(X, v, Omega, p):
    H=np.dot(Omega,np.dot(np.transpose(X),v));
    H1=scl.pinv(0*np.identity(p)+np.dot(Omega,np.dot(np.dot(np.transpose(X),X),Omega)));
    D1=np.transpose(np.diagonal(H1));
    B=np.dot(Omega,np.dot(H1,H));
    return B, D1, H1;

def diagonal_beta(p,beta):
    I=np.identity(p);
    for i in range(0,p):
        I[i,i]=abs(beta[i]);
    return I;

def loglike(error, n, p):
    value=-1*(n/2)*(1+np.log(2*np.pi)+np.log(error/n));
    return value[0,0];

def marginal_score(qq, x):
    return qq*stats.norm._cdf(qq*x)/np.clip(stats.norm._cdf(qq*x), FLOAT_EPS, 1 - FLOAT_EPS);
    
f_m_score = np.vectorize(marginal_score);

def score(Y, X, Xb, q, n, p):
        L=np.zeros((n, 1));
        L=f_m_score(q,Xb);
        return np.asmatrix(np.transpose(np.dot(np.transpose(L),X)));

def m_marginal(x, bb):
        return stats.norm._pdf(x)*np.transpose(bb);
        
f_m_marginal = np.vectorize(m_marginal);

def marginal(Xb, beta, n, p):
        suma=np.zeros((p,1));
        L=np.zeros((n,p));
        L=f_m_marginal(Xb,np.ones((n,1))*np.transpose(beta));
        suma=np.asmatrix(np.mean(L,0));
        return np.transpose(suma);

def tstudent(Z, n, p):
    return 2*(1.0 - scipy.stats.t(n-p-1).cdf(Z));  


def prsquared(lognull,logverot):
    return 1 - (logverot/lognull);


def llr(lognull,logverot):
    return -2*(lognull - logverot);

  
def llr_pvalue(X,llrf):
    df_model = float(la.matrix_rank(X) - 1);
    return stats.chisqprob(llrf, df_model);


def test_additional_predictors2(logver, logvernull, n, p, alpha, error, delta_grid, ipower):
    loglambbda=(logvernull)-(logver);
    r=p-1;
    q=p-2;
    m=1;
    estadistico=-2*loglambbda;
    gl=m*(r-q);
    tcorte=stats.chi2.ppf(1-(alpha/2), gl, loc=0, scale=1);
    pvalue2=stats.chi2.sf((estadistico), gl, loc=0, scale=1);
    return tcorte, estadistico, pvalue2;


def calculo_error(Y, Yt, n, p):
    YM=Y-Yt;
    suma=np.dot(YM.transpose(),YM);
    return suma;        

def desviaciones_betas(Y, H, errorr, n, p):
    suma=np.zeros((p, p));
    suma=(errorr[0,0]/n)*H;
    desviaciones=np.zeros((p,1));
    std_error=np.asmatrix(np.sqrt(np.diag(np.abs(suma))));
    desviaciones=np.transpose(std_error);
    return desviaciones;        

def linealmodel(Y, X, maxiter=100, error=0.01, perturbation=0.001, alpha=0.05, p_special_test=1):
    n=X.shape[0];
    p=X.shape[1];
    Y=np.matrix(Y);
    X=np.matrix(X);
    Omega=np.identity(1);
    Bnull, D0, H0 = calculo_B(np.ones((n,1)), Y, Omega, 1);
    v0 = np.asmatrix(np.dot(np.ones((n,1)),Bnull));
    error0=calculo_error(Y, v0, n, p);
    lognull=loglike(error0, n, p);
    Omega=np.identity(p);
    Bt, Dt, Ht = calculo_B(X, Y, Omega, p);
    vt = np.asmatrix(np.dot(X,Bt));
    error_predicciont=calculo_error(Y, vt, n, p);
    logverot=loglike(error_predicciont, n, p);
    marginalt=marginal(vt, Bt, n, p);
    scoret=score(Y, X, vt, np.ones((n,1)), n, p);
    Desv_final=desviaciones_betas(Y, Ht, error_predicciont, n, p);
    XN=np.zeros((n, 2));
    Tfinal=np.zeros((p, 1));
    P_value=np.zeros((p, 1));
         
    for i in range(0,p):
         if (abs(Desv_final[i])<= error):
             Tfinal[i]=Bt[i]*(1e100);
             P_value[i] = 0;
         else:
             Tfinal[i]=(Bt[i])/(Desv_final[i]);
             P_value[i] = tstudent(np.abs(Tfinal[i]), n, p);
  
    XN=np.asmatrix(np.concatenate((vt,Y),1).transpose());
    Corr=(np.corrcoef(XN));
    Rsquare=Corr[0,1];
    pseudorsquared=prsquared(lognull,logverot);
    llrf=llr(lognull,logverot);
    llrf_pvalue=llr_pvalue(X,llrf);
    if (p>1): 
        Bconst=np.zeros((p-1, 1));
        Omega=np.identity(p-1);
        Bconst, Dconst, Hconst = calculo_B(X[:,0:(p-1)], Y, Omega, 1);
        vconst = np.asmatrix(np.dot(X[:,0:(p-1)],Bconst));
        errorconst=calculo_error(Y, vconst, n, p);
        logmreduc=loglike(errorconst, n, p);
        tcorte1, testm1, pvaluem1 = test_additional_predictors2(logverot, logmreduc, n, p, alpha, error, delta_grid, ipower);
        Btp=np.zeros((p, 1));
        Btp2=np.zeros((p-1, 1));
        Xesp=np.zeros((n, (p-1)));
        Xesp=np.asmatrix(Xesp);
        j=0;
        for i in range(0,p):
            if i==(p_special_test-1):
                i=i;
            else:
                aux=np.asmatrix(X[:,i]);
                if aux.shape[0]==1:
                    Xesp[:,j]=aux.transpose();
                else:
                    Xesp[:,j]=aux;
                j=j+1;
        Btp2=np.zeros((p-1, 1));
        Omega=np.identity(p-1);
        Btp2, Dreduct, Hreduct = calculo_B(Xesp, Y, Omega, 1);
        vreduct = np.asmatrix(np.dot(Xesp,Btp2));
        errorvreduct=calculo_error(Y, vreduct, n, p);
        logmreducp=loglike(errorvreduct, n, p);
        j=0;
        for i in range(0,p):
            if i==(p_special_test-1):
                Btp[i]=0;
            else:
                Btp[i]=Btp2[j];
                j=j+1;
        tcortespecial, testspecial, pvalpspecial = test_additional_predictors2(logverot, logmreducp, n, p, alpha, error, delta_grid, ipower);
    else:
        logmreduc=0;
        logmreducp=0;        
        tcorte1=0;
        testm1=0;
        pvaluem1=0;
        tcortespecial=0;
        testspecial=0;
        pvalpspecial=0;        
    
    return Bt, vt, error_predicciont, Rsquare, lognull, logmreducp, logverot, scoret, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_final, Tfinal, P_value,  tcorte1, testm1, pvaluem1, tcortespecial, testspecial, pvalpspecial;

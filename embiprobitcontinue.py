''' embiprobit(Y1, Y2, X, maxiter=100, error=0.01, perturbation=0.001, alpha=0.05, abseps=1e-6, p_special_test=1):
Parameters Y1 an Bivariate Vector with 0 or 1 with the same raws with X
Y2 an Bivariate Vector with 0 or 1 with the same raws with X
X is the explicative variables, the constant is colocated in the Execution Problems.
maxiter is the maximal iteration number for default is 100.
Error is the error of tolerance to stop the look for the loglikehood in the EM Algorithm. For default is  0.01.
perturbation is a variable to perturb the Matrix X^tX if you like invert its. The default value is 0.001.
alpha is the value to look for the cut point of each test, for default is 0.05.
abseps is the valur of tolerance to Normal Multivariate Distribution, for default es 1e-6.
p_special_test is the variable p of the matrix X that we like to get the multivariate p-value, for default is 1.

Results:

iteracion is the number of iteration that Em Algorithm use to look for the optimal.
rho is the correlation of Variable Y1 with variable Y2.
Sigma is the correlation Matrix, is 2x2 dimentions.
Btt1 is the ponderators of X to Y1.
Btt2 is the ponderators of X to Y2.
vt1 is the predicción to Y1, if vt1[i]>0 then Y1[i] estimate is 1. 
vt2 is the predicción to Y2, if vt2[i]>0 then Y2[i] estimate is 1. 
error_predicciont1 the cuadratic difference of error of Y respect the Y estimate, 
error_predicciont2 the cuadratic difference of error of Y respect the Y estimate,
Rsquare1 the correlation of Y1 estimate with Y1.
Rsquare2 the correlation of Y2 estimate with Y2.
lognull is the loglikehood only the constant or last variable.
logverotp1 is the loglikehood without the variable p_special_test. 
logverot is the loglikehood using the complete X.
scoret1 is the variation of ponderators to Y1, this function appear in (Green, 2008).
scoret2 is the variation of ponderators to Y2, this function appear in (Green, 2008).
scorerho is the variation of rho, this function appear in (Green, 2008).
pseudorsquared is the value calculed as proportion of loglikehood (Green, 2008).  
llrf is the test of -2 diference of likelihood complete model less model with constant only
llrf_pvalue is the p-value of llrf
marginalt is the marginal value that appear in (Green, 2008).
Desv_b1 is the standard desviation of each ponderator to Y1.
Desv_b2 is the standard desviation of each ponderator to Y2.
Desv_rho is the standard desviation of rho.
Tfinal1 is the value ponderator/standard desviation for each ponderator.
Tfinal2 is the value ponderator/standard desviation for each ponderator.
P_value1 is the p-value of each test using the t-student distribution as linear standard model para Tfinal1,  
P_value2 is the p-value of each test using the t-student distribution as linear standard model para Tfinal2,  
tcortemb is the cut point of multilinneal test to the model without constant,
testmb is the test value of multilinneal test to the model without constant,
pvaluemb is the p-value of testmb,
tcortespecial is the cut point of multilinneal test to the model without p_special_test variable.
testspecial is the test value of multilinneal test to the model without p_special_test variable.
pvalpspecial is the p-value of testspecial value.
logverotppy1 is the loglikehood of partial test with ponderator of p_special_test with value 0 to Y1 and not 0 to Y2.
tcortepspecialy1 cut point of partial test with ponderator of p_special_test with value 0 to Y1 and not 0 to Y2.
testpspecialy1 test value for partial test with ponderator of p_special_test with value 0 to Y1 and not 0 to Y2.
pvaluepspecialy1 p-value of testpspecialy1 of partial test with ponderator of p_special_test with value 0 to Y1 and not 0 to Y2.
logverotppy2 is the loglikehood of partial test with ponderator of p_special_test with value 0 to Y2 and not 0 to Y1.
tcortepspecialy2 cut point of partial test with ponderator of p_special_test with value 0 to Y2 and not 0 to Y1.
testpspecialy2 test value for partial test with ponderator of p_special_test with value 0 to Y1 and not 0 to Y2.
pvaluepspecialy2 p-value of partial test with ponderator of p_special_test with value 0 to Y1 and not 0 to Y2.

Greene, W. (1998). Econometric Analysis, Third edition. Prentice-Hall,
New Jersey.

'''


import numpy as np;
import numpy.linalg as la;
import scipy.stats;
from scipy import stats, special, optimize;
from scipy.stats import norm, chisqprob, mvn, distributions;
import scipy.linalg as scl;
#import sympy as sy;
import math;
#import statsmodels;


FLOAT_EPS = np.finfo(float).eps;

def calculo_B0(X, Z, p, perturbation):
    B0=np.zeros((p,1));
    B0=np.asmatrix(np.dot((scl.pinv((perturbation*np.identity(p))+np.dot(np.transpose(X),X))),np.dot(np.transpose(X),Z)));
    #B0=np.asmatrix(np.dot((scl.pinv(np.dot(np.transpose(X),X))),np.dot(np.transpose(X),Z)));
    return np.asmatrix(B0);

def calculo_B(X, v, Omega, p):
    H=np.dot(Omega,np.dot(np.transpose(X),v));
    H1=scl.pinv(0*np.identity(p)+np.dot(Omega,np.dot(np.dot(np.transpose(X),X),Omega)));
    D1=np.transpose(np.diagonal(H1));
    B=np.dot(Omega,np.dot(H1,H));
    return np.asmatrix(B), np.asmatrix(D1);

def calculo_rho(qq1, qq2, XBt1, XBt2, n, p):
     XX=np.zeros((2,n));
     XB1=qq1-XBt1;
     XB2=qq2-XBt2;
     XX=np.asmatrix(np.concatenate((XB1,XB2),1).transpose());
     Sigma = np.asmatrix(np.corrcoef(XX));
     rho = Sigma[0,1];
     return rho, np.asmatrix(Sigma);

def normpdf(x, mu, Sigma, p):
    if x.shape[0]==1:
        x=np.transpose(x);
    if mu.shape[0]==1:
        mu=np.transpose(mu);

    dev=np.zeros((p,1));    
    part1 = np.exp(-0.5*p*np.log(2*np.pi));
    part2 = np.power(np.linalg.det(Sigma),-0.5);
    for i in range(0,p):
        dev[i]=x[i]-mu[i];

    part3 = np.exp(-0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(Sigma)),dev));
    return part1*part2*part3;

def allclose(x, y, rtol=1.e-5, atol=1.e-8):
    return all(less_equal(abs(x-y), atol + rtol * abs(y)))

def generar_q(YY1, YY2, n, p):
    q1=np.zeros((n, 1));
    q2=np.zeros((n, 1));
    q1=((2*YY1)-np.ones((n, 1)));
    q2=YY2;
    return q1, q2;

def calculo_margina1_v1(x, y, y2, qq1, qq2, ss1, rho, factor, maxpts1, abseps1):
    Sigma1=np.array([[1,rho],[rho,1]]);
    #test = np.inf;
    test=30;
    x0=-1*x;
    y0=-1*y;
    dify2=(y2+y0)/ss1;
    lowaux=np.array([x0,-1*test]);
    uppaux=np.array([test,dify2]);
    #phi2=mvstdnormcdf(lowaux,uppaux,Sigma1,maxpts1,abseps1,abseps1);
    mu=np.array([0,0]);
    phi2,inform = mvn.mvnun(lowaux,uppaux,mu,Sigma1);    
    if phi2<=abseps1:
       phi2=abseps1;
    v1=x+(1/phi2)*((stats.norm._pdf(x0)*(stats.norm._cdf((dify2+rho*x)*factor)))-rho*(stats.norm._pdf(dify2)*(stats.norm._sf((x0-rho*(dify2))*factor))));
    v2=y; 
    return v1, v2;
    
def calculo_margina1_v2(x, y, y2, qq1, qq2, ss1, rho, factor, maxpts1, abseps1):
    Sigma1=np.array([[1,rho],[rho,1]]);
    #test = np.inf;
    test = 30;
    x0=-1*x;
    y0=-1*y;
    dify2=(y2+y0)/ss1;
    lowaux=np.array([-1*test,-1*test]);
    uppaux=np.array([x0,dify2]);
    #phi2=mvstdnormcdf(lowaux,uppaux,Sigma1,maxpts1,abseps1,abseps1);
    mu=np.array([0,0]);
    phi2,inform = mvn.mvnun(lowaux,uppaux,mu,Sigma1);   
    if phi2<=abseps1:
       phi2=abseps1;
    v1=x-(1/phi2)*((stats.norm._pdf(x0)*(stats.norm._cdf((dify2+rho*x)*factor)))+rho*(stats.norm._pdf(dify2)*(stats.norm._cdf((x0-rho*(dify2))*factor))));
    v2=y;
    return v1, v2;
    
c_m_v1 = np.vectorize(calculo_margina1_v1); 
c_m_v2 = np.vectorize(calculo_margina1_v2); 

def calculo_v(Y1, Y2, XBt1, XBt2, n, rho, Sigma, q1, q2, maxpts1,abseps1):
    v1=np.zeros((n, 1));
    v2=np.zeros((n, 1));
    E1=np.zeros((n, 1));
    E2=np.zeros((n, 1));
    vv1=np.zeros((n, 1));
    VV1_1=np.zeros((n, 1));
    VV2_1=np.zeros((n, 1));
    VV1_2=np.zeros((n, 1));
    VV2_2=np.zeros((n, 1));
    E1[(Y1==1)]=1;
    E2[(Y1==0)]=1;
    vv1=(Y2-XBt2);
    sigma1=vv1.std();
    factor=1/(np.sqrt(1-rho**2));
    Sigma1=np.array(Sigma);
    VV1_1,VV2_1 = c_m_v1(XBt1, XBt2, Y2, q1, q2, sigma1*np.ones((n, 1)), rho*np.ones((n, 1)),factor*np.ones((n, 1)), maxpts1*np.ones((n, 1)), abseps1*np.ones((n, 1)));
    VV1_2,VV2_2 = c_m_v2(XBt1, XBt2, Y2, q1, q2, sigma1*np.ones((n, 1)), rho*np.ones((n, 1)),factor*np.ones((n, 1)), maxpts1*np.ones((n, 1)), abseps1*np.ones((n, 1)));
    v1=(np.multiply(E1,VV1_1)+np.multiply(E2,VV1_2));
    v2=(np.multiply(E1,VV2_1)+np.multiply(E2,VV2_2));    
    return np.asmatrix(v1), np.asmatrix(v2);

def convertir_v(v, n):
    Y=np.zeros((n, 1));
    for i in range(0,n):
        if v[i]>=0:
           Y[i]=1;
        else:
           Y[i]=0; 
    return Y;

def diagonal_beta(p,beta):
    I=np.identity(p);
    for i in range(0,p):
        I[i,i]=abs(beta[i]);
    return I;

def loglike(Y, X, beta, n, p):
    XB = np.dot(X,beta);
    q=np.zeros((n, 1));
    for i in range(0,n):
        q[i] = 2*Y[i] - 1;
    suma=0;
    for i in range(0,n):
        suma=suma+np.log(stats.norm._cdf(q[i]*XB[i]));

    return suma;

def funcion_margina1_loglike2(y1, y2, x1, x2, qq1, qq2, rho, maxpts1, abseps1):
    rhoq=qq1*(qq2*rho);
    Sigma1=np.array([[1,rhoq],[rhoq,1]]);
    Sigma1=np.asmatrix(Sigma1);
    compo=np.zeros((2, 1));
    #test = -1*np.inf;
    #test = -30;
    #low=np.array([test,test]);
    #uppaux=np.array([qq1*x,y]);
    #phi2=mvstdnormcdf(low,uppaux,Sigma1,maxpts1,abseps1,abseps1);
    #mu=np.array([0,0]);
    #phi2,inform = mvn.mvnun(low,uppaux,mu,Sigma1); 
    compo[0]=qq1-x1;
    compo[1]=y2-x2;
    compo=np.asmatrix(compo);
    H=np.dot(np.transpose(compo),np.dot(scl.pinv(Sigma1),compo));
    #phi2=-0.5*np.log(np.sqrt(1-rho**2))-0.5*H;
    phi2=-0.5*np.log(2*np.pi)-0.5*np.log(np.sqrt(1-rho**2))-0.5*H;
    #phi2=-0.5*H;
    #phi2=np.log(np.sqrt(1-rho**2));
    #if phi2<=abseps1:
    #   phi2=abseps1;
    return phi2;

f_m_loglike2 = np.vectorize(funcion_margina1_loglike2); 

def loglike2(Y1, Y2, Xb1, Xb2, n, p, Sigma, rho, q1, q2, maxpts1, abseps1):
    llogvector=f_m_loglike2(Y1, Y2, Xb1, Xb2, q1, q2, rho*np.ones((n, 1)), maxpts1*np.ones((n, 1)), abseps1*np.ones((n, 1)));
    return np.sum(llogvector);

def funcion_margina1_generar_g(x, y, qq1, qq2, rho, factor):
    rhoi=qq1*rho;
    gg1 = stats.norm._pdf(qq1*x)*stats.norm._cdf((y-rhoi*x)*factor);
    gg2 = stats.norm._pdf(y)*stats.norm._cdf((qq1*x-rho*y)*factor);
    return gg1, gg2;

f_m_generar_g = np.vectorize(funcion_margina1_generar_g); 

def generar_g(Xb1, Xb2, n, p, Sigma, rho, q1, q2):
    g1=q1;
    g2=q2;
    factor=1/(np.sqrt(1-rho**2));
    g1, g2 = f_m_generar_g(Xb1, Xb2, q1, q2, rho*np.ones((n, 1)), factor*np.ones((n, 1)));
    return g1, g2;

def marginal_score2(p, x, x1, x2, rho1, qqq1, qqq2, ggg1, ggg2, maxpts1, abseps1):
    summa_rho=np.zeros((1, 1));
    summa1=np.zeros((1, p));
    summa2=np.zeros((1, p));
    #test = -1*np.inf;
    test = -30;
    low=np.array([test,test]);
    upp=np.zeros((2,1));
    upp=np.array([qqq1*x1,x2]);
    Sigma1=np.array([[1,rho1],[rho1,1]]);
    #phi2=mvstdnormcdf(low,upp,Sigma1,maxpts1,abseps1,abseps1);
    mu=np.array([0,0]);
    phi2,inform = mvn.mvnun(low,upp,mu,Sigma1); 
    if phi2<=abseps1:
       phi2=abseps1;       
    summa1=(((qqq1*ggg1)/phi2)*(x));
    summa2=(((ggg2)/phi2)*(x));
    summa_rho=((qqq1*(normpdf(upp, mu, Sigma1, 2)))/phi2);
    return summa1, summa2, summa_rho;

f_marginal_score2 = np.vectorize(marginal_score2); 

def score2(X, Xb1, Xb2, n, p, rho, qq1, qq2, gg1, gg2, maxpts1, abseps1):
    sumaa1=np.zeros((n, p));
    sumaa2=np.zeros((n, p));
    sumaa_rho=np.zeros((n, 1));
    sumaa1, sumaa2, sumaa_rho=f_marginal_score2(p*np.ones((n, 1)), X, Xb1, Xb2, rho*np.ones((n, 1)), qq1, qq2, gg1, gg2, maxpts1*np.ones((n, 1)), abseps1*np.ones((n, 1)));
    suma1=(np.sum(sumaa1,0));
    suma2=(np.sum(sumaa2,0));
    suma_rho=(np.sum(sumaa_rho[:,0],0)); 
    return np.transpose(suma1), np.transpose(suma2), suma_rho;

def funcion_margina1_1(x, y, rho, factor):
    return stats.norm._pdf(x)*stats.norm._cdf((y-rho*x)*factor);

f = np.vectorize(funcion_margina1_1);

def marginal(Xb1, Xb2, b1, b2, rho, n, p):
        factor=1/(np.sqrt(1-rho**2));
        XB1=f(Xb1, Xb2, rho*np.ones((n, 1)), factor*np.ones((n, 1)));
        XB2=f(Xb2, Xb1, rho*np.ones((n, 1)), factor*np.ones((n, 1)));
        L1 = np.dot(XB1,np.transpose(b1));
        L2 = np.dot(XB2,np.transpose(b2));
        L=L1+L2;
        suma1=np.mean(L,0);
        suma1=np.transpose(suma1);
        return suma1;

def tstudent(Z, n, p):
    return 2*(1.0 - scipy.stats.t(n-p-1).cdf(Z));  


def prsquared(lognull,logverot):
    return 1 - (logverot/lognull);


def llr(lognull,logverot):
    return -2*(lognull - logverot);

  
def llr_pvalue(X,llrf):
    df_model = float(la.matrix_rank(X) - 1);
    return stats.chisqprob(llrf, df_model);

def calculo_error(Y, Yt, n, p):
    YM=Y-Yt;
    suma=np.dot(YM.transpose(),YM);
    return suma;             

def desviaciones2(X, XB1, XB2, n, p, rho, q1, q2, g1, g2, maxpts1, abseps1):
    suma1=np.zeros((p, p));
    suma2=np.zeros((p, p));
    suma3=np.zeros((p, p));
    suma4=np.zeros((p, 1));
    suma5=np.zeros((p, 1));
    suma6=0;
    completa=np.zeros((2*p+1,2*p+1));
    std_errorc=np.zeros((1,2*p+1));
    desviaciones=np.zeros((2*p+1,1));
    desviaciones1=np.zeros((p,1));
    desviaciones2=np.zeros((p,1));
    desviaciones_rho=0;
    upp=np.zeros((1,2));
    mu=np.zeros((2,1));
    #test = np.inf;
    test = 30;
    low=np.array([-1*test,-1*test]);
    factor=1/(np.sqrt(1-rho**2));
    Sigma1=np.ones((2,2));
    for i in range(0,n):
        rhoi=(q1[i,0]*rho);
        MX=np.asmatrix(np.dot(np.transpose(X[i,None]),X[i,None]));
        q1XB1=(q1[i,0]*XB1[i,0]);
        q2XB2=XB2[i,0];
        upp=np.array([q1XB1,q2XB2]);
        Sigma1=np.array([[1,rhoi],[rhoi,1]]);
        mu=np.array([0,0]);
        phi2,inform = mvn.mvnun(low,upp,mu,Sigma1); 
        #phi2=mvstdnormcdf(low,upp,Sigma1,maxpts1,abseps1,abseps1);
        if phi2<=abseps1:
           phi2=abseps1;
        mphi2=normpdf(upp, mu, Sigma1, 2);
        pedazo1=((q1XB1*g1[i])/phi2)-((rhoi*mphi2)/phi2)-((g1[i]**2)/(phi2**2));
        pedazo2=((q2XB2*g2[i])/phi2)-((rhoi*mphi2)/phi2)-((g2[i]**2)/(phi2**2));
        pedazo3=(mphi2/phi2)-((g1[i]*g2[i])/(phi2**2));
        pedazo4=((mphi2/phi2)*(rhoi*(factor*(factor*(q2XB2-rhoi*q1XB1)))-q1XB1-(g1[i]/phi2)));
        pedazo5=q1[i,0]*((mphi2/phi2)*(rhoi*(factor*(factor*(q1XB1-rhoi*q2XB2)))-q2XB2-(g2[i]/phi2)));
        pedazo6=(mphi2/phi2)*(((factor**2)*rhoi)*(1-(factor**2)*(((XB1[i,0]**2)+(XB2[i,0]**2))-2*rhoi*(q1XB1*q2XB2)))+((factor*(q1XB1*q2XB2))-(mphi2/phi2)));
        suma1 = suma1 + (pedazo1[0,0]*MX);
        suma2 = suma2 + (pedazo2[0,0]*MX);
        suma3 = suma3 + (pedazo3[0,0]*((q1[i,0])*MX));
        suma4 = suma4 + (pedazo4[0,0]*(np.transpose(X[i,None])));
        suma5 = suma5 + (pedazo5[0,0]*(np.transpose(X[i,None])));
        suma6 = suma6 + pedazo6[0,0];

    completa[0:p,0:p]=suma1[0:p,0:p];
    completa[(p):(2*p),(p):(2*p)]=suma2[0:p,0:p];
    completa[0:p,(p):(2*p)]=suma3[0:p,0:p];
    completa[p:2*p,0:p]=suma3[0:p,0:p].transpose();
    completa[2*p,2*p]=suma6;
    for i in range(0,p):
        completa[i,2*p]=suma4[i,0];
        completa[p+i,2*p]=suma5[i,0];
        completa[2*p,i]=suma4[i,0];
        completa[2*p,p+i]=suma5[i,0]; 
    
    completai=scl.pinv(completa);
    suma1=scl.pinv(suma1);
    suma2=scl.pinv(suma2);
    std_errorc=np.sqrt(np.diag(np.abs(completai)));
    desviaciones=std_errorc[:,None];
    desviaciones1=desviaciones[0:p];
    desviaciones2=desviaciones[p:2*p];
    desviaciones3=desviaciones[2*p];
    return desviaciones1, desviaciones2, desviaciones3;    

def test_additional_predictors(loghipotesis0, loghipotesis1, m1, n1, p1, alpha):
    det1=1;
    det2=1;
    r=p1-1;
    q=p1-2;
    estadistico=-2*(loghipotesis0 - loghipotesis1);
    gl=m1*(r-q);
    tcorte=stats.chi2.ppf(1-alpha, gl, loc=0, scale=1);
    #pvalue2=stats.chi2.sf((n1-0.5*r-2)*estadistico, gl, loc=0, scale=1);
    pvalue2=stats.chi2.sf(estadistico, gl, loc=0, scale=1);
    return tcorte, estadistico, pvalue2;


def embiprobitcontinuelittle(Ym1, Ym2, Xm, qm1, qm2, maxiter1, error1, perturbation1, maxpts1, abseps1):
    n1=Xm.shape[0];
    p1=Xm.shape[1];
    Btm1=np.zeros((p1, 1));
    Btm2=np.zeros((p1, 1));
    Btm1=calculo_B0(Xm, qm1, p1, perturbation1);
    Btm2=calculo_B0(Xm, qm2, p1, perturbation1);
    Omegat1=np.identity(p1);
    XBtm1 = np.asmatrix(np.dot(Xm,Btm1));
    XBtm2 = np.asmatrix(np.dot(Xm,Btm2));
    rhotm, Sigmatm = calculo_rho(qm1, qm2, XBtm1, XBtm2, n1, p1);
    vtm1, vtm2=calculo_v(Ym1, Ym2, XBtm1, XBtm2, n1, rhotm, Sigmatm, qm1, qm2, maxpts1, abseps1);
    logverot1=loglike2(Ym1, Ym2, XBtm1, XBtm2, n1, p1, Sigmatm, rhotm, qm1, qm2, maxpts1, abseps1);
    Bttm1, Dttm1=calculo_B(Xm, vtm1, Omegat1, p1);
    Bttm2, Dttm2=calculo_B(Xm, vtm2, Omegat1, p1);
    XBttm1 = np.asmatrix(np.dot(Xm,Bttm1));
    XBttm2 = np.asmatrix(np.dot(Xm,Bttm2));
    rhottm, Sigmattm = calculo_rho(qm1, qm2, XBttm1, XBttm2, n1, p1);
    vttm1, vttm2=calculo_v(Ym1, Ym2, XBttm1, XBttm2, n1, rhottm, Sigmattm, qm1, qm2, maxpts1, abseps1);
    logverott1=loglike2(Ym1, Ym2, XBttm1, XBttm2, n1, p1, Sigmattm, rhottm, qm1, qm2, maxpts1, abseps1);
    iteracion1=0;
    while (np.linalg.norm(logverott1-logverot1)/np.linalg.norm(logverott1)) > error1 and (iteracion1 <= (maxiter1-1)):
         logverot1=logverott1;
         Btm1=Bttm1;
         Dtm1=Dttm1;
         Btm2=Bttm2;
         Dtm2=Dttm2;
         XBtm1 = np.asmatrix(np.dot(Xm,Btm1));
         XBtm2 = np.asmatrix(np.dot(Xm,Btm2));
         rhotm=rhottm;
         Sigmatm=Sigmattm;
         Omegat1=np.identity(p1);
         vtm1, vtm2=calculo_v(Ym1, Ym2, XBtm1, XBtm2, n1, rhotm, Sigmatm, qm1, qm2, maxpts1, abseps1);
         Bttm1, Dttm1=calculo_B(Xm, vtm1, Omegat1, p1);
         Bttm2, Dttm2=calculo_B(Xm, vtm2, Omegat1, p1);
         XBttm1 = np.asmatrix(np.dot(Xm,Bttm1));
         XBttm2 = np.asmatrix(np.dot(Xm,Bttm2));
         rhottm, Sigmattm = calculo_rho(qm1, qm2, XBttm1, XBttm2, n1, p1);
         vttm1, vttm2=calculo_v(Ym1, Ym2, XBttm1, XBttm2, n1, rhottm, Sigmattm, qm1, qm2, maxpts1, abseps1);
         logverott1=loglike2(Ym1, Ym2, XBttm1, XBttm2, n1, p1, Sigmattm, rhottm, qm1, qm2, maxpts1, abseps1);
         iteracion1=iteracion1+1;
 

    rhom=rhottm;
    Sigmam=Sigmattm;
    return rhom, Sigmam, iteracion1, Bttm1, Bttm2, vttm1, vttm2, XBttm1, XBttm2, logverott1;

def regularizar(BB1,p2,p_special_test2):
    BB2=np.zeros((p2, 1));
    j=0;    
    for i in range(0,p2):
        if i==(p_special_test2-1):
            BB2[i,0]=0;
        else:
            BB2[i,0]=BB1[j,0];
            j=j+1;

    return BB2;

def embiprobitlittley10(Ym1, Ym2, Xm, qm1, qm2, maxiter1, error1, perturbation1, maxpts1, abseps1, p_special_test):
    n1=Xm.shape[0];
    p1=Xm.shape[1];
    Btm1=np.zeros((p1, 1));
    Btm2=np.zeros((p1, 1));
    Btmm2=np.zeros((p1-1, 1));
   
    Xmesp=np.zeros((n1, (p1-1)));
    Xmesp=np.matrix(Xmesp);
    Xm=np.matrix(Xm);
    j=0;
    for i in range(0,p1):
        if i==(p_special_test-1):
            i=i;
        else:
           Xmesp[:,j]=(np.matrix(Xm[:,i]));
           j=j+1;

    Btmm1=calculo_B0(Xmesp, qm1, p1-1, perturbation1);
    Btm2=calculo_B0(Xm, qm2, p1, perturbation1);
    #if Btm1.shape[0]==1:
    #    Btm1=np.transpose(Btm1);
    #if Btm2.shape[0]==1:
    #    Btm2=np.transpose(Btm2);
    Btm1=regularizar(Btmm1,p1,p_special_test);
    XBtm1 = np.asmatrix(np.dot(Xm,Btm1));
    XBtm2 = np.asmatrix(np.dot(Xm,Btm2));
    Omegat1=np.identity(p1);
    Omegat2=np.identity(p1-1);
    rhotm, Sigmatm = calculo_rho(qm1, qm2, XBtm1, XBtm2, n1, p1);
    vtm1, vtm2=calculo_v(Ym1, Ym2, XBtm1, XBtm2, n1, rhotm, Sigmatm, qm1, qm2, maxpts1, abseps1);
    logverot1=loglike2(Ym1, Ym2, XBtm1, XBtm2, n1, p1, Sigmatm, rhotm, qm1, qm2, maxpts1, abseps1);
    Bttmm1, Dttmm1=calculo_B(Xmesp, vtm1, Omegat2, p1-1);
    Bttm2, Dttm2=calculo_B(Xm, vtm2, Omegat1, p1);
    Dttm1=Omegat1;
    Bttm1=regularizar(Bttmm1,p1,p_special_test);
    XBttm1 = np.asmatrix(np.dot(Xm,Bttm1));
    XBttm2 = np.asmatrix(np.dot(Xm,Bttm2));    
    rhottm, Sigmattm = calculo_rho(qm1, qm2, XBttm1, XBttm2, n1, p1);
    vttm1, vttm2=calculo_v(Ym1, Ym2, XBttm1, XBttm2, n1, rhottm, Sigmattm, qm1, qm2, maxpts1, abseps1);
    logverott1=loglike2(Ym1, Ym2, XBttm1, XBttm2, n1, p1, Sigmattm, rhottm, qm1, qm2, maxpts1, abseps1);
    iteracion1=0;
    while (np.linalg.norm(logverott1-logverot1)/np.linalg.norm(logverott1)) > error1 and (iteracion1 <= (maxiter1-1)):
         logverot1=logverott1;
         Btm1=Bttm1;
         Dtm1=Dttm1;
         Btm2=Bttm2;
         Dtm2=Dttm2;
         rhotm=rhottm;
         Sigmatm=Sigmattm;
         XBtm1 = np.asmatrix(np.dot(Xm,Btm1));
         XBtm2 = np.asmatrix(np.dot(Xm,Btm2));         
         vtm1, vtm2=calculo_v(Ym1, Ym2, XBtm1, XBtm2, n1, rhotm, Sigmatm, qm1, qm2, maxpts1, abseps1);
         Bttmm1, Dttmm1=calculo_B(Xmesp, vtm1, Omegat2, p1-1);
         Bttm2, Dttm2=calculo_B(Xm, vtm2, Omegat1, p1);
         Dttmm1=Omegat1;
         Bttm1=regularizar(Bttmm1,p1,p_special_test);
         XBttm1 = np.asmatrix(np.dot(Xm,Bttm1));
         XBttm2 = np.asmatrix(np.dot(Xm,Bttm2));             
         rhottm, Sigmattm = calculo_rho(qm1, qm2, XBttm1, XBttm2, n1, p1);
         vttm1, vttm2=calculo_v(Ym1, Ym2, XBttm1, XBttm2, n1, rhottm, Sigmattm, qm1, qm2, maxpts1, abseps1);
         logverott1=loglike2(Ym1, Ym2, XBttm1, XBttm2, n1, p1, Sigmattm, rhottm, qm1, qm2, maxpts1, abseps1);
         iteracion1=iteracion1+1;


    rhom=rhottm;
    Sigmam=Sigmattm;
    return rhom, Sigmam, iteracion1, Bttm1, Bttm2, logverott1;


def embiprobitlittley20(Ym1, Ym2, Xm, qm1, qm2, maxiter1, error1, perturbation1, maxpts1, abseps1, p_special_test):
    n1=Xm.shape[0];
    p1=Xm.shape[1];
    Btm1=np.zeros((p1, 1));
    Btm2=np.zeros((p1, 1));
    Btmm2=np.zeros((p1-1, 1));
    
    Xmesp=np.zeros((n1, (p1-1)));
    Xmesp=np.matrix(Xmesp);
    Xm=np.matrix(Xm);
    j=0;
    for i in range(0,p1):
        if i==(p_special_test-1):
            i=i;
        else:
           Xmesp[:,j]=(np.matrix(Xm[:,i]));
           j=j+1;

    Btm1=calculo_B0(Xm, qm1, p1, perturbation1);
    Btmm2=calculo_B0(Xmesp, qm2, p1-1, perturbation1);
    Btm2=regularizar(Btmm2,p1,p_special_test);
    XBtm1 = np.asmatrix(np.dot(Xm,Btm1));
    XBtm2 = np.asmatrix(np.dot(Xm,Btm2));
    Omegat1=np.identity(p1);
    Omegat2=np.identity(p1-1);
    rhotm, Sigmatm = calculo_rho(qm1, qm2, XBtm1, XBtm2, n1, p1);
    vtm1, vtm2=calculo_v(Ym1, Ym2, XBtm1, XBtm2, n1, rhotm, Sigmatm, qm1, qm2, maxpts1, abseps1);
    logverot1=loglike2(Ym1, Ym2, XBtm1, XBtm2, n1, p1, Sigmatm, rhotm, qm1, qm2, maxpts1, abseps1);
    Bttm1, Dttm1=calculo_B(Xm, vtm1, Omegat1, p1);
    Bttmm2, Dttmm2=calculo_B(Xmesp, vtm2, Omegat2, p1-1);
    Dttm2=Omegat1;
    Bttm2=regularizar(Bttmm2,p1,p_special_test);
    XBttm1 = np.asmatrix(np.dot(Xm,Bttm1));
    XBttm2 = np.asmatrix(np.dot(Xm,Bttm2));   
    rhottm, Sigmattm = calculo_rho(qm1, qm2, XBttm1, XBttm2, n1, p1);
    vttm1, vttm2=calculo_v(Ym1, Ym2, XBttm1, XBttm2, n1, rhottm, Sigmattm, qm1, qm2, maxpts1, abseps1);
    logverott1=loglike2(Ym1, Ym2, XBttm1, XBttm2, n1, p1, Sigmattm, rhottm, qm1, qm2, maxpts1, abseps1);
    iteracion1=0;
    while (np.linalg.norm(logverott1-logverot1)/np.linalg.norm(logverott1)) > error1 and (iteracion1 <= (maxiter1-1)):
         logverot1=logverott1;
         Btm1=Bttm1;
         Dtm1=Dttm1;
         rhotm=rhottm;
         Sigmatm=Sigmattm;
         XBtm1 = np.asmatrix(np.dot(Xm,Btm1));
         XBtm2 = np.asmatrix(np.dot(Xm,Btm2));  
         vtm1, vtm2=calculo_v(Ym1, Ym2, XBtm1, XBtm2, n1, rhotm, Sigmatm, qm1, qm2, maxpts1, abseps1);
         Bttm1, Dttm1=calculo_B(Xm, vtm1, Omegat1, p1);
         Bttmm2, Dttmm2=calculo_B(Xmesp, vtm2, Omegat2, p1-1);
         Dttm2=Omegat1;
         Bttm2=regularizar(Bttmm2,p1,p_special_test);
         XBttm1 = np.asmatrix(np.dot(Xm,Bttm1));
         XBttm2 = np.asmatrix(np.dot(Xm,Bttm2));            
         rhottm, Sigmattm = calculo_rho(qm1, qm2, XBttm1, XBttm2, n1, p1);
         vttm1, vttm2=calculo_v(Ym1, Ym2, XBttm1, XBttm2, n1, rhottm, Sigmattm, qm1, qm2, maxpts1, abseps1);
         logverott1=loglike2(Ym1, Ym2, XBttm1, XBttm2, n1, p1, Sigmattm, rhottm, qm1, qm2, maxpts1, abseps1);
         iteracion1=iteracion1+1;
         #print(iteracion);


    rhom=rhottm;
    Sigmam=Sigmattm;
    return rhom, Sigmam, iteracion1, Bttm1, Bttm2, logverott1;

 
def embiprobitcontinue(Y1, Y2, X, maxiter=100, error=0.01, perturbation=0.001,  alpha=0.05, abseps=1e-6, p_special_test=1):
    n=X.shape[0];
    p=X.shape[1];
    maxpts=maxiter;
    q1, q2 = generar_q(Y1, Y2, n, p);
    X0=np.ones((n, 1));
    rho0, Sigma0, iteracion0, Bnull1, Bnull2,  vt01, vt02,  XBttm01, XBttm02, lognull= embiprobitcontinuelittle(Y1, Y2, X0, q1, q2, maxiter, error, perturbation, maxpts, abseps);
    rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, XBtt1, XBtt2, logverot=embiprobitcontinuelittle(Y1, Y2, X, q1, q2, maxiter, error, perturbation, maxpts, abseps);
    marginalt=marginal(XBtt1, XBtt2, Btt1, Btt2, rho, n, p);
    g1, g2 = generar_g(XBtt1, XBtt2, n, p, Sigma, rho, q1, q2);
    scoret1, scoret2, score_rho=score2(X, XBtt1, XBtt2, n, p, rho, q1, q2, g1, g2, maxpts, abseps);
    error_predicciont1=calculo_error(q1, vt1, n, p);
    error_predicciont2=calculo_error(q2, vt2, n, p);
    Desv_b1, Desv_b2, Desv_rho = desviaciones2(X, XBtt1, XBtt2, n, p, rho, q1, q2, g1, g2, maxpts, abseps);
    Tfinal1=np.zeros((p, 1));
    Tfinal2=np.zeros((p, 1));
    P_value1=np.zeros((p, 1));
    P_value2=np.zeros((p, 1));
          
    for i in range(0,p):
         if (abs(Desv_b1[i,0])<= error):
             Tfinal1[i]=(Btt1[i,0])*(1e100);
             P_value1[i] = 0;
         else:
             Tfinal1[i]=(Btt1[i,0])/(Desv_b1[i,0]);
             P_value1[i] = tstudent(np.abs(Tfinal1[i]), n, p);
             

         if (abs(Desv_b2[i,0])<= error):
             Tfinal2[i]=(Btt2[i,0])*(1e100);
             P_value2[i] = 0;
         else:
             Tfinal2[i]=(Btt2[i,0])/(Desv_b2[i,0]);
             P_value2[i] = tstudent(np.abs(Tfinal2[i]), n, p);

  

    XN1=np.asmatrix(np.concatenate((vt1,q1),1).transpose());
    XN2=np.asmatrix(np.concatenate((vt2,q2),1).transpose());
    Corr1=np.corrcoef(XN1);
    Corr2=np.corrcoef(XN2);
    Rsquare1=Corr1[0,1];
    Rsquare2=Corr2[0,1];
    pseudorsquared=prsquared(lognull,logverot);
    llrf=llr(lognull,logverot);
    llrf_pvalue=llr_pvalue(X,llrf);
    tcorte0, test0, pvalue0 =test_additional_predictors(lognull, logverot, 2, n, p, alpha);
    llrf_pvalue=pvalue0;
    if p>1:
        Bconst1=np.zeros((p, 1));
        Bconst2=np.zeros((p, 1));
        rhop, Sigmap, iteracionp, Bconst1, Bconst2, vt1p, vt2p, XBtt1p, XBtt2p, logverotp1= embiprobitcontinuelittle(Y1, Y2, X[:,0:p], q1, q2, maxiter, error, perturbation, maxpts, abseps);
        tcortemb, testmb, pvaluemb =test_additional_predictors(logverotp1, logverot, 2, n, p, alpha);
        Xesp=np.zeros((n, (p-1)));
        Xesp=np.matrix(Xesp);
        Xnew=X;
        Xnew=np.matrix(Xnew);
        j=0;
        for i in range(0,p):
            if i==(p_special_test-1):
               i=i;
            else:
               Xesp[:,j]=(np.matrix(Xnew[:,i]));
               j=j+1;
        rhopp, Sigmapp, iteracionpp, Bconstp1, Bconstp2, vt1pp, vt2pp, XBtt1pp, XBtt2pp, logverotpp1= embiprobitcontinuelittle(Y1, Y2, Xesp, q1, q2, maxiter, error, perturbation, maxpts, abseps);
        #print(logverot);
        #print(logverotpp1);     
        #print(-2*(logverotpp1 - logverot));
        tcortepspecial, testpspecial, pvaluepspecial =test_additional_predictors(logverotpp1, logverot, 2, n, p, alpha);
        #print(pvaluepspecial);        
        rhoppy1, Sigmappy1, iteracionppy1, Bconstp1y1, Bconstp2y1, logverotppy1 = embiprobitlittley10(Y1, Y2, X, q1, q2, maxiter, error, perturbation, maxpts, abseps, p_special_test);
        tcortepspecialy1, testpspecialy1, pvaluepspecialy1 =test_additional_predictors(logverotppy1, logverot, 1, n, p, alpha);
        rhoppy2, Sigmappy2, iteracionppy2, Bconstp1y2, Bconstp2y2, logverotppy2 = embiprobitlittley20(Y1, Y2, X, q1, q2, maxiter, error, perturbation, maxpts, abseps, p_special_test);
        tcortepspecialy2, testpspecialy2, pvaluepspecialy2 =test_additional_predictors(logverotppy2, logverot, 1, n, p, alpha);
    else:
        logverotpp1=0;  
        tcortemb=0;
        testmb=0;
        pvaluemb=0;
        tcortepspecial=0;
        testpspecial=0;
        pvaluepspecial=0;
        logverotppy1=0;
        tcortepspecialy1=0;
        testpspecialy1=0;
        pvaluepspecialy1=0;
        logverotppy2=0;
        tcortepspecialy2=0;
        testpspecialy2=0;
        pvaluepspecialy2=0;  

      
    return rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logverot, logverotpp1, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2;


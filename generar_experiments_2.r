library("mvtnorm")

n=100
x<-rnorm(n)
b01=0.5
b02=0.2

nsim=1
B1all=c(-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5)
B2all=c(-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5)
rhoall=c(-0.5,-0.3,0,0.3,0.5)


###########################################
#
# Primera opcion
#
###########################################

param=matrix(0,ncol=3,nrow=605)
i=0
for(b1 in B1all){
	for(b2 in B2all){
		for(rho in rhoall){	
			i=i+1
			param[i,]=c(b1,b2,rho)	
		}
	}
}

myind=(param[,1]==-2.5 | param[,1]==2.5 | param[,2]==2.5 | param[,2]==-2.5)
newparam=param[myind,]
l=1
for(j in 1:200){
#cat(j,"\n");
	b1=newparam[j,1]
	b2=newparam[j,2]
	rho=newparam[j,3]
	for(i in 1:nsim){
				e=rmvnorm(n,mean=c(0,0),sigma=matrix(c(1,rho,rho,1),ncol=2));
				z1=b01+b1*x+e[,1];
				z2=b02+b2*x+e[,2];
				y1s=1*(z1>0);
				y2s=1*(z2>0);
				tmp=cbind(y1s,y2s,x);
				nfile=paste("D:/bioinformatica/tablew",as.character(l),sep="");
				nfile=paste(nfile,".txt",sep="");
				write.table(tmp,file=nfile,quote=FALSE,row.names=FALSE,col.names=FALSE,sep=",");
				l=l+1;
}
}


###############################################
#
# Analisis resultados
#
###############################################

B1all=c(-2,-1.5,-1,-0.5,0,0.5,1,1.5,2)
B2all=c(-2,-1.5,-1,-0.5,0,0.5,1,1.5,2)
rhoall=c(-0.5,-0.3,0,0.3,0.5)
param=matrix(0,ncol=3,nrow=405)
i=0
for(b1 in B1all){
	for(b2 in B2all){
		for(rho in rhoall){	
			i=i+1
			param[i,]=c(b1,b2,rho)	
		}
	}
}

res0=read.table("/Users/susana/projects/probit/output/results_biprobit_experiments.txt")
res=read.table("/Users/susana/projects/probit/output/results_biprobit_experimentsV2.txt")
resall=rbind(res0,res)
colnames(resall)=c("b1","b2","rho","iterNum","b01","b1hat","b02","b2hat","rhohat","b01sd","b1hatsd","b02sd","b2hatsd","rhohatsd","b01statsd","b1hatstat","b02stat","b2hatstat","rhohatstat","b01pvalue","b1hatpvalue","b02pvalue","b2hatpvalue","rhohatpvalue","logLik1","b01mod0","b02mod0","rhohatmod0","b01sdmod0","b02sdmod0","rhohatsdmod0","b01statsdmod0","b02statmod0","rhohatstatmod0","b01pvaluemod0","b02pvaluemod0","rhohatpvaluemod0","logLik0")

summaryres=matrix(0,ncol=38,nrow=405)
for(j in 1:405){
	b1=param[j,1]
	b2=param[j,2]
	rho=param[j,3]
	indxtmp=(resall[,1]==b1 & resall[,2]==b2 & resall[,3]==rho)
	ntmp=sum(indxtmp)
	tmp=resall[indxtmp,]
	summaryres[j,]=apply(tmp,2,mean)
}

plot(summaryres[,c(6,8)],pch=20)



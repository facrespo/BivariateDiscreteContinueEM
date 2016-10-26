#This file make the analisys of the results the experiment of bi discrete
#and continue variables

library(stats);
setwd("E:/bioinformatica/");
datos <- read.csv("E:/bioinformatica/resultado_cont_dist_02.txt", header = TRUE, sep=",");
hist(datos$pvaluebivariateSNPH01);
ks.test(datos$pvaluebivariateSNPH01,"punif",0,1);
estadistico=-2*(datos$logwSNP  - datos$logmodelcomplet);
logico=estadistico<0;
pp=1-pchisq(estadistico, 2, ncp=0, lower.tail = T, log.p = F);
hist(pp);
proporcion1=sum(datos$pvaluebivariateSNPH01<=0.05)/1000;
proporcion2=sum(datos$pvaluebivariateSNPH01[logico==TRUE]==1);
proporcion3=(sum(datos$pvaluebivariateSNPH01<=0.05)/(1000-proporcion2));
hist(datos$pvaluebivariateSNPH01[datos$pvaluebivariateSNPH01<1]);
hist(datos$pvaluebivariateSNPH01[logico==FALSE]);
mean(datos$Correlation);
setwd("E:/bioinformatica/");
datos2 <- read.csv("E:/bioinformatica/resultado02bidis.txt", header = TRUE, sep=",");
hist(datos2$pvaluebivariateSNPH01);
proporcion1=sum(datos2$pvaluebivariateSNPH01<=0.05)/1000;
proporcion2=sum(datos2$pvaluebivariateSNPH01==1);
proporcion3=(sum(datos2$pvaluebivariateSNPH01<=0.05)/(1000-proporcion2));
mean(datos2$Correlation);
data_aqi<-read.csv("/Users/chenyuguan/Desktop/OneDrive/RPI/Courses/Data\ Analytics/china-aqi-test/CompletedDataset.csv",header = T)
str(data_aqi)

# is headline 1, ## is headline 2 and ### is comments

#preliminary data cleaning
data_aqi<-data_aqi[,-c(1,10)] #remove two variables to simplify work
library("ggplot2")
ggplot(data=data_aqi,aes(x=AQI))+geom_histogram(aes(y=..density..),color="black",fill="white",binwidth = 10)+geom_density(alpha=0.2,fill="#FF6666")
n<-which(data_aqi$AQI==max(data_aqi$AQI)) 
data_aqi<-data_aqi[-n,] ###remove the outlier
##principle components analysis
install.packages("pls")
library("pls") #install and activate required package for PCA
pcr.fit<-pcr(AQI~.,data=data_aqi,scale=T,validation="CV") #principle components regression
validationplot(pcr.fit,val.type = "MSEP") #cross validation to minimize error of prediction
dataforpcr<-data_aqi[,-1] #remove the first column "AQI", leave all influencing factors
matrixaqi<-as.matrix(dataforpcr) #convert dataframe to matrix
matrixaqi<-apply(matrixaqi,2,as.numeric) #convert all data to be numeric ones
pr<-prcomp(matrixaqi,scale. = T) #function for principle components analysis
biplot(pr,scale = 1) #output pca graph
pr.var<-pr$sdev^2 #output variation explained of principle components
pve<-pr.var/sum(pr.var) #output variation explained fraction of every component
plot(pve,xlab=" Principal Component",
     ylab=" Proportion of Variance Explained", ylim=c(0,1),type="b") #plot variation explained

#New dataset for modeling
newdata<-data_aqi[,-c(2,6,10)]
newdata[,1]<-as.numeric(newdata$AQI)
##group observations based on AQI value
newdata[,8]<-cut(newdata$AQI,br=c(1,50,100,150,300),labels=c('Good','Moderate','Slightly harmful','Harmful'))
colnames(newdata)[8]<-"AQIclass"
ggplot(data = newdata,aes(x=AQIclass))+geom_bar(color="steelblue",fill="white")
##visual approach for clustering
ggplot(newdata,aes(x=Longititute,y=Altitude,col=AQIclass))+geom_point()+scale_color_manual(values=c("#006600","#00e500","#F08080","#FF0000"))
ggplot(newdata,aes(x=Longititute,y=Temperature,col=AQIclass))+geom_point()+scale_color_manual(values=c("#006600","#00e500","#F08080","#FF0000"))
ggplot(newdata,aes(x=Altitude,y=Temperature,col=AQIclass))+geom_point()+scale_color_manual(values=c("#006600","#00e500","#F08080","#FF0000"))
ggplot(newdata,aes(x=GDP,y=GreenCoverageRate,col=AQIclass))+geom_point()+scale_color_manual(values=c("#006600","#00e500","#F08080","#FF0000"))
ggplot(newdata,aes(x=PopulationDensity,y=GDP,col=AQIclass))+geom_point()+scale_color_manual(values=c("#006600","#00e500","#F08080","#FF0000"))
ggplot(newdata,aes(x=PopulationDensity,y=GreenCoverageRate,col=AQIclass))+geom_point()+scale_color_manual(values=c("#006600","#00e500","#F08080","#FF0000"))
##hierarchical clustering based on geometrical factors
Geomedata<-newdata[,c(3,4,5,8)] ###create new dataframe for clustering
a<-as.matrix(Geomedata[,1:3])
b<-as.character(Geomedata[,4])
Geomedata<-list(a,b) ###convert dataframe to matrix for clustering
sd.data<-scale(a) ###scale all factors
data.dist<-dist(sd.data) ###calculate Euler's distance
plot(hclust(data.dist),labels=b,main = "complete linkage",xlab="",sub="",ylab="") 
###use "complete linkage" for hierarchical clustering
hc.out<-hclust(dist(sd.data),method="complete") ###output complete linkage clustering result
hc.cluster<-cutree(hc.out,3) ###cut clustering tree into 3 parts
table(hc.cluster,b) ###compare clustering result with actual result
plot(hclust(data.dist,method = "average"),labels=b,main = "average linkage",xlab="",sub="",ylab="") 
###use "single linkage" for hierarchical clustering
hc.out<-hclust(dist(sd.data),method="single") ###output single linkage clustering result
hc.cluster<-cutree(hc.out,3)
table(hc.cluster,b)
###use "average linkage" for hierarchical clustering
hc.out<-hclust(dist(sd.data),method="average") ###output average linkage clustering result
hc.cluster<-cutree(hc.out,3) ###cut clustering tree into 3 parts
table(hc.cluster,b) ###compare clustering result with actual result
abline(h=3.2,col="red") ###average linkage outputs best clustering, mark corresponding height
plot(hclust(data.dist,method = "single"),labels=b,main = "single linkage",xlab="",sub="",ylab="") 
###Finally, we choose average linkage for hierarchical clustering.
##Classification
N<-which(hc.cluster==2|hc.cluster==1) ###specify observations belonging to clustering 1&2
Classification<-newdata[-c(N,which(newdata$GDP>5000|newdata$PopulationDensity>6000)),-1]
###remove clustering 1&2 observation and some outliers for better training and prediction
install.packages("caret") 
library("caret") ###install and activate "caret" package for classification
partition_index<-createDataPartition(Classification$AQIclass,p=0.8,list = F)
###split dataset into train and test subset, test subset contains 20% of the whole data
test<-Classification[-partition_index,] ###name test subset
train<-Classification[partition_index,] ###name train subset
control<-trainControl(method = "cv",number = 10) ###set 10-fold cross validation
metric<-"Accuracy" ###use "Accuracy" to evaluate models
###we compare 5 different classification algorithms for better model
set.seed(10) ###set seeds to control cross-validation data consistence
###1st: linear discriminant analysis
fit.lda<-train(AQIclass~.,data=train,method="lda",metric=metric,trControl=control)
set.seed(10)
###2nd: recursive partitioning and regression trees
fit.rpart<-train(AQIclass~.,data=train,method="rpart",metric=metric,trControl=control)
set.seed(10)
###3rd: k-Nearest Neighbour Classification
fit.knn<-train(AQIclass~.,data=train,method="knn",metric=metric,trControl=control)
set.seed(10)
###4th: support vector machines
fit.svm<-train(AQIclass~.,data=train,method="svmRadial",metric=metric,trControl=control)
set.seed(10)
###5th: random forest
fit.rf<-train(AQIclass~.,data=train,method="rf",metric=metric,trControl=control)
results <- resamples(list(lda=fit.lda, rpart=fit.rpart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results) ###create list and compare models accuracy
dotplot(results) ###vividly show accuracies of different models 
###results show that randomForest are most accurate
print(fit.rf) ###see details of randomForest model and assigning mtry=2 gives best result
install.packages("randomForest")
library("randomForest")
Validation<-randomForest(AQIclass~.,data = train,ntree=500,mtry=2,importance=T)
###repeat randomForest model for validation
p<-predict(Validation,test,type="class") ###predict classification results using randomForest
table(p,test$AQIclass) ###compare model results and actual AQIclass








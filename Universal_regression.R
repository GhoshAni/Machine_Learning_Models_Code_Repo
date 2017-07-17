
###libraries
library(dummies)
library(vegan)
library(h2o)
library(randomForest)
library(DMwR)
library(car)
library(rpart)
library(e1071)
library(nnet)
library(FNN)
library(C50)
library(ada)
## get universal bank data set in R
rm(list = ls(all = TRUE))
getwd()

univ <- read.table("UnivBank.txt", header = TRUE, sep = ',')
class(univ)
names(univ)
## for regression Income as a target variable (y = income)
## for classification who is likely to offer a new personal loan ( y = personal loan)

## chanhe col names
library(data.table)

setnames(x = univ, old = c("ID" ,"Age","Experience", "Income" ,"ZIP.Code",          
                           "Family","CCAvg" ,"Education","Mortgage", "Personal.Loan",
  "Securities.Account" ,"CD.Account", "Online","CreditCard"), 
         new = c("ID" ,"age","experience", "income" ,"zip",          
                 "family","CCAvg" ,"education","mortgage", "loan","sec_acct" ,"cd_acct", "online","creditcard"))



str(univ)
class(univ)
data1 <- univ
class(data1)
### attributes name

attr = c("experience", "income","zip", "family" ,"CCAvg" ,"education","mortgage"  
           , "loan"  , "sec_acct",   "cd_acct"  ,  "online"  ,   "creditcard" ) 

names(data1)
cat_attr = c("family", "education","loan","sec_acct", "cd_acct", "online" ,"creditcard")


num_attr = setdiff(x = attr, y = cat_attr)

## conversion of data types

## categorical to factors
## numeric to numeric

names(data1)
## drop ID 
data1$ID = NULL
data1$age = NULL
str(data1)
cat_data <- data.frame(apply(data1[,cat_attr],2, function(x) as.factor(as.character(x))))
class(cat_data)
sum(is.na(cat_data))

num_data <- data.frame(sapply(data1[,num_attr], as.numeric))
sum(is.na(num_data))
class(num_data)
str(data1)
str(cat_data)
str(num_data)
dim(cat_data)
dim(num_data)

## dummify family and education
family=dummy(cat_data$family)
education=dummy(cat_data$education)
bankdata=cbind(bankdata,Family,edu)
cat_data$family=NULL
cat_data$education=NULL
##rm(data2)
##rm(data2)
data2 <- data.frame(cbind(cat_data, family,education, num_data ))
class(data2)
sum(is.na(data2))
str(data2)
### 3 data sets
cat_data --- #categorical data
num_data --- #numeric data
data2   ---#combined data

  

##Understand the spread of the data using the numerical attribute 
##and see how the target is varying using the categorical attributes.

##Identify the important patterns using visualizations

##library(mlbench)
names(data2)
head(data2, n =  20)
dim(data2)
sapply(data2, class)


summary(data2[, num_attr])

##mean_age = 45.34 sd = 11.463
#mean_exp = 20.1 sd = 11.467
#mean_ income = 73.77 sd = 46.033
#mean_CCavg 1.938  sd = 1.747
#mean_mort = 56.5  sd = 101.71

## standard deviation
sapply(data2[, num_attr], sd)

## correlation matrix
correlations = cor (data2 [, num_attr])

## age and exp has collinearity of .99
## age with ccavg and income and mortgage shows negative corr ?? 
## exp and income and ccavg and mortgage shows negative corr ??
##income with ccavg positive  and mortgage cor 

names(num_data)

### histogram for numeric variables
par(mfrow = c(1,3))
for (i in 1:3){
  hist(num_data[,i], main = names(num_data)[i]  )
}

## income shows a  right skewed distribution

## exp is mostly b/w 0 to 40
par(mfrow = c(1,3))
for (i in 1:3){
  hist(num_data[,i], main = names(data2)[i]  )
}
## loan is a right skewed distribution
## so both loan and income are right skwed distribution.
names(data2)
## family, exp, age, mortgage should affect income 
## there is an outlier in loan  and income both  beyond 200

head(num_data, n = 20)

## density plot
library(lattice)
par(mfrow = c(1,3))
for (i in 1:3){
  plot(density(num_data[,i]), main = names(num_data)[i]  )
}

## income is unimodal with highest concentration of income in 50 - 55 range.
## age is almost multimodal??
## exp seems to be bimodal??

## target variable income vs all numeric variables

## correlations
library(corrplot)

correlations <- cor(num_data)
##create  correlation plot
corrplot(correlations , method = "circle", number.cex = 7/ ncol(correlations))
names(num_data)

##ccAvg has a negative corrlelation with age, exp, zip, and positive correlation with 
##income and mortgage
## age and exp highly correlated so dropped age in previous code itself

## scatter plot for multivariate analysis.



attach(num_data)
plot(x = num_data$experience , y = income, xlab = "experience", ylab = "income" )
abline(lm( income ~ experience ), col = "red", lwd = 2, lty = 1)
lines(lowess(income, experience) , col = "blue", lwd = 2, lty =1)

attach(num_data)
plot(x = num_data$CCAvg , y = income, xlab = "ccAvg", ylab = "income" )
abline(lm( income ~ CCAvg ), col = "red", lwd = 2, lty = 1)
lines(lowess(income, CCAvg) , col = "blue", lwd = 2, lty =1)

attach(num_data)
plot(x = num_data$mortgage , y = income, xlab = "mortgage", ylab = "income" )
abline(lm( income ~ mortgage ), col = "red", lwd = 2, lty = 1)
lines(lowess(income, mortgage) , col = "blue", lwd = 2, lty =1)

## standardize

library(vegan)
num_stand <- decostand(num_data, "standardize")

str(num_data)
str(num_stand)

attach(num_stand)
plot(x = num_stand$mortgage , y = income, xlab = "mortgage", ylab = "income" )
abline(lm( income ~ mortgage ), col = "red", lwd = 2, lty = 1)
lines(lowess(income, mortgage) , col = "blue", lwd = 2, lty =1)

## after standardization income increases with mortgage possible outliers??

attach(num_stand)
plot(x = num_stand$CCAvg , y = income, xlab = "ccAvg", ylab = "income" )
abline(lm( income ~ CCAvg ), col = "red", lwd = 2, lty = 1)
lines(lowess(income, CCAvg) , col = "blue", lwd = 2, lty =1)

## most credit card avg spending happens when income is low.
## but with income increasing some ccAvg also increasing
## possible outliers??
## do I  need normalization??

attach(num_stand)
plot(x = num_stand$experience, y = income, xlab = "exp", ylab = "income")
abline(lm(income ~ experience), col = "red", lwd = 2)
lines(lowess(income, experience), col = "blue", lwd = 2)

###*************************************************************####
num_data
cat_data
data2
univ

### **********split datar*****************####

names(data2)
##back up
data3 <- data2
str(data2)
str(data3)
str(num_stand)
data2$experience = NULL
data2$income = NULL
data2$zip = NULL
data2$CCAvg = NULL
data2$mortgage = NULL
new_data <- data.frame(cbind(data2, num_stand))
## standardized final data set
str(new_data)
## split data
set.seed(1234)
train_index <- sample(x = nrow(new_data), size =  0.6 * nrow(new_data))
train_data <- new_data[train_index,]
test_data <- new_data[-train_index,]
str(train_data)
str(test_data)

dim(train_data)
str(train_data)
########*****************build auto encoder******************#########
library(h2o)

h2o.init(ip='localhost', port=54321, max_mem_size = '1g')

## take y variable out before running autoencoder
str(train_data)
str(test_data)
train_data1 <- train_data[, -c(14)]
test_data1 <-  test_data[, -c(14)]
str(train_data1)
str(test_data1)

##rm(train_data1)
##str(test_data)
# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train_data1, destination_frame = "train_data1.hex")
test.hex <- as.h2o(x = test_data1, destination_frame = "test_data1.hex")

##rm(test.hex)
y = "income"
##rm(x)
x_train = setdiff(colnames(train.hex), y) ## y is dropped already
str(x_train)
x_test = setdiff(colnames(test.hex), y)
str(train.hex)

##rm(aec)
## extract features from deeplearning
aec <- h2o.deeplearning(x = x_train, autoencoder = T,
                        training_frame = train.hex,
                        activation = "Tanh",
                        hidden = c(20), epochs = 100)

aec_test <- h2o.deeplearning(x = x_test, autoencoder = T, 
                             training_frame = test.hex,
                             activation = "Tanh", hidden = c(20), epochs = 100)
# Extract features from train data
##rm(features_train)
features_train <- as.data.frame(h2o.deepfeatures(data = train.hex[,x_train], object = aec))
str(features_train)
# Extract features from test data
##rm(features_test)
features_test <- as.data.frame(h2o.deepfeatures(data = test.hex[,x_test], object = aec_test))
str(features_test)
#### auto encoders feature extraction done(non linarity), next step PCA(linearity)
#### feature extraction with PCA
## PCA needs numerical data so lets split numerical data set in test and train

str(num_stand)

## remove target income 
num_dataForPCA <- num_stand[, -c(2)]
str(num_dataForPCA)
## split into test and train

set.seed(1234)
train_index <- sample(x = nrow(num_dataForPCA), size =  0.6 * nrow(num_dataForPCA))

#rm(pca_train)
#rm(pca_test)
pca_train <- num_dataForPCA[train_index,]
pca_test  <-  num_dataForPCA[-train_index,]

str(pca_train)
## run PCA on train and test

pca_ml_tr <-  princomp(x = pca_train)
pca_ml_ts <-  princomp(x = pca_test)

pca_ml_tr$scores

## all the datasets to be used for model building
features_train
features_test
pca_ml_tr
pca_ml_ts
num_stand
new_data
train_data
train_data1
test_data
test_data1
str(train_data)
str(test_data)
#creating the data set with impt components
pca_train_data <- data.frame(pca_ml_tr$scores[,1:4])
pca_test_data <- data.frame(pca_ml_ts$scores[, 1:4])
# add extracted features with original data to train the model
# rm(train)
# rm(test)
train <- data.frame(features_train,pca_train_data, train_data)
test <- data.frame( features_test, pca_test_data, test_data )
names(train)
names(test)

str(new_data)

# run random forest to select most importnt features
##rm(rf_DL)
require(randomForest)
rf_DL <- randomForest(train$income ~ ., data=train, 
                      keep.forest=TRUE, ntree=30)

# importance of attributes
round(importance(rf_DL), 2)
##rm(importanceValues)
importanceValues = data.frame(attribute=rownames(round(importance(rf_DL), 2)),
                              MeanDecreaseGini=round(importance(rf_DL), 2))

##importanceValues = importanceValues[order(-importanceValues$MeanDecreaseGini),]

importanceValues = importanceValues[order(importanceValues$IncNodePurity, decreasing = TRUE),]
dim(importanceValues)
importanceValues[1:37,]

# Taking 38 Important attributes till sec_account
Top20ImpAttrs = as.character(importanceValues$attribute[1:38])
train_Imp = subset(train,select = c(Top20ImpAttrs, "income"))
test_Imp = subset(test,select = c(Top20ImpAttrs,  "income"))
names(train)
names(test)

### build models on train_imp and test it on test_imp
names(train_Imp)
names(test_Imp)

## now build model on train_imp and predict it on test_imp
## regression using CART

library(rpart)
dtCart = rpart(income ~.,data=train_Imp,method="anova")    
summary(dtCart)


plot(dtCart, main = "Decision Tree For Income",  uniform = TRUE)
text(dtCart, cex = 0.7, use.n = TRUE, xpd = TRUE)

## on test
predcartTrain_loan = predict(dtCart, newdata = train_Imp, type = "vector")

predcartTest_loan = predict(dtCart, newdata = test_Imp, type = "vector")

library(DMwR)
names(train)
new_data

## income
## regr.eval works wth numeric vectors
regr.eval(train[,"income"], predcartTrain_loan, train.y = train_Imp[,"income"])## rmse 0.63
regr.eval(test[,"income"], predcartTest_loan, train.y = test_Imp[,"income"]) ## rmse 1.82


## all the datasets to be used for model building
features_train
features_test
pca_ml_tr
pca_ml_ts
num_stand
new_data
train
test
names(train_data)
names(test_data)

## feature set

str(train_Imp)
test_Imp

##Linear Regression---- done
##Decision Tree(Regression Tree) -- done
##SVM ---- done
##Neural Network
##KNN--- done
##Ada-boost --- NA
##Random Forest---- done
##GBM----- done
##Deep Learning.
##then stacking


## regression using linear regression using all attributes
reg_model <- lm(income~.,   data = train_Imp)
summary(reg_model)

# backward stepAIC 
library(MASS)
LinRegAIC = stepAIC(reg_model, direction = "backward")

reg_model1 <- lm(income~ Comp.3 + Comp.1 + DF.L1.C14 + loan + mortgage + DF.L1.C10 + 
                   DF.L1.C8 + DF.L1.C13 + DF.L1.C16 + DF.L1.C2 + education1 + 
                   DF.L1.C17 + family4 + family3 + family2 + education3, data = train_Imp )

summary(reg_model1)

## error metric evaluation

library(DMwR)

regr.eval(train_Imp$income, reg_model1$fitted.values)
## mae       mse      rmse      mape 
## 0.5132310 0.4153512 0.6444775 2.0407893 

Pred<-predict(reg_model1,test_Imp)

regr.eval(test_Imp$income, Pred)
##mae       mse      rmse      mape 
##17.01937 410.65163  20.26454  85.29471 

## original linear

regr.eval(train_Imp$income, reg_model$fitted.values)

Pred<-predict(reg_model,test_Imp)
regr.eval(test_Imp$income, Pred)

## mae       mse      rmse      mape 
## 17.81210 453.95239  21.30616  90.57703 

######***********************#####

###************** SUPPORT VECTOR MACHINE(REGRESSION)********######
library(DMwR)
library(e1071)

train_Imp_bk <- train_Imp
test_imp_bk <- test_Imp
str(train_Imp)
str(test_Imp)
### loan is a  factor which needs to be converted to numeric
library(dummies)
dm_loan_tr = dummy(train_Imp$loan)
dm_loan_ts = dummy(test_Imp$loan)
dm_sc_aact_tr = dummy(train_Imp$sec_acct)
dm_sc_aact_ts = dummy(test_Imp$sec_acct)
dm_online_tr = dummy(train_Imp$online)
dm_online_ts = dummy (test_Imp$online)
str(train_Imp)
str(test_Imp)
dim(train_Imp)
train_Imp[ 7]
#rm(test_imp_2)
train_imp_1 = train_Imp[-c(7, 37,38)]
test_imp_2 = test_Imp[-c(7, 37, 38)]
str(train_imp_1)
str(test_imp_2)

## combine
train_imp_svm = cbind(train_imp_1,dm_sc_aact_tr, dm_loan_tr, dm_online_tr )
test_imp_svm = cbind(test_imp_2, dm_sc_aact_ts, dm_loan_ts, dm_online_ts)

str(train_imp_svm)
str(test_imp_svm)
str(train_Imp)
str(test_Imp)

## append y variable
train_imp_svm$income = NULL
test_imp_svm$income = NULL

dim(train_imp_svm)
train_imp_svm = data.frame(cbind(train_imp_svm , train_Imp$income))
test_imp_svm = data.frame(cbind(test_imp_svm, test_Imp$income))

setnames(train_imp_svm, old=c("train_Imp.income"), new=c("income"))
setnames(test_imp_svm, old=c("test_Imp.income"), new=c("income"))

### Build best SVM model 
svm_model = svm(x = train_imp_svm[,1:41], 
                y = train_imp_svm[,42], 
                type = "nu-regression", 
                kernel = "linear", cost = 1e-7) 


# Look at the model summary
summary(svm_model)

# Predict on train data and check the performance
regr.eval(train_imp_svm$income, predict(svm_model, train_imp_svm[,1:41]))
## mae       mse      rmse      mape 
## 0.7789204 0.9822904 0.9911056 1.4094546 

## mae      mse     rmse     mape 
## 0.809533 1.078284 1.038404 1.393388 

# Predict on test data and check the performance  
regr.eval(test_imp_svm$income, predict(svm_model, test_imp_svm[,1:41]))
pred_svm = predict(svm_model, test_imp_svm[,1:41])
########**** KNN ******###

library(vegan)
library(dummies)
library(FNN)
library(DMwR)
library(class)

str(train_imp_svm)
# k = 1
pred_Train_Knn = knn(train_imp_svm[,1:41], 
                     train_imp_svm[,1:41], 
                     train_imp_svm$income, k = 1)

cm_Train = table(pred_Train_Knn, train_imp_svm$income)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)

pred_Test = knn(test_imp_svm[,1:41], 
                test_imp_svm[,1:41], 
                test_imp_svm$income, k = 1)

cm_Test = table(pred_Test, test_imp_svm$income)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)

# k = 3
pred_Train_Knn = knn(train_imp_svm[,1:41], 
                     train_imp_svm[,1:41], 
                     train_imp_svm$income, k = 3)

cm_Train = table(pred_Train_Knn, train_imp_svm$income)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)

pred_Test = knn(test_imp_svm[,1:21], 
                test_imp_svm[,1:21], 
                test_imp_svm$income, k = 3)

cm_Test = table(pred_Test, test_imp_svm$income)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
##0.3586667
# Test the final model performance on evaluation data and report the results
pred_Test = knn(test_imp_svm[,1:21], 
                test_imp_svm[,1:21], 
                test_imp_svm$income, k = 1)

cm_Test = table(pred_Test, test_imp_svm$income)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
cm_Test
accu_Test
## 0.995 on test
###################*************Random Forest**********###

# Build the classification model using randomForest
str(train_Imp)
rf_model = randomForest(income ~ ., data=train_Imp, 
                        keep.forest=TRUE, ntree=100) 

# Print 
print(rf_model)


# Predict on Train data 
pred_Train = predict(rf_model, 
                     train_Imp[,setdiff(names(train_Imp), "income")],
                     type="response", 
                     norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Train = table("actual"= train_Imp$income, "predicted" = pred_Train);
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
##rm(pred_Train, cm_Train)

str(train_Imp)
str(train)
# Predicton Test Data
pred_Test = predict(rf_model, 
                    test_Imp[,setdiff(names(test_Imp), "income")],
                    type="response", 
                    norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Test = table("actual"= test_Imp$income, "predicted" = pred_Test);
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
##rm(pred_Test, cm_Test)

accu_Train
accu_Test
## 0.002
## used random forest to extract features; do not need to build rf model on top of that,
## accuracy is bad

####GBM
##Deep Learning.
## neural net

###### ******************** GBM ********************************#######
## all the datasets to be used for model building
#####################********************************************###############
# Load H2o library
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "2g")

# Import a local R train data frame to the H2O cloud
##train.hex <- as.h2o(x = train_data, destination_frame = "train.hex")
##rm(train.hex)
str(train_Imp)
str(train)
train.hex <- as.h2o(x = train_Imp, destination_frame = "train.hex")
# Build H2O GBM
gbm <- h2o.gbm(model_id = "GBM.hex", ntrees = 100, 
               learn_rate=0.01, max_depth = 4,   
               y = "income", x = setdiff(names(train.hex), "income"),
               training_frame = train.hex)

# Get the auc of the GBM model
h2o.auc(gbm)

# Examine the performance of the best model
gbm

# Important Variables.
h2o.varimp(gbm)

# Import test data frame to the H2O cloud
test.hex <- as.h2o(x = test_Imp, destination_frame = "test.hex")
str(test_data)

# Predict on same training data set
predict.hex = h2o.predict(gbm,newdata = test.hex[,setdiff(names(test.hex), "income")])
str(test.hex)
data_GBM = h2o.cbind(test.hex[,"income"], predict.hex)

# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)


# Shutdown H2O
h2o.shutdown(F)

# Hit Rate and Penetration calculation
conf_Matrix_GBM = table(pred_GBM$income, pred_GBM$predict) 

Accuracy = (conf_Matrix_GBM[1,1]+conf_Matrix_GBM[2,2])/sum(conf_Matrix_GBM)
##0 ?????

###----------------------neural network for regression--------------
model_nn =nnet(income~.,train_Imp,linout=T,size=4)
## prediction on train
pred_train_nn=predict(object = model_nn,newdata = train_Imp)
regr.eval(pred_train_nn,train_Imp$income)
###  mae       mse      rmse      mape 
#### 0.4489138 0.3296934 0.5741893 4.1702291 
##prediction on test
pred_nn = predict(object = model_nn,newdata = test_Imp)
regr.eval(pred_nn,test_Imp$income)
##   mae      mse     rmse     mape 
### 1.562621 3.855585 1.963564 1.726851 

###----------------------deep learning or regression----------------
h2o.init(ip='localhost', port=54321, max_mem_size = '1g')

train.hex=as.h2o(x = train_Imp,destination_frame = "train.hex")
test.hex=as.h2o(x = test_Imp,destination_frame = "test.hex")
model_dl =h2o.deeplearning(x = setdiff(names(train.hex),"income"),y = "income"
                           , training_frame = train.hex,model_id = "model_dl.hex"
                           , activation = "Maxout",hidden = c(30,20)
                           ,input_dropout_ratio = 0.2,epochs = 50,seed = 123)
## pred on train
pred.hex=h2o.predict(model_dl,newdata = train.hex[,setdiff(names(train.hex), "income")])
pred_dl =as.data.frame(pred.hex)
regr.eval(pred_dl,train_Imp$income)
## mae        mse       rmse       mape 
##1385.82431 1155.01290   33.98548 5180.39911 

## pred on test
pred.hex=h2o.predict(model_dl,newdata = test.hex[,setdiff(names(test.hex), "income")])
pred_dl =as.data.frame(pred.hex)
regr.eval(pred_dl,test_Imp$income)

##  mae         mse        rmse        mape 
## 1887.89764  2633.48328    51.31748 10985.88967
## test is overfitting

#####-----------stacking of predicted values.................
#................lm as meta learner...............................

###Combining training predictions of CART, C5.0 & Log Regression together
##rm(train_Pred_Models)

train_Pred_Models = cbind( income = test_Imp$income, cart = predcartTest_loan, 
                                reg = Pred , svm = pred_svm, 
                                knn = pred_Test , GBM = pred_GBM, NN = pred_nn,
                                dl =  pred_dl  )


#####-----------stacking of predicted values.................
#................lm as meta learner...............................
names(model2)
reg_model2 <- lm(income~.,  data =   train_Pred_Models )
final_reg=lm(formula = INC~.,data = final_predR)
pred_final= predict(object = reg_model2, newdata =  train_Pred_Models)
regr.eval(pred_final,train_Pred_Models$income)
regression=data.frame(cbind(train_Pred_Models$income,pred_final))























rm(list = ls(all = TRUE))

#Setting the working directory
getwd()

#reading the UnivBank data csv file
data = read.table("UnivBank.txt", header = TRUE, sep = ',')
str(data)
names(data)

#Removing "ID","ZIP.Code" and "Experience"
drop_Attr = c("ID","ZIP.Code","Experience")
attr = setdiff(names(data),drop_Attr)
data = data[,attr]
rm(drop_Attr)

# Convert attribute to appropriate type  
cat_Attr = c("Family", "Education", "Securities.Account", 
             "CD.Account", "Online", "CreditCard", "Personal.Loan")
num_Attr = setdiff(attr, cat_Attr)
rm(attr)

cat_Data <- data.frame(sapply(data[,cat_Attr], as.factor))
num_Data <- data.frame(sapply(data[,num_Attr], as.numeric))

data = cbind(num_Data,cat_Data)

#checking for NA values
sum(is.na(data[])) #value is 0

#Using the Numerical attributes for PCA
data_PCA = data[,num_Attr]
head(data_PCA)

library(vegan)
# Normalizing the data
data_PCA = decostand(data_PCA,method = "range")
head(data_PCA)

#Separating Train data and Test data
set.seed(123)
train_RowIDs_PCA = sample(1:nrow(data_PCA), nrow(data_PCA)*0.7)
train_PCA = data_PCA[train_RowIDs_PCA,]
test_PCA = data_PCA[-train_RowIDs_PCA,]
head(train_PCA)

#Running PCA on Train and test data
pca_traindata <- prcomp(train_PCA)
pca_testdata <- prcomp(test_PCA)

summary(pca_traindata)
summary(pca_testdata)

#Feature selection based on PCA
#Since there is a cumulative variance of 92.76 for 3 PCA components only those are selected
pca_traindata <- pca_traindata$x[,1:3]
pca_testdata <- pca_testdata$x[,1:3]

#Separating Train data and Test data for Auto-encoders
set.seed(123)
train_RowIDs = sample(1:nrow(data), nrow(data)*0.7)
train_Data = data[train_RowIDs,]
test_Data = data[-train_RowIDs,]
#Non-Linear feature selection using Auto-encoder
library(h2o)

h2o.init(ip='localhost', port=54321, max_mem_size = '1g')

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train_Data, destination_frame = "train_Data.hex")
test.hex <- as.h2o(x = test_Data, destination_frame = "test_Data.hex")

y = "Personal.Loan"
x = setdiff(colnames(train.hex), y)


## extract features from deeplearning
aec <- h2o.deeplearning(x = x, autoencoder = T,
                        training_frame = train.hex,
                        activation = "Tanh",
                        hidden = c(20), epochs = 100)

# Extract features from train data
features_train <- as.data.frame(h2o.deepfeatures(data = train.hex[,x], object = aec, layer = 0))

# Extract features from test data
features_test <- as.data.frame(h2o.deepfeatures(data = test.hex[,x], object = aec, layer = 0))

#Combining all the train and test features extracted with the Original dataset
train <- data.frame(train_Data,train_PCA,features_train)
test <- data.frame(test_Data, test_PCA, features_test)
names(train)
names(test)

h2o.shutdown(F)

#Using Random Forest to extract the important attributes
require(randomForest)
rf_Model <- randomForest(train$Personal.Loan ~ ., data = train, keep.forest = TRUE, ntree = 30)

print(rf_Model)

#Importance of attributes
rf_Model$importance
round(importance(rf_Model),2)

varImpPlot(rf_Model)

rf_Imp_Attr=data.frame(rf_Model$importance)
rf_Imp_Attr=data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr)=c('Attributes','Importance')
rf_Imp_Attr=rf_Imp_Attr[order(rf_Imp_Attr$Importance,decreasing = TRUE),]

#Top 30 Important Attributes
Top20ImpAttrs =  as.character(rf_Imp_Attr$Attributes[1:30])

#Final Train and Test data set
train_Imp = subset(train,select = c(Top20ImpAttrs, "Personal.Loan"))
test_Imp = subset(test,select = c(Top20ImpAttrs,  "Personal.Loan"))
names(train)
names(test)
str(train_Imp)
str(test_Imp)

#Building Classification Model using Logistic Regression
BankLogModel = glm(Personal.Loan~., data = train_Imp, family = binomial)
summary(BankLogModel)

#Accuracy on the training set
predictTrainLog = predict(BankLogModel, type = "response", newdata = train_Imp)

# Confusion matrix with threshold of 0.5
cmTrainLog = table(train_Imp$Personal.Loan, predictTrainLog > 0.5)
# Accuracy on Train Set
AccuracyTrainLog = sum(diag(cmTrainLog))/sum(cmTrainLog)
AccuracyTrainLog #0.9697143

# Predictions on the test set
predictTestLog = predict(BankLogModel, type="response", newdata=test_Imp)

# Confusion matrix with threshold of 0.5
cmTestLog = table(test_Imp$Personal.Loan, predictTestLog > 0.5)

# Accuracy on Test Set
AccuracyTestLog = sum(diag(cmTestLog))/sum(cmTestLog)
AccuracyTestLog #0.9666667

#rm(BankLogModel,predictTrainLog,predictTestLog,cmTrainLog,cmTestLog,AccuracyTestLog,AccuracyTrainLog)

#Building classification model using C5.0(Decision Tree)
library(C50)
BankC50 = C5.0(Personal.Loan~., data = train_Imp, rules = TRUE)
summary(BankC50)
C5imp(BankC50,pct = TRUE)

#Calculating recall for Train and Test
predict_C50_Train = predict(BankC50, newdata = train_Imp, type = "class")
cm_C50_Train =  table(train_Imp$Personal.Loan, predict_C50_Train )
recall_C50_Train = cm_C50_Train[2,2]/(cm_C50_Train[2,1]+cm_C50_Train[2,2]) 
recall_C50_Train #0.9365559

predict_C50_Test = predict(BankC50, newdata = test_Imp, type = "class")
cm_C50_Test =  table(test_Imp$Personal.Loan, predict_C50_Test )
recall_C50_Test = cm_C50_Test[2,2]/(cm_C50_Test[2,1]+cm_C50_Test[2,2]) 
recall_C50_Test #0.8590604

#rm(BankC50,predict_C50_Test,predict_C50_Train,cm_C50_Test,cm_C50_Train,recall_C50_Test,recall_C50_Train)

#Building classification model using CART(Decision Tree)
library(rpart)
library(rpart.plot)
BankCART = rpart(Personal.Loan~., data = train_Imp, method = "class")
plot(BankCART,main="Classification Tree for loan Class",margin=0.15,uniform=TRUE)
text(BankCART,use.n=T)

summary(BankCART)

#Calculating recall for Train and Test
predict_CART_Train = predict(BankCART, newdata = train_Imp, type = "class")
cm_CART_Train =  table(train_Imp$Personal.Loan, predict_CART_Train )
recall_CART_Train = cm_CART_Train[2,2]/(cm_CART_Train[2,1]+cm_CART_Train[2,2]) 
recall_CART_Train #0.8519637

predict_CART_Test = predict(BankCART, newdata = test_Imp, type = "class")
cm_CART_Test =  table(test_Imp$Personal.Loan, predict_CART_Test )
recall_CART_Test = cm_CART_Test[2,2]/(cm_CART_Test[2,1]+cm_CART_Test[2,2]) 
recall_CART_Test #0.852349

#rm(BankCART,predict_CART_Test,predict_CART_Train,cm_CART_Test,cm_CART_Train,recall_CART_Test,recall_CART_Train)

#Building classification model using SVM
library(dummies)

train_svm = train_Imp
test_svm = test_Imp
#Converting categorical attributes to numeric
train_svm$Education = as.numeric(train_svm$Education)
train_svm$Family = as.numeric(train_svm$Family)
test_svm$Education = as.numeric(test_svm$Education)
test_svm$Family = as.numeric(test_svm$Family)


str(train_svm)
str(test_svm)

#Applying the SVM model
library(e1071)

model = svm(x = train_svm[,1:20], 
            y = train_svm$Personal.Loan, 
            type = "C-classification", 
            kernel = "linear", cost = 10, gamma = 0.1) 
summary(model)

# Predict on train data  
pred_Train  =  predict(model, train_svm[,1:20])  

# Build confusion matrix and find accuracy   
cm_Train = table(train_svm$Personal.Loan, pred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
accu_Train #0.9691429
#rm(pred_Train, cm_Train,accu_Train)

# Predict on test data
pred_Test = predict(model, test_svm[,1:20]) 

# Build confusion matrix and find accuracy   
cm_Test = table(test_svm$Personal.Loan, pred_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
accu_Test #0.9693333
#rm(pred_Test, cm_Test,accu_Test)

#Building a KNN model for classification
train_knn = train_Imp
test_knn = test_Imp
#Converting categorical attributes to numeric
train_knn$Education = as.numeric(train_knn$Education)
train_knn$Family = as.numeric(train_knn$Family)
test_knn$Education = as.numeric(test_knn$Education)
test_knn$Family = as.numeric(test_knn$Family)

str(train_knn)
str(test_knn)

# Check how records are split with respect to target attribute.
table(train_knn$Personal.Loan)
table(test_knn$Personal.Loan)

library(FNN)
#Building the best KNN model
#k = 1
pred_Train_KNN = knn(train_knn[,1:20], 
                 train_knn[,1:20], 
                 train_knn$Personal.Loan, k = 1)

cm_Train_knn = table(pred_Train_KNN, train_knn$Personal.Loan)
accu_Train_knn= sum(diag(cm_Train_knn))/sum(cm_Train_knn)
rm(pred_Train_KNN, cm_Train_knn)

pred_Test_KNN = knn(train_knn[,1:20], 
                test_knn[,1:20], 
                train_knn$Personal.Loan, k = 1)

cm_Test_knn = table(pred_Test_KNN, test_knn$Personal.Loan)
accu_Test_knn= sum(diag(cm_Test_knn))/sum(cm_Test_knn)
rm(pred_Test_KNN, cm_Test_knn)

accu_Train_knn #1
accu_Test_knn #0.9573333
rm(accu_Test_knn,accu_Train_knn)

#k = 3
pred_Train_KNN = knn(train_knn[,1:20], 
                     train_knn[,1:20], 
                     train_knn$Personal.Loan, k = 3)

cm_Train_knn = table(pred_Train_KNN, train_knn$Personal.Loan)
accu_Train_knn= sum(diag(cm_Train_knn))/sum(cm_Train_knn)
rm(pred_Train_KNN, cm_Train_knn)

pred_Test_KNN = knn(train_knn[,1:20], 
                    test_knn[,1:20], 
                    train_knn$Personal.Loan, k = 3)

cm_Test_knn = table(pred_Test_KNN, test_knn$Personal.Loan)
accu_Test_knn= sum(diag(cm_Test_knn))/sum(cm_Test_knn)
##rm(pred_Test_KNN, cm_Test_knn)

accu_Train_knn #0.9688571
accu_Test_knn #0.958
##rm(accu_Test_knn,accu_Train_knn)

#k = 5
pred_Train_KNN = knn(train_knn[,1:20], 
                     train_knn[,1:20], 
                     train_knn$Personal.Loan, k = 5)

cm_Train_knn = table(pred_Train_KNN, train_knn$Personal.Loan)
accu_Train_knn= sum(diag(cm_Train_knn))/sum(cm_Train_knn)


pred_Test_KNN = knn(train_knn[,1:20], 
                    test_knn[,1:20], 
                    train_knn$Personal.Loan, k = 5)

cm_Test_knn = table(pred_Test_KNN, test_knn$Personal.Loan)
accu_Test_knn= sum(diag(cm_Test_knn))/sum(cm_Test_knn)


accu_Train_knn #0.9585714
accu_Test_knn #0.9546667


pred_Train_KNN_5 = pred_Train_KNN
pred_Test_KNN_5 = pred_Test_KNN
rm(pred_Train_KNN, cm_Train_knn)
rm(pred_Test_KNN, cm_Test_knn)
rm(accu_Test_knn,accu_Train_knn)

#k = 7
pred_Train_KNN = knn(train_knn[,1:20], 
                     train_knn[,1:20], 
                     train_knn$Personal.Loan, k = 7)

cm_Train_knn = table(pred_Train_KNN, train_knn$Personal.Loan)
accu_Train_knn= sum(diag(cm_Train_knn))/sum(cm_Train_knn)
rm(pred_Train_KNN, cm_Train_knn)

pred_Test_KNN = knn(train_knn[,1:20], 
                    test_knn[,1:20], 
                    train_knn$Personal.Loan, k = 7)

cm_Test_knn = table(pred_Test_KNN, test_knn$Personal.Loan)
accu_Test_knn= sum(diag(cm_Test_knn))/sum(cm_Test_knn)
rm(pred_Test_KNN, cm_Test_knn)

accu_Train_knn #0.9517143
accu_Test_knn #0.946
rm(accu_Test_knn,accu_Train_knn)

#Choosing K = 5 because accuracy of train and test are almost equal
# Condensing train Data
library(class)
keep = condense(train_knn, train_knn$Personal.Loan)
length(keep)
#nrow(train_knn)
keep

pred = knn(train_knn[keep,1:20], 
           test_knn[,1:20], 
           train_knn$Personal.Loan[keep], k=5)

cm <- table(pred, test_knn$Personal.Loan)
cm
accu=sum(diag(cm))/sum(cm)
accu #0.9266667
rm(cm,accu)
rm(accu_Test_knn,accu_Train_knn)

#Building a Classification Model using Ada-Boost
train_ada = train_Imp
test_ada = test_Imp
#Converting categorical attributes to numeric
train_ada$Education = as.numeric(train_ada$Education)
train_ada$Family = as.numeric(train_ada$Family)
test_ada$Education = as.numeric(test_ada$Education)
test_ada$Family = as.numeric(test_ada$Family)

str(train_ada)
str(test_ada)

# Check how records are split with respect to target attribute.
table(train_ada$Personal.Loan)
table(test_ada$Personal.Loan)

library(ada)
#Building ada boost model
model_ada = ada(x = train_ada[,1:20], y = train_ada$Personal.Loan, iter = 20, loss = "logistic")

pred_Train_ada = predict(model_ada,train_ada[,1:20])
cm_Train_ada = table(train_ada$Personal.Loan,pred_Train_ada)
accu_Train_ada = sum(diag(cm_Train_ada))/sum(cm_Train_ada) ## 0.9942857
#rm(cm_Train_ada, pred_Train_ada)

pred_Test_ada = predict(model_ada,test_ada[,1:20])
cm_Test_ada = table(test_ada$Personal.Loan,pred_Test_ada)
accu_Test_ada = sum(diag(cm_Test_ada))/sum(cm_Test_ada)
#0.9773333

##rm(accu_Test_ada,accu_Train_ada)

#Building a classification model on GBM
# Load H2o library
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "1g")

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train_Imp, destination_frame = "train.hex")


# Build H2O GBM
gbm <- h2o.gbm(model_id = "GBM.hex", ntrees = 100, 
               learn_rate=0.01, max_depth = 4,   
               y = "Personal.Loan", x = setdiff(names(train.hex), "Personal.Loan"),
               training_frame = train.hex)

# Get the auc of the GBM model
h2o.auc(gbm) #0.9985223

# Examine the performance of the best model
gbm

# Important Variables.
h2o.varimp(gbm)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = test_Imp, destination_frame = "test.hex")


# Predict on same training data set
predict.hex = h2o.predict(gbm,newdata = test.hex[,setdiff(names(test.hex), "Personal.Loan")])

data_GBM = h2o.cbind(test.hex[,"Personal.Loan"], predict.hex)

# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)
head(pred_GBM)

# Shutdown H2O
h2o.shutdown(F)

# Hit Rate and Penetration calculation
conf_Matrix_GBM = table(pred_GBM$Personal.Loan, pred_GBM$predict) 

Accuracy = (conf_Matrix_GBM[1,1]+conf_Matrix_GBM[2,2])/sum(conf_Matrix_GBM)
Accuracy #0.9846667

#Classification using Neural network
train_nn = train_Imp
test_nn = test_Imp
#Converting categorical attributes to numeric
train_nn$Education = as.numeric(train_nn$Education)
train_nn$Family = as.numeric(train_nn$Family)
test_nn$Education = as.numeric(test_nn$Education)
test_nn$Family = as.numeric(test_nn$Family)

library(nnet)
model_nn=nnet(Personal.Loan~.,train_nn,size=5)
summary(model_nn)
pred_nn_train=predict(object = model_nn,newdata = train_nn,type = "class")
cm_nn_train=table(Actual=train_nn$Personal.Loan,Predicted=pred_nn_train)
accu_nn_train = sum(diag(cm_nn_train))/sum(cm_nn_train) #0.9722857
pred_nn_test=predict(object = model_nn,newdata = test_nn,type = "class")
cm_nn_test = table(Actual=test_nn$Personal.Loan,Predicted=pred_nn_test)
accu_nn_test = sum(diag(cm_nn_test))/sum(cm_nn_test) #0.9706667

#Stacking for classification
predictTestLog=ifelse(predictTestLog>0.5,1,0)
prediction_final = data.frame("loan" = test_Imp$Personal.Loan, "GLM" = predictTestLog, "C50" = predict_C50_Test,
                               "CART"= predict_CART_Test, "NN"= pred_nn_test, "GBM" = pred_GBM$predict,
                               "SVM" = pred_Test, "KNN" = pred_Test_KNN, "Ada"= pred_Test_ada)
str(prediction_final)
prediction_final$GLM = as.factor(prediction_final$GLM)
#-------------------Stacking with glm as meta learner---------------------------------------------
model_glm=glm(loan~.,data = prediction_final,family = binomial)

classification=predict(object = model_glm,newdata = prediction_final)
classification=ifelse(classification>0.5,1,0)
table(test_Imp$Personal.Loan,classification)
final=data.frame(cbind(test_Imp$Personal.Loan,classification))





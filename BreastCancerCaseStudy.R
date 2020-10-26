# File: EARIN miniproyect 2
# Authors: Alejandro Cirugeda 
# 
# Description:
# The task of this project is to make a case study of the different classifiers and techniques
# used in datamining and machine learning and the advantages and disadvantages of each one of them.
# In order to do that we will using a Breast Cancer Dataset. After the case study we will be able to
# distinguish the best technique to create a classifier for this data for distinguish between malign
# and benign tumours.

#Data: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

library(caret)
library(party)
library(rpart)
library(e1071)
library(ggplot2)
library(randomForest)
library(mlbench)
library(rpart.plot)
library(ROCR)

# We will be using BreastCancer dataset form mlbech pacakge which is the same one 
# from UCI but with headers and NA values removed.
data(BreastCancer)

head(BreastCancer)

#      Attribute                     Domain
#--------------------------------------------
#  1.  ID                            id number
#  2.  Cl.thickness                  1 - 10
#  3.  Cell Size                     1 - 10
#  4.  Cell Shape                    1 - 10
#  5.  Marginal Adhesion             1 - 10
#  6.  Single Epithelial Cell Size   1 - 10
#  7.  Bare Nuclei                   1 - 10
#  8.  Bland Chromatin               1 - 10
#  9.  Normal Nucleoli               1 - 10
#  10. Mitoses                       1 - 10
#  11. Class                         Bening - Malign

summary(BreastCancer)

barplot(prop.table(table(BreastCancer$Class)), col=c("#99CCFF","#FFCC99"))


#### Data preparation #######
# In order to improve the results of te classifiers we need to prepare the data
# First we ommit the empty values because can produce some errors in some classifier
BreastCancer <- na.omit(BreastCancer)

# We convert the class attribute to categorical
BreastCancer$Class <- factor(BreastCancer$Class)

# In this dataset we have a Id collumn that wont be usefull, therefore we remove it
BreastCancer$Id <- NULL 

# Also we use preprocesing in order to improve the outcome of the differents classifiers:
procMethod <- preProcess(BreastCancer, method = c("center","scale"));
BreastCancer <- predict(procMethod,BreastCancer)



# #### DATA SPLIT #####
# Now we partition the data between trainset and testset using a vector variable called ind.
# Each split will be assign a value.
# trainset will be 75%    --  ind == 1     
# testset will be  25%    --  ind == 2
set.seed(12345)
ind <- sample(2, nrow(BreastCancer), replace = TRUE, prob=c(0.75, 0.25))



# We need to have both classes in each split of the data in order for the creation of correct models
table(BreastCancer[ind == 1,]$Class)
# benign = 321
# malign = 174

table(BreastCancer[ind == 2,]$Class)
# benign = 123
# malign = 65

# ################### CTREES ########################
#Decision trees are simple top-down models in which each node in the tree creates a binary split which after a 
#succession of internal nodes we arrived at the terminal node which the prediction is made.

model_ctree <- ctree(Class ~ ., data=BreastCancer[ind == 1,])

plot(model_ctree, main="Decision tree with ctrees")

model_ctree.prediction <- predict(model_ctree, newdata=BreastCancer[ind == 2,])
confusionMatrix(model_ctree.prediction, BreastCancer[ind == 2,]$Class)
#accuracy = 0.9468

# Now we represent the results with a ROC curve
model_ctree.prob <-  1- unlist(treeresponse(model_ctree, BreastCancer[ind == 2,]), use.names=F)[seq(1,nrow(BreastCancer[ind == 2,])*2,2)]
model_ctree.prob.rocr <- prediction(model_ctree.prob, BreastCancer[ind == 2,'Class'])
model_ctree.perf <- performance(model_ctree.prob.rocr, measure = "tpr",x.measure = "fpr")

plot(model_ctree.perf, col=2, main="ROC Curve ctree")



# ################## RPART ######################
# recursive partitioning and regression trees.  Categorical or continuous variables can be used depending on
#whether one wants classification trees or regression trees. we will used rpart library.
model_rpart <- rpart(Class ~ .,data = BreastCancer[ind == 1,])
rpart.plot(model_rpart, main="Decision tree with rpart")

model_rpart.prediction <- predict(model_rpart, newdata=BreastCancer[ind == 2,])


# Have to change here the form of confusion matrix due an error with ConfusionMatrix function
model_rpart.pred <- predict(model_rpart, newdata=BreastCancer[ind == 2,], type = 'class')
confusionMatrix(table(model_rpart.pred, BreastCancer[ind == 2,]$Class))
#accuracy = 0.9202

#Now we represent the roc curve:
pred <- prediction(model_rpart.prediction[, 2], BreastCancer[ind == 2,]$Class)
model_rpart.perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(model_rpart.perf, col = 3, main = "ROC Curve rpart")


# ################## RANDOM FOREST #################
# Now we will use a technique call Random Forest
#Random forest is very similar to bagging, the main different is that they used a modified tree 
#learning algorithm which decided a random subset of features. For this classification we will use the
#“caret” and “randomForest” libraries
model_rf <- cforest(Class ~ ., data=BreastCancer[ind == 1,], control = cforest_unbiased(mtry = ncol(BreastCancer)-2))
model_rf.prediction <- predict(model_rf, newdata=BreastCancer[ind == 2,])

# Now we calculate the confusion matrix
confusionMatrix(model_rf.prediction, BreastCancer[ind == 2,]$Class)
#accuracy = 0.9628

model_rf.prob <-  1- unlist(treeresponse(model_rf, BreastCancer[ind == 2,]), use.names=F)[seq(1,nrow(BreastCancer[ind == 2,])*2,2)]

model_rf.prob.rocr <- prediction(model_rf.prob, BreastCancer[ind == 2,'Class'])
model_rf.perf <- performance(model_rf.prob.rocr, measure = "tpr",x.measure = "fpr")
plot(model_rf.perf, col=4, main = "random forest ROC Curve")



# ################ BAYER CLASSIFIER ##################
#Naive Bayes is a supervised learning classification algorithm based on Bayes Theorem.
#It is an extension of Exact Bayes classifier and assumes all variables are independent of each other.
#This method is suitable when dimensionality of inputs is high. It uses a simple method of conditional probability 
#and can be computationally inexpensive and outperform more sophisticated models. We will use the “e1071” library
#for this task

model_bayer<- naiveBayes(Class ~ ., BreastCancer[ind == 1,])

model_bayer.prediction <- predict(model_bayer, newdata=BreastCancer[ind == 2,])

# Now we calculate the confusion matrix
confusionMatrix(model_bayer.prediction, BreastCancer[ind == 2,]$Class)
#accuracy = 0.9628


model_bayer.prob <- predict(model_bayer, type="raw", newdata=BreastCancer[ind == 2,], probability = TRUE)


model_bayer.prob.rocr <- prediction(model_bayer.prob[,2], BreastCancer[ind == 2,'Class'])
model_bayer.perf <- performance(model_bayer.prob.rocr, measure = "tpr",x.measure = "fpr")
plot(model_bayer.perf, col=5, main = "Bayer ROC Curve")




# ##################### SVM ##############################
#Support Vector Machines generates a partition of the data produce by a hyperlane. This line is produce using all
#the attributes of the data and by a clear gap that should be as wide as possible. For this classifier the library
#“e1071” will used.

#For the svm classifier firts we need to tune it to discover the optimal parameters
model_svm.tune <- tune(svm, Class~., data = BreastCancer[ind == 1,],
                       ranges = list(gamma = 2^(-8:1), cost = 2^(0:4)),
                       tunecontrol = tune.control(sampling = "fix"))

model_svm.tune # We obtain the tune parameter and use it in the svm classifier
# cost = 1
# gamma = 0.125


# and use it for the creation of the model
model_svm <- svm(Class~., data = BreastCancer[ind == 1,], cost=1, gamma=0.125, probability = TRUE)
model_svm.prediction <- predict(model_svm, newdata=BreastCancer[ind == 2,])



# and calculate the confusion matrix
confusionMatrix(model_svm.prediction, BreastCancer[ind == 2,]$Class)
#accuracy = 0.9521

model_svm.prob <- predict(model_svm, type="prob", newdata=BreastCancer[ind == 2,], probability = TRUE)
model_svm.prob.rocr <- prediction(attr(model_svm.prob, "probabilities")[,2], BreastCancer[ind == 2,'Class'])
model_svm.perf <- performance(model_svm.prob.rocr, measure = "tpr",x.measure = "fpr")


plot(model_svm.perf, col=6, main="SVM ROC Curve")



# ######## COMPARATION OF THE CLASSIFIERS ########
# Now we can plot the different results of the classifier in the same graph in order to compare them

plot(model_ctree.perf, col=2, main="comparation ROC curves", lwd=2) # ctree
plot(model_rpart.perf, col=3, add=TRUE, , lwd=2)                      # rpart
plot(model_rf.perf, col=4, add=TRUE, lwd=2)                         # random forest
plot(model_svm.perf, col=5, add=TRUE, lwd=2)                        # svm
plot(model_bayer.perf, col=6, add=TRUE, lwd=2)                      # bayer
# the legend
legend(0.6, 0.8, c("ctree", "rpart", "rand forest", "SVM", "Bayer"), 2:6)



# Conclusion:
# As we can see in the final graph both Support Vector Machines and Bayer classifier have very similar
# ROC curves. But if we check the accuracy of the classifiers we can see that bayer has 96.28%
# against 95.21% of the SVM.
#
# Therefore we can conclude that for this dataset the best classifier is Naive Bayes


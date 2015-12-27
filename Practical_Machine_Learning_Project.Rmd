---
title: "Practical Machine Learning Project"
author: "HD"
date: "December 24, 2015"
output: html_document
---

# Executive Summary

This project analyzed data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants as they perform barbell lifts correctly and incorrectly in 5 different ways (Classes A-E). Machine learning models were built to predict the manner in which the exercise were performed. The expected accuracy of the final model is *98.5%* and the overall out of sample error is *1.5%*. It accurately predicted the classification of 20 observations in the testing data set.

## Background 

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.  

Six participants performed barbell lifts in 5 different ways: *class A* corresponds to the correct execution of the exercise, while *classes B-E* correspond to the incorrect execution of the exercise. 

## Data

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

The training data for this project are from here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are from here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Pre-loading the R packages

```{r}
library(caret, quietly = TRUE)
library(corrplot)
library(randomForest)
```
# A. Getting and Cleaning the Data

## A.1 Downloading and uploading the data

Download the data from the links to the current working directory: 
```{r}
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

Upload the files to the memory, setting the missing data (i.e., NA, #DIV/0!, and "") as NA:

```{r}
training<- read.csv("train.csv", 
                    na.strings = c("NA", "#DIV/0!", ""))
testing<- read.csv("test.csv", 
                    na.strings = c("NA", "#DIV/0!", ""))
```

## A.2 Cleaning near zero variables

```{r}
# returns the positions of the zero or near zero variables
NZV<- nearZeroVar(training) 
# delete the near zero variables
training<- training[, -NZV] 
testing<- testing[, -NZV]
```

## A.3 Deleting variables with more than *60%* *NAs*

```{r}
# calculate the percentage of NAs for each variable
mostNA<- sapply (training, function(x) mean(is.na(x))) > 0.6 
# delete the based on the variable value
training<- training[, mostNA==FALSE]
testing<- testing[, mostNA==FALSE]
```

## A.4 Deleting the variables for information purpose

Take a quick look at the training data:

```{r}
str(training, list.len=10)
```

The first 6 variables (*X*, *user_name*, *raw_timestamp_part_1*, *raw_timestamp_part_2*, *cvtd_timestamp*, and *num_window*) are for information purpose, therefore, delete them.

```{r}
training<- training[, -(1:6)]
testing<- testing[, -(1:6)]
```

# B. Slice the Data for Cross Validation
For the purpose of cross validation, split the cleaned training set into two groups: 
* training1 - 60% of the data for training the model
* training2- 40% of the data for testing and accuracy measurement

```{r}
set.seed(12345)
inTrain <- createDataPartition(y=training$classe, p=0.60, list=FALSE)
training1  <- training[inTrain, ]
training2  <- training[-inTrain, ]
dim(training1)
dim(training2)
```

# C. Developing the Model

## C.1 Selecting the most important variables for prediction

After cleaning the data, the total variables are reduced from 160 to 53 (52 of them as predictors, one of them as the outcome). This step is to select the most important variables for prediction. 

An initial model was built based on Random Forest algorithm using all the 52 prediction variables in the *training1* data set:

```{r}
set.seed(678173)
initialModel <- randomForest(classe~., data = training1, 
                 ntree = 100, importance = TRUE)
```

The variable importance plot (Figure 1) was then used to select the most important 10 variables for the final model: *roll_belt*, *yaw_belt*, *pitch_belt*, *magnet_dumbbell_z*, *magnet_dumbbell_y*, , *accel_dumbbell_y*, *gyros_arm_y*, *gyros_forearm_y*, *magnet_forearm_z*, and *pitch_forearm*.  

```{r}
varImpPlot (initialModel, n.var = 10,
            main = "Figure1. variable importance plot")
```

## C.2 Correlated predictors

Analyz the correlations among the selected 10 prediction variables: 

```{r}
predictVar<- c("roll_belt", "yaw_belt", "pitch_belt", "magnet_dumbbell_z", "magnet_dumbbell_y", "accel_dumbbell_y", "gyros_arm_y", "gyros_forearm_y", "magnet_forearm_z", "pitch_forearm")
predictor_corr <- round(cor(training1[, predictVar]), 2)
```

The Figure 2 below visualized the correlation matrix using hierarchical cluster order as the ordering method: 

```{r}
corrplot(predictor_corr, order = "hclust", tl.col="black", 
         diag = FALSE, tl.pos = "lt", mar=c(0,0,1,0),
         title = "Figure2. Visualization of the correlation matrix")
```

Figure 2 shows that *yaw_belt* and *roll_belt* have very high correlation, therefore, delete the *yaw_belt* variable.

## C.3 Fit the final model

The final model was built based on Random Forest algorithm using the 9 most important variables. A 2-fold cross-validation control was also used when building the final model.

```{r}
set.seed(76934)
finalModel <- train(classe~roll_belt+pitch_belt+
                      magnet_dumbbell_z+magnet_dumbbell_y+
                      accel_dumbbell_y+gyros_arm_y+
                      gyros_forearm_y+magnet_forearm_z+pitch_forearm, 
                     data=training1, method="rf",
                     trControl=trainControl(method="cv",number=2),
                     prox=TRUE, verbose=TRUE, allowParallel=TRUE)
```

## C.4 Final model testing using *training2* and out of sample error estimation

Apply the *finalModel* on the *training2* data set to estimate the expected accuracy:

```{r}
predictions <- predict(finalModel, newdata=training2)
confusionMat <- confusionMatrix(predictions, training2$classe)
confusionMat
```
**The accuracy for the final model on the testing dataset is *98.5%* which gives an estimated out of sample error of *1.5%*.**

# D. Course project submission

Apply the *finalModel* on the *testing* data set, and create 20 .TXT file for the submission:

```{r}
answers <- predict(finalModel, newdata=testing)
# function for creating a .txt file
pml_write_files = function(x)
  {
   n = length(x)
   for (i in 1:n)
     {
      filename = paste0("problem_id_", i, ".txt")
      write.table(x[i], file = filename, quote=FALSE, row.names=FALSE)
     }
  }
# run the pml_write_files
pml_write_files(answers)
```



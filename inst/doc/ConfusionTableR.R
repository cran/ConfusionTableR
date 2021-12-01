## ---- include = FALSE, echo=FALSE---------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.height= 5, 
  fig.width=7
)

## ----getcaretdata, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
library(caret)
library(dplyr)
library(mlbench)
library(tidyr)
library(e1071)
library(randomForest)

# Load in the iris data set for this problem 
data(iris)
df <- iris
# View the class distribution, as this is a multiclass problem, we can use the multi-uclassification data table builder
table(iris$Species)


## ----split_data, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
train_split_idx <- caret::createDataPartition(df$Species, p = 0.75, list = FALSE)
# Here we define a split index and we are now going to use a multiclass ML model to fit the data
train <- df[train_split_idx, ]
test <- df[-train_split_idx, ]
str(train)


## ----train_data, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
rf_model <- caret::train(Species ~ .,
                         data = df,
                         method = "rf",
                         metric = "Accuracy")

rf_model


## ----conf_mat, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
# Make a prediction on the fitted model with the test data
rf_class <- predict(rf_model, newdata = test, type = "raw") 
predictions <- cbind(data.frame(train_preds=rf_class, 
                                test$Species))
# Create a confusion matrix object
cm <- caret::confusionMatrix(predictions$train_preds, predictions$test.Species)
print(cm) 

## ----using_multi_function, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
# Implementing function to collapse data
library(ConfusionTableR)
mc_df <- ConfusionTableR::multi_class_cm(predictions$train_preds, predictions$test.Species,
                                         mode="everything")
# Access the reduced data for storage in databases
print(mc_df$record_level_cm)
glimpse(mc_df$record_level_cm)


## ----using_multi_function_cm1, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
mc_df$confusion_matrix

## ----using_multi_function_cm2, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
mc_df$cm_tbl

## ----load_cancer, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
# Load in the data
library(dplyr)
library(ConfusionTableR)
library(caret)
library(tidyr)
library(mlbench)

# Load in the data
data("BreastCancer", package = "mlbench")
breast <- BreastCancer[complete.cases(BreastCancer), ] #Create a copy
breast <- breast[, -1]
breast$Class <- factor(breast$Class) # Create as factor
for(i in 1:9) {
 breast[, i] <- as.numeric(as.character(breast[, i]))
}


## ----predict_cm, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
#Perform train / test split on the data
train_split_idx <- caret::createDataPartition(breast$Class, p = 0.75, list = FALSE)
train <- breast[train_split_idx, ]
test <- breast[-train_split_idx, ]
rf_fit <- caret::train(Class ~ ., data=train, method="rf")
#Make predictions to expose class labels
preds <- predict(rf_fit, newdata=test, type="raw")
predicted <- cbind(data.frame(class_preds=preds), test)


## ----binary_df, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----
bin_cm <- ConfusionTableR::binary_class_cm(predicted$class_preds, predicted$Class)
# Get the record level data
bin_cm$record_level_cm
glimpse(bin_cm$record_level_cm)

## ----visual_confusion_matrix, warning=FALSE, error=FALSE, message=FALSE, fig.height= 5, fig.width=7----

ConfusionTableR::binary_visualiseR(train_labels = predicted$class_preds,
                                   truth_labels= predicted$Class,
                                   class_label1 = "Not Stranded", 
                                   class_label2 = "Stranded",
                                   quadrant_col1 = "#28ACB4", 
                                   quadrant_col2 = "#4397D2", 
                                   custom_title = "Breast Cancer Confusion Matrix", 
                                   text_col= "black")





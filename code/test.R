rm(list = ls())
knitr::opts_chunk$set(echo = TRUE)
set.seed(88)
library(dplyr)
library(haven)
library(tidyr)
library(glmnet)
library(caret)
library(class)
library(xgboost)
library(pROC)
library(neuralnet)
library(plyr) 
library(DiagrammeR)

findNonNAColumns = function(data) {
  return(colnames(data)[apply(data, 2, function(col) all(!is.na(col)))])
}
# Import 2017 - 2018 blood pressure data
# Doc: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.htm
BPX_J = read_xpt("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT")
BPX_J = BPX_J %>%
  mutate(BPXSYAVG = rowMeans(select(., starts_with("BPXSY")), na.rm = TRUE),
         BPXDIAVG = rowMeans(select(., starts_with("BPXDI")), na.rm = TRUE)) %>%
  filter(complete.cases(BPXSYAVG, BPXDIAVG)) %>%
  mutate(BPXLEVEL = case_when(
    # Low blood pressure
    (BPXSYAVG < 90) | (BPXDIAVG < 60) ~ 1,
    # Normal blood pressure
    BPXSYAVG < 120 & BPXDIAVG < 80 ~ 2,
    # Elevated blood pressure
    (BPXSYAVG >= 120 & BPXSYAVG < 130) & (BPXDIAVG < 80) ~ 3,
    # High blood pressure
    (BPXSYAVG >= 130) | (BPXDIAVG >= 80) ~ 4
  ))
columns_no_na = findNonNAColumns(BPX_J)
BPX_J = BPX_J[c("BPXLEVEL", "SEQN", "BPACSZ", "BPXPLS", "BPXPTY", "BPXSYAVG", "BPXDIAVG")]
BPX_J = BPX_J %>%
  mutate(
    BPXLEVEL = as.factor(BPXLEVEL),
    BPACSZ = as.factor(BPACSZ),
    BPXPTY = as.factor(BPXPTY)
  )
BPXAVG = BPX_J[,c(6, 7)]
BPX_J = BPX_J[,-c(6, 7)]

# Import 2017 - 2018 demographic data
# Doc: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.htm
DEMO_J = read_xpt("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT")
currentVar = colnames(BPX_J)
FULLDATA = BPX_J %>% left_join(DEMO_J, by = "SEQN")
columns_no_na1 = setdiff(findNonNAColumns(FULLDATA), currentVar)
FULLDATA = FULLDATA[c(colnames(BPX_J), "RIAGENDR", "RIDAGEYR", "RIDRETH3", 
                      "DMDHHSIZ", "DMDHHSZA", "DMDHHSZB", "DMDHHSZE")]
FULLDATA = FULLDATA %>%
  mutate(
    RIAGENDR = as.factor(RIAGENDR),
    RIDRETH3 = as.factor(RIDRETH3)
  )

# Import 2017 - 2018 Total Nutrient Intakes, First Day
# Doc: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1TOT_J.htm
DR1TOT_J = read_xpt("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1TOT_J.XPT")
currentVar = colnames(FULLDATA)
FULLDATA = FULLDATA %>% left_join(DR1TOT_J, by = "SEQN")
FULLDATA = FULLDATA[, -c(13:42)]
FULLDATA = FULLDATA[, -c(79:149)]
FULLDATA = na.omit(FULLDATA)
columns_no_na2 = setdiff(findNonNAColumns(FULLDATA), currentVar)

# Prepare data for training and testing
FULLDATA = FULLDATA[,-2]
n.obs = nrow(FULLDATA)
set.seed(88)
index.train = sample(seq(n.obs), floor(n.obs * 0.8), replace = FALSE)
# Data frame train and test
train = FULLDATA[index.train, ]
train.X = model.matrix(~ . - 1, train[, -1])
test = FULLDATA[-index.train, ]
test.X = model.matrix(~ . - 1, test[, -1])
# Train and test with scale
train.X.mx = scale(train.X)
train.X.df = as.data.frame(train.X.mx)
BPXLEVEL.train = train$BPXLEVEL
train.scale = cbind(BPXLEVEL.train, train.X.df)
test.X.mx = scale(test.X)
test.X.df = as.data.frame(test.X.mx)
BPXLEVEL.test = test$BPXLEVEL
test.scale = cbind(BPXLEVEL.test, test.X.df)




# KNN 10 neighbors with 10 fold CV
# Use raw data
knn.model = train(
  BPXLEVEL ~ .,
  data = train,
  method = "knn",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(k = 15)
)
knn.predictions = predict(knn.model, test)
mean(knn.predictions == test$BPXLEVEL)
# ROC - AUC
multiclass.roc(as.numeric(test$BPXLEVEL) - 1, 
               as.numeric(knn.predictions) - 1)$auc



# Gradient boosting with 10 fold CV
# Use raw data
xgb.model = train(
  BPXLEVEL ~ .,
  data = train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(max_depth = 10, 
                         eta = 0.01, 
                         gamma = 0, 
                         colsample_bytree = 1, 
                         min_child_weight = 1, 
                         subsample = 1, 
                         nrounds = 200)
)
xgb.predictions = predict(xgb.model, test)
mean(xgb.predictions == test$BPXLEVEL)
# ROC - AUC
multiclass.roc(as.numeric(test$BPXLEVEL) - 1, 
               as.numeric(xgb.predictions) - 1)$auc


# Tune xgboost model
xgb.tune.grid = expand.grid(
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.2),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.8, 1),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.8, 1),
  nrounds = c(100, 200, 300)
)
xgb.tuned.model = train(
  BPXLEVEL ~ .,
  data = train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = xgb.tune.grid
)
xgb.tuned.predictions = predict(xgb.tuned.model, test)
mean(xgb.tuned.predictions == test$BPXLEVEL)
multiclass.roc(as.numeric(test$BPXLEVEL) - 1, 
               as.numeric(xgb.tuned.predictions) - 1)$auc



# Softmax classification with 10 fold CV
# Use raw data
cvfit = cv.glmnet(train.X, BPXLEVEL.train, family = "multinomial", type.multinomial = "grouped")
softmax.predictions = predict(cvfit, newx = test.X, 
                              s = cvfit$lambda.min, type = "class")
mean(softmax.predictions == BPXLEVEL.test)
# ROC - AUC
softmax.rocr = multiclass.roc(as.numeric(BPXLEVEL.test) - 1, 
                              as.numeric(softmax.predictions) - 1)
softmax.rocr$auc
coef(cvfit)

rm(list = ls())
set.seed(88)
library(dplyr)
library(haven)
library(tidyr)
library(glmnet)
library(caret)
library(class)
library(xgboost)
library(pROC)
library(e1071)

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
    # Normal blood pressure
    BPXSYAVG < 120 & BPXDIAVG < 80 ~ 0,
    # Elevated blood pressure
    (BPXSYAVG >= 120 & BPXSYAVG < 130) & (BPXDIAVG < 80) ~ 1,
    # High blood pressure
    (BPXSYAVG >= 130) | (BPXDIAVG >= 80) ~ 2
  ))
columns_no_na = findNonNAColumns(BPX_J)
BPX_J = BPX_J[c("BPXLEVEL", "SEQN", "BPACSZ", "BPXPLS", 
                "BPXPTY", "BPXSYAVG", "BPXDIAVG")]
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
FULLDATA = FULLDATA %>% left_join(DR1TOT_J, by = "SEQN")
FULLDATA = FULLDATA[, -c(13:42)]
FULLDATA = FULLDATA[, -c(79:149)]
FULLDATA = na.omit(FULLDATA)

# Import 2017 - 2018 Diabetes
# Doc: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.htm
currentVar = colnames(FULLDATA)
DIQ_J = read_xpt("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT")
FULLDATA = FULLDATA %>% left_join(DIQ_J, by = "SEQN")
FULLDATA = FULLDATA[c(currentVar, "DIQ010", "DIQ050")]
FULLDATA = FULLDATA %>%
  mutate(
    DIQ010 = as.factor(DIQ010),
    DIQ050 = as.factor(DIQ050)
  )

# Import 2017 - 2018 Health Insurance
# Doc: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HIQ_J.htm
currentVar = colnames(FULLDATA)
HIQ_J = read_xpt("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HIQ_J.XPT")
FULLDATA = FULLDATA %>% left_join(HIQ_J, by = "SEQN")
FULLDATA = FULLDATA[c(currentVar, "HIQ011")]
FULLDATA = FULLDATA %>%
  mutate(
    HIQ011 = as.factor(HIQ011)
  )

# Prepare data for training and testing
FULLDATA = FULLDATA[,-2]
n.obs = nrow(FULLDATA)
index.train = sample(seq(n.obs), floor(n.obs * 0.8), replace = FALSE)
# Data frame train and test
train = FULLDATA[index.train, ]
test = FULLDATA[-index.train, ]
train.X = train[, -1]
train.Y = train$BPXLEVEL
test.X = test[, -1]
test.Y = test$BPXLEVEL
# Model matrix train and test
train.X.mm = model.matrix(~ . - 1, train.X)
test.X.mm = model.matrix(~ . - 1, test.X)
# Data frame train and test with dummy
train.X.dummy = cbind(train.Y, as.data.frame(train.X.mm))
colnames(train.X.dummy)[1] = "BPXLEVEL"
test.X.dummy = cbind(test.Y, as.data.frame(test.X.mm))
colnames(test.X.dummy)[1] = "BPXLEVEL"

# KNN 10 neighbors with 10 fold CV
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
confusionMatrix(as.factor(knn.predictions), test.Y)

# Softmax classification with 10 fold CV
cvfit = cv.glmnet(train.X.mm, as.matrix(train.Y), family = "multinomial", 
                  type.multinomial = "grouped")
softmax.predictions = predict(cvfit, newx = test.X.mm, 
                              s = cvfit$lambda.min, type = "class")
mean(softmax.predictions == test.Y)
# ROC - AUC
softmax.rocr = multiclass.roc(as.numeric(test.Y) - 1, 
                              as.numeric(softmax.predictions) - 1)
softmax.rocr$auc
confusionMatrix(as.factor(softmax.predictions), test.Y)

set.seed(88)
# Gradient boosting with 10 fold CV
xgb.data = xgb.DMatrix(data = as.matrix(train.X.dummy[, -1]), 
                       label = recode(train.X.dummy$BPXLEVEL, 
                                      '0'=0, '1'=1, '2'=2))
xgb.test.X = data.matrix(test.X.dummy[, -1])
hyperparameters = list(
  eta = 0.0052,
  subsample = 0.75,
  col_subsample = 0.8,
  max_depth = 10
)
   
params = list(
  eta = hyperparameters$eta,
  subsample = hyperparameters$subsample,
  colsample_bytree = hyperparameters$col_subsample,  
  max_depth = hyperparameters$max_depth,
  tree_method = "exact",
  objective = "multi:softmax",
  num_class = 3
)
   
cv.xgb = xgb.cv(
  params = params,
  data = xgb.data,
  nfold = 10,
  metrics = "merror",
  verbose = 0,
  nrounds = 18
)
  
eval.log = as.data.frame(cv.xgb$evaluation_log)
min.mlogloss = min(eval.log[, 4])
min.mlogloss.index = which.min(eval.log[, 4])
xgb.model = xgboost(params = params, 
                    data = xgb.data, 
                    nrounds = min.mlogloss.index, 
                    verbose = 0)
xgb.predictions = predict(xgb.model, xgb.test.X)
mean(xgb.predictions == test.Y)
multiclass.roc(as.numeric(test.Y) - 1, 
               as.numeric(xgb.predictions) - 1)$auc
confusionMatrix(as.factor(xgb.predictions), test.Y)
importance.matrix = xgb.importance(colnames(train.X.dummy[, -1]), model = xgb.model)
importance.matrix
xgb.plot.importance(importance.matrix)

threshold = sort(importance.matrix$Importance)
for (i in 1:length(selected.features)) {
  selected.features = importance.matrix$Feature[which(importance.matrix$Importance > threshold[i])]
  set.seed(88)
  # Gradient boosting with 10 fold CV
  xgb.data = xgb.DMatrix(data = as.matrix(train.X.dummy[selected.features]), 
                         label = recode(as.matrix(train.Y), '0'=0, '1'=1, '2'=2))
  xgb.test.X = data.matrix(test.X.dummy[selected.features])
  hyperparameters = list(
    eta = 0.0052,
    subsample = 0.75,
    col_subsample = 0.8,
    max_depth = 10
  )
     
  params = list(
    eta = hyperparameters$eta,
    subsample = hyperparameters$subsample,
    colsample_bytree = hyperparameters$col_subsample,  
    max_depth = hyperparameters$max_depth,
    tree_method = "exact",
    objective = "multi:softmax",
    num_class = 3
  )
     
  cv.xgb = xgb.cv(
    params = params,
    data = xgb.data,
    nfold = 10,
    metrics = "merror",
    verbose = 0,
    nrounds = 18
  )
    
  eval.log = as.data.frame(cv.xgb$evaluation_log)
  min.mlogloss = min(eval.log[, 4])
  min.mlogloss.index = which.min(eval.log[, 4])
  xgb.model = xgboost(params = params, 
                      data = xgb.data, 
                      nrounds = min.mlogloss.index, 
                      verbose = 0)
  xgb.predictions = predict(xgb.model, xgb.test.X)
  accuracy = mean(xgb.predictions == test.Y)
  auc = multiclass.roc(as.numeric(test.Y) - 1, 
                      as.numeric(xgb.predictions) - 1)$auc
  cat("Threshold:", threshold[i], "Accuracy:", accuracy, "AUC:", auc, "\n")
}

# SVM
svm.model = svm(BPXLEVEL ~ ., data = train.X.dummy, kernel = "radial")
svm.predictions = predict(svm.model, test.X.dummy)
mean(svm.predictions == test.Y)
multiclass.roc(as.numeric(test.Y) - 1, 
               as.numeric(svm.predictions) - 1)$auc
confusionMatrix(as.factor(svm.predictions), test.Y)

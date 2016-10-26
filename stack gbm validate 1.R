rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(xgboost)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

#train.binary <- read_csv("data/train_recode.csv.gz")
#test.binary <- read_csv("data/test_recode.csv.gz")

train.factor <- read_csv("data/train_recode_factor.csv.gz")
test.factor <- read_csv("data/test_recode_factor.csv.gz")

set.seed(666)
train_python <- train.factor %>%
  sample_frac(.8)

test_python <- train.factor %>%
  anti_join(train_python %>%
              select(id))

train_python_a <- train_python %>%
  sample_frac(.5)
train_python_b <- train_python %>%
  anti_join(train_python_a %>%
              select(id))

train.file.a <- "data/train_python_a.csv.gz"
train.file.b <- "data/train_python_b.csv.gz"
test.file <- "data/test_python.csv.gz"

#write.csv(train_python_a, gzfile(train.file.a), row.names=FALSE)
#write.csv(train_python_b, gzfile(train.file.b), row.names=FALSE)
#write.csv(test_python, gzfile(test.file), row.names=FALSE)

nn_test_50 <- read_csv("python/nn_preds_test_50epochs.csv")
nn_test_25 <- read_csv("python/nn_preds_test_25epochs.csv")

glimpse(test_python)
glimpse(train_python)

pred <- nn_test_25 %>%
  inner_join(test_python %>%
               select(id, loss)) %>%
  mutate_at(vars(matches("nn_")), funs(exp))

pred %>%
  mutate_at(vars(matches("nn_")), funs(abs(.-loss))) %>%
  summarise_at(vars(matches("nn_")), mean)

pred_50 <- nn_test_50 %>%
  inner_join(test_python %>%
               select(id, loss)) %>%
  mutate_at(vars(matches("nn_")), funs(exp))

pred_50 %>%
  mutate_at(vars(matches("nn_")), funs(abs(.-loss))) %>%
  summarise_at(vars(matches("nn_")), mean)

################################################################################
#baseline single GBM

#setdiff(train_python_factor$id, train_python$id)
#setdiff(test_python$id, test_python_factor$id)

model.data <- train_python

model.y <- model.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)  

model.data <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)

set.seed(666)
param.1 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.01,
                "subsample" = .9,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 9,
                "silent" = 0,
                "nthread" = 12)

bst.1 = xgb.train(params = param.1, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(777)
param.2 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.01,
                "subsample" = 1,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 8,
                "silent" = 0,
                "nthread" = 12)

bst.2 = xgb.train(params = param.2, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))

oos.test <- test_python %>%
  select(-id, -loss) %>%
  as.matrix()

##1121.249
data_frame(oos.loss = test_python$loss,
           loss.pred = predict(bst.1, oos.test)) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

##1122.68
data_frame(oos.loss = test_python$loss,
           loss.pred = predict(bst.2, oos.test)) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

###########################################################
set.seed(888)
param.3 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.005,
                "subsample" = .9,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 9,
                "silent" = 0,
                "nthread" = 12)

bst.3 = xgb.train(params = param.3, 
                  data = xgbtrain, 
                  nrounds = 4000,
                  verbose = 1, watchlist = list(train=xgbtrain))

oos.test <- test_python %>%
  select(-id, -loss) %>%
  as.matrix()

##1120.567
data_frame(oos.loss = test_python$loss,
           loss.pred = predict(bst.3, oos.test)) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))


##1121.065
data_frame(oos.loss = test_python$loss,
           gbm.1 = predict(bst.1, oos.test),
           gbm.2 = predict(bst.2, oos.test),
           gbm.3 = predict(bst.3, oos.test)) %>%
  mutate(loss.pred = (gbm.1 + gbm.2)/2) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

##1120.349
data_frame(oos.loss = test_python$loss,
           gbm.1 = predict(bst.1, oos.test),
           gbm.2 = predict(bst.2, oos.test),
           gbm.3 = predict(bst.3, oos.test)) %>%
  mutate(loss.pred = (gbm.1 + gbm.2)/2) %>%
  mutate(loss.pred = (loss.pred + gbm.3)/2) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

################################################################################
#Stack
nn_train <- read_csv("python/nn_preds_train_5epochs.csv")
nn_test <- read_csv("python/nn_preds_test_5epochs.csv")

nn_test <- read_csv("python/submission_keras.csv")
nn_test <- read_csv("python/nn_preds_quick.csv")

nn_test1 <- read_csv("python/nn_preds_test2_250epochs.csv")
nn_test2 <- read_csv("python/nn_preds_test_250epochs.csv")

trees_a <- read_csv("python/trees_preds_a.csv")
trees_b <- read_csv("python/trees_preds_b.csv")
trees_test <- read_csv("python/trees_preds_test.csv")

nn_test %>%
  inner_join(train.factor %>%
               select(id, loss_true=loss)) %>%
  summarise(mae = mean(abs(loss_true-loss)))

nn_test %>%
  inner_join(train.factor %>%
               select(id, loss_true=loss)) %>%
  summarise(mae = mean(abs(loss_true-exp(nn_pred_1))))

nn_test1 %>%
  inner_join(train.factor %>%
               select(id, loss)) %>%
  mutate_at(vars(matches("nn_")), funs(exp)) %>%
  mutate_at(vars(matches("nn_")), funs(abs(.-loss))) %>%
  summarise_at(vars(matches("nn_")), mean)
nn_test2 %>%
  inner_join(train.factor %>%
               select(id, loss)) %>%
  mutate_at(vars(matches("nn_")), funs(exp)) %>%
  mutate_at(vars(matches("nn_")), funs(abs(.-loss))) %>%
  summarise_at(vars(matches("nn_")), mean)


trees_test_summary <- trees_test %>%
  inner_join(train.factor %>%
               select(id, loss)) %>%
  mutate_at(vars(matches("pred")), funs(exp)) %>%
  mutate_at(vars(matches("pred")), funs(abs(.-loss))) %>%
  summarise_at(vars(matches("pred")), mean)

trees_test_summary %>%
  gather(model, mae) %>%
  arrange(mae)

#####
#stack model
#####

model.y <- train_python %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)  

model.data <- train_python %>%
  inner_join(bind_rows(trees_a, trees_a)) %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)

set.seed(666)
bst.1.stack = xgb.train(params = param.1, 
                        data = xgbtrain, 
                        nrounds = 2000,
                        verbose = 1, watchlist = list(train=xgbtrain))

set.seed(777)
bst.2.stack = xgb.train(params = param.2, 
                        data = xgbtrain, 
                        nrounds = 2000,
                        verbose = 1, watchlist = list(train=xgbtrain))

oos.test <- test_python %>%
  inner_join(trees_test) %>%
  select(-id, -loss) %>%
  as.matrix()

##1131.224
data_frame(oos.loss = test_python$loss,
           loss.pred = predict(bst.1.stack, oos.test)) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

##1133.165
data_frame(oos.loss = test_python$loss,
           loss.pred = predict(bst.2.stack, oos.test)) %>%
  mutate(abs.error = abs(oos.loss-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

################################################################################
#Blend on single models
nn_test <- read_csv("python/submission_keras.csv")
df <- data_frame(id = test_python$id,
                 oos.loss = test_python$loss,
                 gbm.1 = predict(bst.1, oos.test),
                 gbm.2 = predict(bst.2, oos.test),
                 gbm.3 = predict(bst.3, oos.test)) %>%
  inner_join(nn_test) %>%
  mutate(gbm.blend = (gbm.1 + gbm.2)/2) %>%
  mutate(gbm.loss = exp(gbm.blend),
         loss.pred = (loss+gbm.loss)/2) %>%
  mutate(abs.error = abs(oos.loss-loss.pred))

##1112.2
df %>%
  summarise(mae = mean(abs.error))
#########################################
df <- data_frame(id = test_python$id,
                 oos.loss = test_python$loss,
                 gbm.1 = predict(bst.1, oos.test),
                 gbm.2 = predict(bst.2, oos.test),
                 gbm.3 = predict(bst.3, oos.test)) %>%
  inner_join(nn_test) %>%
  mutate(gbm.blend = (gbm.1 + gbm.2)/2) %>%
  mutate(gbm.loss = exp(gbm.3),
         loss.pred = (loss+gbm.loss)/2) %>%
  mutate(abs.error = abs(oos.loss-loss.pred))

##1111.522
df %>%
  summarise(mae = mean(abs.error))
#########################################
df <- data_frame(id = test_python$id,
                 oos.loss = test_python$loss,
                 gbm.1 = predict(bst.1, oos.test),
                 gbm.2 = predict(bst.2, oos.test),
                 gbm.3 = predict(bst.3, oos.test)) %>%
  inner_join(nn_test) %>%
  mutate(gbm.blend = (gbm.1 + gbm.2)/2) %>%
  mutate(gbm.blend = (gbm.blend + gbm.3)/2) %>%
  mutate(gbm.loss = exp(gbm.blend),
         loss.pred = (loss+gbm.loss)/2) %>%
  mutate(abs.error = abs(oos.loss-loss.pred))

##1111.522
df %>%
  summarise(mae = mean(abs.error))



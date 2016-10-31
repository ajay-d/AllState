rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(ggplot2)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

train.factor <- read_csv("data/train_recode_factor.csv.gz")
test.factor <- read_csv("data/test_recode_factor.csv.gz")

nn.full <- read_csv("python/keras_full.csv")
nn.full.log <- read_csv("python/keras_full_log.csv")
################################################################################
param.1 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.01,
                "subsample" = .9,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 9,
                "silent" = 0,
                "nthread" = 12)

param.2 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.01,
                "subsample" = 1,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 8,
                "silent" = 0,
                "nthread" = 12)

param.3 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.01,
                "gamma" = 1,
                "alpha" = 1,
                "subsample" = .8,
                "colsample_bytree" = .5,
                #"colsample_bylevel" = .9,
                "max_depth" = 12,
                "min_child_weight" = 1,
                "silent" = 0,
                "nthread" = 12)

param.4 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.005,
                "subsample" = .9,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 9,
                #"gamma" = 1,
                #"alpha" = 1,
                "silent" = 0,
                "nthread" = 12)

#####Log Loss#####
model.data <- train.factor
model.y <- model.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)  

model.data <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)

set.seed(666)
bst.1 = xgb.train(params = param.1, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))
set.seed(777)
bst.2 = xgb.train(params = param.2, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))
set.seed(888)
bst.3 = xgb.train(params = param.3, 
                  data = xgbtrain, 
                  nrounds = 1788,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(999)
bst.4 = xgb.train(params = param.4, 
                  data = xgbtrain, 
                  nrounds = 4000,
                  verbose = 1, watchlist = list(train=xgbtrain))

################################################################################
#####log(loss + shift)#####
model.data <- train.factor
model.y <- model.data %>%
  mutate(loss = log(loss+200)) %>%
  select(loss) %>%
  use_series(loss)  

model.data <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
set.seed(666)
bst.1s = xgb.train(params = param.1, 
                   data = xgbtrain, 
                   nrounds = 2000,
                   verbose = 1, watchlist = list(train=xgbtrain))

set.seed(777)
bst.2s = xgb.train(params = param.2, 
                   data = xgbtrain, 
                   nrounds = 2000,
                   verbose = 1, watchlist = list(train=xgbtrain))

set.seed(888)
bst.3s = xgb.train(params = param.3, 
                   data = xgbtrain, 
                   nrounds = 1800,
                   verbose = 1, watchlist = list(train=xgbtrain))

set.seed(999)
bst.4s = xgb.train(params = param.4, 
                   data = xgbtrain, 
                   nrounds = 4000,
                   verbose = 1, watchlist = list(train=xgbtrain))

################################################################################


################################################################################

submission  <- data_frame(id = test.factor$id,
                          gbm.1 = predict(bst.1, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.2 = predict(bst.2, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.3 = predict(bst.3, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.4 = predict(bst.4, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.1s = predict(bst.1s, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.2s = predict(bst.2s, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.3s = predict(bst.3s, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.4s = predict(bst.4s, test.factor %>% select(-id) %>% as.matrix())) %>%
  inner_join(nn.full %>%
               rename(nn.1 = loss)) %>%
  inner_join(nn.full.log %>%
               rename(nn.2 = loss))

#Loss 3 Only
#1117.36
df <- submission %>%
  mutate(loss.pred =  gbm.3) %>%
  mutate(loss =  exp(loss.pred)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#Loss 3 Shift Only
#1115.91
df <- submission %>%
  mutate(loss.pred =  gbm.3s) %>%
  mutate(loss =  exp(loss.pred) - 200) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#NN + loss 3 shift
#1110.327
df <- submission %>%
  mutate(loss.pred =  exp(gbm.3s) - 200) %>%
  mutate(loss =  (loss.pred+nn.1)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#NN + loss 3 shift + average 3 GBMs
#1110.23
df <- submission %>%
  mutate(loss.pred.1 =  exp(gbm.3s) - 200) %>%
  mutate(loss.pred.2 =  (gbm.1 + gbm.1 + gbm.3 + gbm.4)/4) %>%
  mutate(loss.pred.2 =  exp(loss.pred.2)) %>%
  mutate(loss =  (loss.pred.1+loss.pred.2+nn.1)/3) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

gg <- submission %>%
  mutate(loss.pred.1 =  exp(gbm.3s) - 200) %>%
  mutate(loss.pred.2 =  (gbm.1 + gbm.1 + gbm.3 + gbm.4)/4) %>%
  mutate(loss.pred.2 =  exp(loss.pred.2)) %>%
  select(id, loss.pred.1, loss.pred.2, nn.1) %>%
  gather(variable, value, -id)

gg %>%
  group_by(variable) %>%
  summarise(mean = mean(value),
            median = median(value))

ggplot(gg, aes(x=value, color=variable)) +
  geom_density() +
  xlim(0,10000)

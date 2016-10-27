rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)


train.binary <- read_csv("data/train_recode.csv.gz")
test.binary <- read_csv("data/test_recode.csv.gz")

train.factor <- read_csv("data/train_recode_factor.csv.gz")
test.factor <- read_csv("data/test_recode_factor.csv.gz")

################################################################################

model.data <- train.binary
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
                "subsample" = .8,
                "colsample_bytree" = .8,
                "max_depth" = 10,
                "silent" = 0,
                "nthread" = 12)

bst.1 = xgb.train(params = param.1, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(777)
param.2 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "eta" = 0.03,
                "subsample" = 1,
                "colsample_bytree" = .5,
                "colsample_bylevel" = .9,
                "max_depth" = 8,
                "silent" = 0,
                "nthread" = 12)

bst.2 = xgb.train(params = param.2, 
                  data = xgbtrain, 
                  nrounds = 1000,
                  verbose = 1, watchlist = list(train=xgbtrain))

################################################################################

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
bst.3 = xgb.train(params = param.1, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(777)
bst.4 = xgb.train(params = param.2, 
                  data = xgbtrain, 
                  nrounds = 1000,
                  verbose = 1, watchlist = list(train=xgbtrain))

################################################################################

model.data <- train.factor
model.y <- model.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)  

model.data <- model.data %>%
  select(-starts_with('new')) %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)

set.seed(666)
bst.5 = xgb.train(params = param.1, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(777)
bst.6 = xgb.train(params = param.2, 
                  data = xgbtrain, 
                  nrounds = 1000,
                  verbose = 1, watchlist = list(train=xgbtrain))

################################################################################

submission  <- data_frame(id = test.binary$id,
                          loss.1 = predict(bst.1, test.binary %>% select(-id) %>% as.matrix()),
                          loss.2 = predict(bst.2, test.binary %>% select(-id) %>% as.matrix()),
                          loss.3 = predict(bst.3, test.factor %>% select(-id) %>% as.matrix()),
                          loss.4 = predict(bst.4, test.factor %>% select(-id) %>% as.matrix()),
                          loss.5 = predict(bst.5, test.factor %>% select(-id) %>% select(-starts_with('new')) %>% as.matrix()),
                          loss.6 = predict(bst.6, test.factor %>% select(-id) %>% select(-starts_with('new')) %>% as.matrix()))

cor(submission$loss.1, submission$loss.2)
cor(submission$loss.1, submission$loss.3)
cor(submission$loss.3, submission$loss.4)
cor(submission$loss.1, submission$loss.5)

#Loss 1 only
#full data actual: 1122.78
df <- submission %>%
  mutate(loss = exp(loss.1)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#Loss 2 only
#full data actual: 1121.95
df <- submission %>%
  mutate(loss = exp(loss.2)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#Loss 3 only
#full data actual: 1118.45
df <- submission %>%
  mutate(loss = exp(loss.3)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#Loss 4 only
#full data actual: 1119.65
df <- submission %>%
  mutate(loss = exp(loss.4)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#Loss 5 only
#full data actual: 1118.99
df <- submission %>%
  mutate(loss = exp(loss.5)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)


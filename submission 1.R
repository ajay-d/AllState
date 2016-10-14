rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)


train.binary <- read_csv("train_recode.csv.gz")
test.binary <- read_csv("test_recode.csv.gz")
train <- read_csv("data/train.csv.zip")

################################################################################

set.seed(666)

model.data <- train.binary %>%
  sample_frac(.8)

#final out of sample test set
oos.data <- train.binary %>%
  anti_join(model.data %>%
              select(id))

watch.data <- model.data %>%
  sample_frac(.2)

#remove watch data from training set
model.data <- model.data %>%
  anti_join(watch.data %>%
              select(id))

nrow(model.data) + nrow(oos.data) + nrow(watch.data)
nrow(train)

model.y <- model.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)

watch.y <- watch.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)

oos.y <- oos.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)

model.data <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()

watch.data <- watch.data %>%
  select(-id, -loss) %>%
  as.matrix()

oos.data <- oos.data %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
xgbtest <- xgb.DMatrix(data=watch.data, label=watch.y)
watchlist <- list(train=xgbtrain, 
                  test=xgbtest)

param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae",
              "eta" = 0.05,
              "subsample" = .9,
              "max_depth" = 8,
              "silent" = 0,
              "nthread" = 12)

bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 200, 
                watchlist = watchlist)

#OOS mae / 1139
data_frame(oos.loss = oos.y,
           loss.pred = predict(bst, oos.data)) %>%
  mutate(abs.error = abs(exp(oos.loss)-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))

#actual:1141
submission  <- data_frame(id = test.binary$id,
                          loss = predict(bst, test.binary %>% select(-id) %>% as.matrix())) %>%
  mutate(loss = exp(loss))

summary(train$loss)
summary(submission$loss)

write.csv(submission, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

set.seed(666)

model.data <- train.binary
model.y <- model.data %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)  

model.data <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae",
              "eta" = 0.05,
              "subsample" = .9,
              "max_depth" = 8,
              "silent" = 0,
              "nthread" = 12)

bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 200,
                verbose = 1, watchlist = list(train=xgbtrain))

#full data actual:1134
submission  <- data_frame(id = test.binary$id,
                          loss = predict(bst, test.binary %>% select(-id) %>% as.matrix())) %>%
  mutate(loss = exp(loss))

summary(train$loss)
summary(submission$loss)

write.csv(submission, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

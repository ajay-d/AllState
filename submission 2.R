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
                "subsample" = .9,
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
                "eta" = 0.02,
                "subsample" = .9,
                "max_depth" = 8,
                "silent" = 0,
                "nthread" = 12)

bst.2 = xgb.train(params = param.2, 
                  data = xgbtrain, 
                  nrounds = 1000,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(888)
param.3 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "booster" = "dart",
                "sample_type" = "weighted", #or "uniform"
                "normalize_type" = "forest", #or "forest
                "eta" = 0.01,
                "subsample" = .9,
                "max_depth" = 9,
                "silent" = 0,
                "nthread" = 12)

bst.3 = xgb.train(params = param.3, 
                  data = xgbtrain, 
                  nrounds = 2000,
                  verbose = 1, watchlist = list(train=xgbtrain))

set.seed(999)
param.4 <- list("objective" = "reg:linear",
                "eval_metric" = "mae",
                "booster" = "dart",
                "sample_type" = "weighted", #or "uniform"
                "normalize_type" = "forest", #or "forest
                "eta" = 0.02,
                "subsample" = .9,
                "max_depth" = 8,
                "silent" = 0,
                "nthread" = 12)

bst.4 = xgb.train(params = param.4, 
                  data = xgbtrain, 
                  nrounds = 1000,
                  verbose = 1, watchlist = list(train=xgbtrain))


submission  <- data_frame(id = test.binary$id,
                          loss.1 = predict(bst.1, test.binary %>% select(-id) %>% as.matrix()),
                          loss.2 = predict(bst.2, test.binary %>% select(-id) %>% as.matrix()),
                          loss.3 = predict(bst.3, test.binary %>% select(-id) %>% as.matrix()),
                          loss.4 = predict(bst.4, test.binary %>% select(-id) %>% as.matrix()))

#Loss 1 only
#full data actual:1121
submission <- submission %>%
  mutate(loss = exp(loss.1)) %>%
  select(id, loss) 

write.csv(submission, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
#Geometric Average
#full data actual:1121
submission <- submission %>%
  mutate(loss = (loss.1*loss.2*loss.3*loss.4)^.25) %>%
  mutate(loss = exp(loss)) %>%
  select(id, loss) 

write.csv(submission, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
#Arithmetic Average
#full data actual:1121
submission <- submission %>%
  mutate(loss = (loss.1+loss.2+loss.3+loss.4)/4) %>%
  mutate(loss = exp(loss)) %>%
  select(id, loss) 

write.csv(submission, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
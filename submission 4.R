rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)


train.factor <- read_csv("data/train_recode_factor.csv.gz")
test.factor <- read_csv("data/test_recode_factor.csv.gz")

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

################################################################################


################################################################################

submission  <- data_frame(id = test.factor$id,
                          gbm.1 = predict(bst.1, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.2 = predict(bst.2, test.factor %>% select(-id) %>% as.matrix()),
                          gbm.3 = predict(bst.3, test.factor %>% select(-id) %>% as.matrix()))

#Loss 1 + 2 blend
#full data actual: 1117.36
df <- submission %>%
  mutate(loss.pred = (gbm.1 + gbm.2)/2) %>%
  mutate(loss.pred = (loss.pred + gbm.3)/2) %>%
  mutate(loss =  exp(loss.pred)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)



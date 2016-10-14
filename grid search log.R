rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)


train.binary <- read_csv("train_recode.csv.gz")
test.binary <- read_csv("test_recode.csv.gz")
train <- read_csv("data/train.csv.zip")

################################################################################
#####Grid Search Log#####

set.seed(666)

model.data <- train.binary %>%
  sample_frac(.8)

#final out of sample test set
oos.data <- train.binary %>%
  anti_join(model.data %>%
              select(id))

model.data <- model.data %>%
  anti_join(oos.data %>%
              select(id))

nrow(model.data) + nrow(oos.data)
nrow(train)

model.y <- model.data %>%
  inner_join(train %>%
               select(id, loss)) %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)

oos.y <- oos.data %>%
  inner_join(train %>%
               select(id, loss)) %>%
  select(loss) %>%
  mutate(loss = log(loss)) %>%
  use_series(loss)

model.data <- model.data %>%
  select(-id) %>%
  as.matrix()

oos.data <- oos.data %>%
  select(-id) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
xgb.grid.log <- NULL

for(s1 in c(.5, .8, .9, 1))
  #for(s2 in c(.5, .8, .9, 1))
    #for(s3 in c(.5, .8, .9, 1))
      for(eta in c(.01, .02, .03, .05))
        for(max_depth in c(8, 9, 10, 11, 12, 15))
          for(nrounds in c(250, 500, 750, 1000, 2000, 3000)) {
      
      set.seed(666)
      
            s2 <- 1
            s3 <- 1
      param <- list("objective" = "reg:linear",
                    "eval_metric" = "rmse",
                    "eval_metric" = "mae",
                    "eta" = eta,
                    "subsample" = s1,
                    "colsample_bytree" = s2,
                    "colsample_bylevel" = s3,
                    "max_depth" = max_depth,
                    "silent" = 0,
                    "nthread" = 12)

      bst = xgb.train(params = param, 
                      data = xgbtrain, 
                      nrounds = nrounds,
                      verbose = 0)
      
      pred <- data_frame(oos.loss = oos.y,
                         loss.pred = predict(bst, oos.data)) %>%
        mutate(abs.error = abs(exp(oos.loss)-exp(loss.pred))) %>%
        summarise(mae = mean(abs.error))

      
      df <- data_frame(eta = eta,
                       max_depth = max_depth,
                       nrounds = nrounds,
                       subsample = s1,
                       colsample_bytree = s2,
                       colsample_bylevel = s3,
                       mae = pred$mae)
      
      xgb.grid.log <- bind_rows(xgb.grid.log, df)
      
      print(nrow(xgb.grid.log))
    }

write.csv(xgb.grid.log, file='GridLogLossBinary.csv', row.names = FALSE)




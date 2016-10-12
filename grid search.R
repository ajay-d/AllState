rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample.sub <- read_csv("data/sample_submission.csv.zip")
train <- read_csv("data/train.csv.zip")
test <- read_csv("data/test.csv.zip")


#2 cat variables
all.cat.vars <- train %>%
  select(starts_with('cat'))

cat.table <- NULL
for(i in names(all.cat.vars)) {
  
  #get cardinality
  var.card <- all.cat.vars %>%
    count_(i, sort=TRUE)
  
  df <- data_frame(var = i,
                   categories = nrow(var.card))
  
  cat.table <- bind_rows(cat.table, df)
}

cat.2.vars <- cat.table %>%
  filter(categories == 2) %>%
  use_series(var)

train.recode <- train %>%
  mutate_at(vars(one_of(cat.2.vars)), funs(c("A" = 0, "B" = 1)[.]))

################################################################################
#####GBM cont+cat 2 vars - Grid Search#####

model.vars <- names(train.recode) %>%
  str_subset('cont') %>%
  union(cat.2.vars)

set.seed(666)

model.data <- train.recode %>%
  sample_frac(.8) %>%
  select(id, one_of(model.vars))

#final out of sample test set
oos.data <- train.recode %>%
  anti_join(model.data %>%
              select(id)) %>%
  select(id, one_of(model.vars))

model.data <- model.data %>%
  anti_join(oos.data %>%
              select(id))

nrow(model.data) + nrow(oos.data)
nrow(train)

model.y <- model.data %>%
  inner_join(train.recode %>%
               select(id, loss)) %>%
  select(loss) %>%
  use_series(loss)

oos.y <- oos.data %>%
  inner_join(train.recode %>%
               select(id, loss)) %>%
  select(loss) %>%
  use_series(loss)

model.data <- model.data %>%
  select(-id) %>%
  as.matrix()

oos.data <- oos.data %>%
  select(-id) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
xgb.grid <- NULL

for(eta in c(.01, .025, .05))
  for(max_depth in c(8, 9, 10, 11, 12))
    for(nrounds in c(250, 500, 750, 1000)) {
      set.seed(666)
      
      param <- list("objective" = "reg:linear",
                    "eval_metric" = "rmse",
                    "eval_metric" = "mae",
                    "eta" = eta,
                    "subsample" = .9,
                    "max_depth" = max_depth,
                    "silent" = 0,
                    "nthread" = 12)

      bst = xgb.train(params = param, 
                      data = xgbtrain, 
                      nrounds = nrounds,
                      verbose = 0)
      
      pred <- data_frame(oos.loss = oos.y,
                         loss.pred = predict(bst, oos.data)) %>%
        mutate(abs.error = abs(oos.loss-loss.pred)) %>%
        summarise(mae = mean(abs.error))

      
      df <- data_frame(eta = eta,
                       max_depth = max_depth,
                       nrounds = nrounds,
                       mae = pred$mae)
      
      xgb.grid <- bind_rows(xgb.grid, df)
      
      print(nrow(xgb.grid))
    }

#.05, 8/9/10, 250/300
#mae=1325
write.csv(xgb.grid, file='GridLoss.csv', row.names = FALSE)

################################################################################
#####GBM cont+cat 2 vars - Grid Search Log#####

model.vars <- names(train.recode) %>%
  str_subset('cont') %>%
  union(cat.2.vars)

set.seed(666)

model.data <- train.recode %>%
  sample_frac(.8) %>%
  select(id, one_of(model.vars))

#final out of sample test set
oos.data <- train.recode %>%
  anti_join(model.data %>%
              select(id)) %>%
  select(id, one_of(model.vars))

model.data <- model.data %>%
  anti_join(oos.data %>%
              select(id))

nrow(model.data) + nrow(oos.data)
nrow(train)

model.y <- model.data %>%
  inner_join(train.recode %>%
               select(id, loss)) %>%
  mutate(loss = log(loss)) %>%
  select(loss) %>%
  use_series(loss)

oos.y <- oos.data %>%
  inner_join(train.recode %>%
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

for(eta in c(.01, .025, .05))
  for(max_depth in c(8, 9, 10, 11, 12))
    for(nrounds in c(250, 500, 750, 1000)) {
      set.seed(666)
      
      param <- list("objective" = "reg:linear",
                    "eval_metric" = "rmse",
                    "eval_metric" = "mae",
                    "eta" = eta,
                    "subsample" = .9,
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
                       mae = pred$mae)
      
      xgb.grid.log <- bind_rows(xgb.grid.log, df)
      
      print(nrow(xgb.grid.log))
    }

#mae=1265
write.csv(xgb.grid.log, file='GridLogLoss.csv', row.names = FALSE)




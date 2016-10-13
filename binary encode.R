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
                   categories = nrow(var.card)) %>%
    mutate(binary = R.utils::intToBin(categories),
           binary.length = nchar(binary))
  
  cat.table <- bind_rows(cat.table, df)
}

cat.2.vars <- cat.table %>%
  filter(categories == 2) %>%
  use_series(var)

train.recode <- train %>%
  mutate_at(vars(one_of(cat.2.vars)), funs(c("A" = 0, "B" = 1)[.]))

cat.2plus.vars<- cat.table %>%
  filter(categories == 20) %>%
  use_series(var)

cat.table.long <- NULL
for(i in cat.2plus.vars) {
  
  train.var <- train %>%
    group_by_(i) %>%
    summarise(sd = sd(loss),
              mean = mean(loss),
              n.train = n()) %>%
    mutate(pct.train = n.train/sum(n.train),
           cv = mean/sd)
  
  test.var <- test %>%
    group_by_(i) %>%
    summarise(n.test = n()) %>%
    mutate(pct.test = n.test/sum(n.test))
  
  var.table <- inner_join(train.var, test.var) %>%
    rename_(value=i) %>%
    mutate(var = i)
  
  cat.table.long <- bind_rows(cat.table.long, var.table)
}

cat.table.long <- cat.table.long %>%
  group_by(var) %>%
  arrange(var, desc(pct.train)) %>%
  mutate(test.flag = ifelse(abs(pct.train - pct.test) > .001, 1, 0),
         #start at 0
         cat.number = row_number()-1,
         cat.number = ifelse(pct.train < .001, NA, cat.number)) %>%
  fill(cat.number)
  
for(i in cat.2plus.vars) {
  
  n.new.vars <- cat.table.long %>%
    filter(var==i) %>%
    summarise(cat.number = max(cat.number)) %>%
    use_series(cat.number)
  
  len.binary <- nchar(R.utils::intToBin(n.new.vars))
  
  col.names <- paste(i, 1:len.binary, sep='_')
  col.values <- R.utils::intToBin(n.new.vars) %>% str_split('')
  col.values[[1]]
  
  train.recode.long <- train.recode %>%
    select_('id', i) %>%
    rename_(value=i) %>%
    inner_join(cat.table.long %>%
                 ungroup %>%
                 filter(var==i) %>%
                 select(value, cat.number)) %>%
    mutate(binary = R.utils::intToBin(cat.number)) %>%
    mutate(binary2 = binary %>% str_split_fixed('', n=4) %>% as.character() %>% paste(collapse = "_"))
    rowwise() %>%
    mutate(binary = R.utils::intToBin(cat.number) %>% str_split('') %>% extract2(1) %>% paste(collapse = "_"))
    
    #rowwise() %>%
    mutate(binary = binary %>% str_split('') %>% extract2(1) %>% paste(collapse = "_")) %>%
    separate(binary, into=col.names, sep='_') %>%
    select(-cat.number, -value) %>%
    mutate_if(is.character, as.numeric)
  
}
  
################################################################################
#####GBM cont only#####

model.vars <- names(train) %>%
  str_subset('cont')

set.seed(666)

model.data <- train %>%
  sample_frac(.8) %>%
  select(id, one_of(model.vars))

watch.data <- model.data %>%
  sample_frac(.2) %>%
  select(id, one_of(model.vars))

model.y <- model.data %>%
  inner_join(train %>%
               select(id, loss)) %>%
  select(loss) %>%
  use_series(loss)

watch.y <- watch.data %>%
  inner_join(train %>%
               select(id, loss)) %>%
  select(loss) %>%
  use_series(loss)

model.data <- model.data %>%
  select(-id) %>%
  as.matrix()

watch.data <- watch.data %>%
  select(-id) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
xgbtest <- xgb.DMatrix(data=watch.data, label=watch.y)
watchlist <- list(train=xgbtrain, 
                  test=xgbtest)

# change booster to gblinear, so that we are fitting a linear model
# alpha is the L1 regularizer on weights, increase this value will make model more conservative. 
# lambda is the L2 regularizer on weights, increase this value will make model more conservative
param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae", 
              "booster" = "gblinear",
              "nthread" = 12, 
              "alpha" = 0.0001,
              "lambda" = 1)

param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae",
              "eta" = 0.01,
              "subsample" = .9,
              "max_depth" = 6,
              "silent" = 0,
              "nthread" = 12)

#~1800 test mae
bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 120, 
                watchlist = watchlist)

xgb.plot.tree(feature_names = colnames(xgbtrain), model = bst, n_first_tree = 2)

# Compute feature importance matrix
importance_matrix <- xgb.importance(colnames(xgbtrain), model = bst)

################################################################################
#####GBM cont+cat 2 vars#####

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

watch.data <- model.data %>%
  sample_frac(.2) %>%
  select(id, one_of(model.vars))

#remove watch data from training set
model.data <- model.data %>%
  anti_join(watch.data %>%
              select(id))

nrow(model.data) + nrow(oos.data) + nrow(watch.data)
nrow(train)

model.y <- model.data %>%
  inner_join(train.recode %>%
               select(id, loss)) %>%
  select(loss) %>%
  use_series(loss)

watch.y <- watch.data %>%
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

watch.data <- watch.data %>%
  select(-id) %>%
  as.matrix()

oos.data <- oos.data %>%
  select(-id) %>%
  as.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
xgbtest <- xgb.DMatrix(data=watch.data, label=watch.y)
watchlist <- list(train=xgbtrain, 
                  test=xgbtest)

# change booster to gblinear, so that we are fitting a linear model
# alpha is the L1 regularizer on weights, increase this value will make model more conservative. 
# lambda is the L2 regularizer on weights, increase this value will make model more conservative

param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae",
              "eta" = 0.1,
              "subsample" = .9,
              "max_depth" = 6,
              "silent" = 0,
              "nthread" = 12)

#~1370 test mae
bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 120, 
                watchlist = watchlist)

y.hat <- predict(bst, oos.data)
#OOS mae / 1340
data_frame(oos.loss = oos.y,
           loss.pred = y.hat) %>%
  mutate(abs.error = abs(oos.loss-loss.pred)) %>%
  summarise(mae = mean(abs.error))

xgb.dump(bst, with_stats = T)[1:10]

#~1429 test mae
bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 200, 
                watchlist = watchlist)
#OOS mae / 1396
data_frame(oos.loss = oos.y,
           loss.pred = predict(bst, oos.data)) %>%
  mutate(abs.error = abs(oos.loss-loss.pred)) %>%
  summarise(mae = mean(abs.error))

param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae",
              "eta" = 0.3,
              "subsample" = .9,
              "max_depth" = 6,
              "silent" = 0,
              "nthread" = 12)

#~1260 test mae
bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 100, 
                watchlist = watchlist)

#OOS mae /1350
data_frame(oos.loss = oos.y,
           loss.pred = predict(bst, oos.data)) %>%
  mutate(abs.error = abs(oos.loss-loss.pred)) %>%
  summarise(mae = mean(abs.error))

cv.nround <- 500
cv.nfold <- 5

xgb.cv(params = param,
       data = xgbtrain,
       nfold = cv.nfold,
       nrounds = cv.nround,
       early_stopping_rounds = 10)

param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse",
              "eval_metric" = "mae",
              "eta" = 0.3,
              "subsample" = .9,
              "max_depth" = 7,
              "silent" = 0,
              "nthread" = 12)

#~1130 test mae
bst = xgb.train(params = param, 
                data = xgbtrain, 
                nrounds = 200, 
                watchlist = watchlist)
#OOS mae / 1374
data_frame(oos.loss = oos.y,
           loss.pred = predict(bst, oos.data)) %>%
  mutate(abs.error = abs(oos.loss-loss.pred)) %>%
  summarise(mae = mean(abs.error))



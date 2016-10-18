rm(list=ls(all=TRUE))

library(h2o)
library(readr)
library(dplyr)
library(tidyr)
library(e1071)
library(Rtsne)
library(ggplot2)
library(xgboost)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

train.binary <- read_csv("train_recode.csv.gz")
test.binary <- read_csv("test_recode.csv.gz")

localH2O <- h2o.init(ip = 'localhost', nthreads=10, max_mem_size = '64g')

set.seed(666)

model.data <- train.binary %>%
  sample_frac(.8) %>%
  #log loss
  mutate(loss = log(loss))

#final out of sample test set
oos.data <- train.binary %>%
  anti_join(model.data %>%
              select(id))

#remove watch data from training set
model.data <- model.data %>%
  anti_join(oos.data %>%
              select(id))

nrow(model.data) + nrow(oos.data)
nrow(train.binary)

#Keep same data for python
# train.file <- "train_python.csv.gz"
# test.file <- "test_python.csv.gz"
# write.csv(model.data, gzfile(train.file), row.names=FALSE)
# write.csv(oos.data, gzfile(test.file), row.names=FALSE)

x <- setdiff(names(model.data), c('id', 'loss'))
y <- 'loss'

train.h2o <- model.data %>% as.h2o()
test.h2o <- oos.data %>% as.h2o()

h2o.rf.1 <- h2o.randomForest(model_id = 'rf_1',
                             training_frame = train.h2o,
                             x = x,
                             y = y, 
                             ntrees = 50, 
                             max_depth = 20,
                             seed = 111,
                             sample_rate = .9)

h2o.rf.2 <- h2o.randomForest(model_id = 'rf_2',
                             training_frame = train.h2o,
                             x = x,
                             y = y, 
                             ntrees = 100, 
                             max_depth = 20,
                             seed = 222)

h2o.deep.1 <- h2o.deeplearning(model_id = "dl_1", 
                               training_frame = train.h2o, 
                               x = x,
                               y = y,
                               distribution = "gaussian",
                               activation = "Maxout",
                               hidden = c(20,20,20),
                               epochs = 5,
                               l1 = 1e-5,
                               l2 = 1e-5,
                               max_w2 = 10,
                               seed = 333)

h2o.deep.2 <- h2o.deeplearning(model_id = "dl_2", 
                               training_frame = train.h2o, 
                               x = x,
                               y = y,
                               distribution = "gaussian",
                               activation = "Maxout",
                               hidden = c(10,10,10,10,10),
                               epochs = 5,
                               l1 = 1e-5,
                               l2 = 1e-5,
                               max_w2 = 10,
                               seed = 444)

oos.level.1 <- data_frame(id = oos.data$id,
                          rf.1 = h2o.predict(h2o.rf.1, newdata = test.h2o) %>% as.vector(),
                          rf.2 = h2o.predict(h2o.rf.2, newdata = test.h2o) %>% as.vector(),
                          nn.1 = h2o.predict(h2o.deep.1, newdata = test.h2o) %>% as.vector(),
                          nn.2 = h2o.predict(h2o.deep.2, newdata = test.h2o) %>% as.vector())

train.level.1 <- data_frame(id = model.data$id,
                            rf.1 = h2o.predict(h2o.rf.1, newdata = train.h2o) %>% as.vector(),
                            rf.2 = h2o.predict(h2o.rf.2, newdata = train.h2o) %>% as.vector(),
                            nn.1 = h2o.predict(h2o.deep.1, newdata = train.h2o) %>% as.vector(),
                            nn.2 = h2o.predict(h2o.deep.2, newdata = train.h2o) %>% as.vector())



train.tsne <- Rtsne(as.matrix(model.data %>% select(-id, -loss)), 
                    check_duplicates = FALSE, 
                    pca = TRUE, 
                    perplexity = 30, 
                    theta = 0.5, 
                    dims = 3)

oos.tsne <- Rtsne(as.matrix(oos.data %>% select(-id, -loss)), 
                  check_duplicates = FALSE, 
                  pca = TRUE, 
                  perplexity = 30, 
                  theta = 0.5, 
                  dims = 3)

model.y <- model.data %>% use_series(loss)  

model.data <- model.data %>%
  inner_join(train.level.1) %>%
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

oos.test <- oos.data %>%
  inner_join(oos.level.1) %>%
  select(-id, -loss) %>%
  as.matrix()

data_frame(oos.loss = oos.data$loss,
           loss.pred = predict(bst.1, oos.test)) %>%
  mutate(abs.error = abs(exp(oos.loss)-exp(loss.pred))) %>%
  summarise(mae = mean(abs.error))


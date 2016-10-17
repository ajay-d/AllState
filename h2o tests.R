rm(list=ls(all=TRUE))

options(java.parameters = "-Xmx64g" )


library(h2o)
library(readr)
library(dplyr)
library(tidyr)
library(magrittr)
library(extraTrees)

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

valid.data <- model.data %>%
  sample_frac(.2)

#remove watch data from training set
model.data <- model.data %>%
  anti_join(valid.data %>%
              select(id))

nrow(model.data) + nrow(oos.data) + nrow(valid.data)
nrow(train.binary)

#####extratrees#####
x_extra <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()
y_extra <- model.data %>%
  select(loss) %>%
  as.matrix()

extra_1 <- extraTrees(x_extra, 
                      y_extra,
                      ntree=500,
                      numRandomCuts = 1,
                      #numRandomCuts = 5,
                      numThreads = 12)
extra_2 <- extraTrees(x_extra, 
                      y_extra,
                      ntree=500,
                      numRandomCuts = 5,
                      numThreads = 12)

pred_extra_1 <- predict(extra_1, oos.data %>% 
                                     select(-id, -loss) %>%
                                     as.matrix())



x <- setdiff(names(model.data), c('id', 'loss'))
y <- 'loss'

model.data <- model.data %>% as.h2o()
valid.data <- valid.data %>% as.h2o()

h2o.gbm.1 <- h2o.gbm(y = y, 
                     x = x, 
                     model_id='gbm_1',
                     #distribution = 'gamma',
                     distribution = 'gaussian',
                     training_frame = model.data,
                     validation_frame = valid.data,
                     ntrees=1000, 
                     max_depth=10, 
                     learn_rate=0.1,
                     sample_rate = 0.7,
                     col_sample_rate = 0.7,
                     #stopping_rounds = 2,
                     #stopping_tolerance = 0.01,
                     #score_each_iteration = T,
                     seed = 100)

h2o.gbm.1
h2o.mae(h2o.gbm.1, valid = TRUE)
h2o.mae(h2o.gbm.1, train = TRUE)
h2o.varimp(h2o.gbm.1)

test.h2o <- oos.data %>% as.h2o()
pred_1 <- h2o.predict(h2o.gbm.1, newdata = test.h2o) %>%
  as.data.frame() %>%
  as.tbl() %>%
  mutate(y = oos.data$loss,
         mae = abs(y - predict))

mean(pred_1$mae)

#####GBM grid#####

ntrees_opt <- c(100,250,500)
maxdepth_opt <- c(7,8,9,10)
learnrate_opt <- c(.01,.05,0.1,0.2)
hyper_parameters <- list(ntrees=ntrees_opt, 
                         max_depth=maxdepth_opt, 
                         learn_rate=learnrate_opt)

gbm.grid.1 <- h2o.grid("gbm", 
                       grid_id = 'gbm_grid_1',
                       hyper_params = hyper_parameters, 
                       y = y, 
                       x = x, 
                       distribution="gaussian",
                       training_frame = data.train,
                       validation_frame = data.validate)

grid <- NULL
for (model_id in gbm.grid.1@model_ids) {
  mae <- h2o.mae(h2o.getModel(model_id), valid = TRUE)
  
  ntrees <- h2o.getModel(model_id)@allparameters$ntrees
  max_depth <- h2o.getModel(model_id)@allparameters$max_depth
  learn_rate <- h2o.getModel(model_id)@allparameters$learn_rate
  
  df <- data_frame(mae=mae,
                   model_id=model_id,
                   ntrees=ntrees,           
                   max_depth=max_depth,
                   learn_rate=learn_rate)
  grid <- bind_rows(grid, df)
}

#####Deep#####
args(h2o.deeplearning)


#####RF#####

h2o.rf.1 <- h2o.randomForest(y = y, 
                             x = x, 
                             model_id='rf_1',
                             training_frame = model.data,
                             validation_frame = valid.data,
                             ntrees = 100, 
                             max_depth = 30,
                             stopping_rounds = 2,
                             seed = 200)

pred_1 <- h2o.predict(h2o.rf.1, newdata = test.h2o)





h2o.shutdown(prompt=FALSE)


rm(list=ls(all=TRUE))

library(h2o)
library(readr)
library(dplyr)
library(tidyr)
library(e1071)
library(Rtsne)
library(ggplot2)
library(magrittr)
library(SwarmSVM)


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

x <- setdiff(names(model.data), c('id', 'loss'))
y <- 'loss'

model.data <- model.data %>% as.h2o()
valid.data <- valid.data %>% as.h2o()
test.h2o <- oos.data %>% as.h2o()

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

pred_1 <- h2o.predict(h2o.gbm.1, newdata = test.h2o) %>%
  as.data.frame() %>%
  as.tbl() %>%
  mutate(y = oos.data$loss,
         mae = abs(y - exp(predict)))

mean(pred_1$mae)

#####GBM grid#####

ntrees_opt <- c(100,250,500)
maxdepth_opt <- c(7,8,9,10)
learnrate_opt <- c(.01,.05,0.1,0.2)
hyper_parameters <- list(ntrees=ntrees_opt, 
                         max_depth=maxdepth_opt, 
                         learn_rate=learnrate_opt)

gbm.grid.1 <- h2o.grid(algorithm= "gbm", 
                       grid_id = 'gbm_grid_1',
                       hyper_params = hyper_parameters, 
                       y = y, 
                       x = x, 
                       distribution="gaussian",
                       training_frame = model.data,
                       validation_frame = valid.data)

hyper_parameters <- list(max_depth = seq(2,10,2), ##add more levels
                         learn_rate=learnrate_opt)

#stop early
gbm.grid.2 <- h2o.grid(algorithm= "gbm", 
                       grid_id = 'gbm_grid_1',
                       hyper_params = hyper_parameters, 
                       search_criteria = list(strategy = "Cartesian"),
                       y = y, 
                       x = x, 
                       ntrees = 10000,
                       col_sample_rate = 0.8,
                       sample_rate = 0.8,
                       stopping_rounds=5,
                       stopping_metric="MSE",
                       stopping_tolerance=0.1, ##lower
                       
                       distribution="gaussian",
                       training_frame = model.data,
                       validation_frame = valid.data)

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

m1 <- h2o.deeplearning(model_id="dl_model_first", 
                       training_frame=model.data, 
                       validation_frame=valid.data,   ## validation dataset: used for scoring and early stopping
                       x=x,
                       y=y,
                       distribution="gaussian",
                       #activation="Rectifier",  ## default
                       hidden=c(200,200,200),       ## default: 2 hidden layers with 200 neurons each
                       epochs=10,
                       variable_importances=T    ## not enabled by default
)

summary(m1)
head(as.data.frame(h2o.varimp(m1)))

m1 <- h2o.deeplearning(model_id="dl_model_first", 
                       training_frame=model.data, 
                       validation_frame=valid.data,   ## validation dataset: used for scoring and early stopping
                       x=x,
                       y=y,
                       distribution="gaussian",
                       activation="Maxout",  ## default
                       hidden=c(20,20,20),       ## default: 2 hidden layers with 200 neurons each
                       epochs=10,
                       l1=1e-5,
                       l2=1e-5,
                       max_w2=10
)

pred_2 <- h2o.predict(m1, newdata = test.h2o) %>%
  as.data.frame() %>%
  as.tbl() %>%
  mutate(y = oos.data$loss,
         mae = abs(y - exp(predict)))

mean(pred_2$mae)

m2 <- h2o.deeplearning(model_id="dl_model_two", 
                       training_frame=model.data, 
                       validation_frame=valid.data,   ## validation dataset: used for scoring and early stopping
                       x=x,
                       y=y,
                       distribution="gaussian",
                       activation="Rectifier",  ## default
                       hidden=c(200,200,200),       ## default: 2 hidden layers with 200 neurons each
                       epochs=1000000,                      ## hopefully converges earlier...
                       stopping_rounds=2,
                       stopping_metric="MSE",
                       stopping_tolerance=0.01,
                       l1=1e-5,
                       l2=1e-5)
summary(m2)
plot(m2)

m3 <- h2o.deeplearning(model_id="dl_model_two", 
                       training_frame=model.data, 
                       validation_frame=valid.data,   ## validation dataset: used for scoring and early stopping
                       x=x,
                       y=y,
                       distribution="gaussian",
                       activation="Rectifier",  ## default
                       hidden=c(200,200,200),       ## default: 2 hidden layers with 200 neurons each
                       epochs=1000000,                      ## hopefully converges earlier...
                       stopping_rounds=5,
                       stopping_metric="MSE",
                       stopping_tolerance=0.001,
                       l1=1e-5,
                       l2=1e-5)
summary(m3)
plot(m3)

search_criteria = list(strategy = "RandomDiscrete", 
                       max_models = 100, seed=1234567, 
                       stopping_rounds = 5, 
                       stopping_tolerance = 1e-2)

hyper_1 <- list(
  activation = c("Tanh", "Maxout", "Rectifier"),
  hidden = list(c(20,20), c(50,50), c(30,30,30), c(25,25,25,25),
              c(10,10,10,10), c(50,50,50,50), c(20,20,20,20,20)),
  input_dropout_ratio = c(.01,.05,.1,.2)
  )

deep.grid.1 <- h2o.grid(algorithm = "deeplearning",
                        grid_id = "dl_grid_1",
                        hyper_params = hyper_1, 
                        y = y, 
                        x = x, 
                        distribution="gaussian",
                        training_frame = model.data,
                        validation_frame = valid.data,
                        score_validation_samples = 0,
                        epochs = 10,
                        max_w2 = 10,
                        l1 = 1e-5,
                        l2 = 1e-5)
##Faster
hyper_2 <- list(
  activation = c("TanhWithDropout", "MaxoutWithDropout", "RectifierWithDropout"),
  hidden = list(c(20,20), c(50,50), c(30,30,30), c(25,25,25,25),
              c(10,10,10,10), c(50,50,50,50), c(20,20,20,20,20)),
  input_dropout_ratio = c(.01,.05,.1,.2)
  )

deep.grid.2 <- h2o.grid(algorithm = "deeplearning",
                        grid_id = "dl_grid_2",
                        hyper_params = hyper_2, 
                        search_criteria = search_criteria,
                        y = y, 
                        x = x, 
                        training_frame = model.data,
                        validation_frame = valid.data,
                        score_validation_samples = 0,
                        epochs = 10,
                        max_w2 = 10,
                        l1 = 1e-5,
                        l2 = 1e-5)

grid.1 <- h2o.getGrid("dl_grid_1",sort_by="mse",decreasing=FALSE)
grid.2 <- h2o.getGrid("dl_grid_2",sort_by="mse",decreasing=FALSE)

deep.grid.1.summary <- NULL
for (model_id in deep.grid.1@model_ids) {
  mae <- h2o.mae(h2o.getModel(model_id), valid = TRUE)
  
  activation <- h2o.getModel(model_id)@allparameters$activation
  hidden <- h2o.getModel(model_id)@allparameters$hidden
  input_dropout_ratio <- h2o.getModel(model_id)@allparameters$input_dropout_ratio
  
  df <- data_frame(mae=mae,
                   model_id=model_id,
                   activation=activation,           
                   hidden=hidden,
                   input_dropout_ratio=input_dropout_ratio)
  deep.grid.1.summary <- bind_rows(deep.grid.1.summary, df)
}

deep.grid.2.summary <- NULL
for (model_id in deep.grid.2@model_ids) {
  mae <- h2o.mae(h2o.getModel(model_id), valid = TRUE)
  
  activation <- h2o.getModel(model_id)@allparameters$activation
  hidden <- h2o.getModel(model_id)@allparameters$hidden
  input_dropout_ratio <- h2o.getModel(model_id)@allparameters$input_dropout_ratio
  
  df <- data_frame(mae=mae,
                   model_id=model_id,
                   activation=activation,           
                   hidden=hidden,
                   input_dropout_ratio=input_dropout_ratio)
  deep.grid.2.summary <- bind_rows(deep.grid.2.summary, df)
}

predict(h2o.getModel(grid.1@model_ids[[1]]), newdata = test.h2o) %>%
  as.data.frame() %>%
  as.tbl() %>%
  mutate(y = oos.data$loss,
         mae = abs(y - exp(predict))) %>%
  summarise(mae = mean(mae))

predict(h2o.getModel(grid.2@model_ids[[1]]), newdata = test.h2o) %>%
  as.data.frame() %>%
  as.tbl() %>%
  mutate(y = oos.data$loss,
         mae = abs(y - exp(predict))) %>%
  summarise(mae = mean(mae))

h2o.getModel(grid.1@model_ids[[1]])@allparameters$hidden
h2o.getModel(grid.2@model_ids[[1]])@allparameters$hidden

#####RF#####

h2o.rf.1 <- h2o.randomForest(y = y, 
                             x = x, 
                             model_id = 'rf_1',
                             training_frame = model.data,
                             validation_frame = valid.data,
                             ntrees = 100, 
                             max_depth = 30,
                             stopping_rounds = 2,
                             seed = 200,
                             sample_rate = .9)

h2o.predict(h2o.rf.1, newdata = test.h2o) %>%
  as.data.frame() %>%
  as.tbl() %>%
  mutate(y = oos.data$loss,
         mae = abs(y - exp(predict))) %>%
  summarise(mae = mean(mae))

search_criteria = list(strategy = "RandomDiscrete", 
                       max_models = 100, 
                       seed=1234567, 
                       stopping_rounds=5,
                       stopping_metric="MSE",
                       stopping_tolerance=0.001)

hyper_rf <- list(
  max_dept = seq(2,50,2),
  ntrees = seq(50,100,10),
  sample_rate = c(.6, .7, .8, .9, 1)
  )

rf.grid <- h2o.grid(algorithm = "randomForest",
                        grid_id = "rf_grid",
                        hyper_params = hyper_rf, 
                        search_criteria = search_criteria,
                        y = y, 
                        x = x, 
                        training_frame = model.data,
                        validation_frame = valid.data)

h2o.shutdown(prompt=FALSE)


#####SVM Swarm#####

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

svm.train <- model.data %>%
  select(-id, -loss) %>%
  as.matrix()
svm.y <- model.data$loss

svm.valid <- valid.data %>%
  select(-id, -loss) %>%
  as.matrix()
svm.valid.y <- valid.data$loss

csvm.1 = clusterSVM(x = svm.train, y = svm.y,
                    valid.x = svm.valid, valid.y = svm.valid.y,
                    cluster.method = "mlKmeans",
                    type = 11, #11,12,13
                    seed = 2016, verbose = 1, centers = 10)
csvm.1 = clusterSVM(x = svm.train, y = svm.y,
                    valid.x = svm.valid, valid.y = svm.valid.y,
                    cluster.method = "mlKmeans",
                    type = 12, #11,12,13
                    seed = 2016, verbose = 1, centers = 15)

csvm.1$valid.score
csvm.2$valid.score

test.data <- oos.data %>%
  select(-id, -loss) %>%
  as.matrix()

svm.score.1 <- predict(csvm.1, test.data)
svm.score.2 <- predict(csvm.2, test.data)

#####SVM#####

svm.1 <- svm(x = svm.train,
             y = svm.y)

pred_svm <- data_frame(id = oos.data$id,
                       y = oos.data$loss,
                       svm_pred_1 = svm.score.1[[1]],
                       svm_pred_2 = svm.score.2[[1]]) %>%
  mutate(mae.1 = abs(y - exp(svm_pred_1)),
         mae.2 = abs(y - exp(svm_pred_2)))

mean(pred_svm$mae.1)
mean(pred_svm$mae.2)



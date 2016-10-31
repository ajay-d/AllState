rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(ggplot2)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

train.factor <- read_csv("data/train_recode_factor.csv.gz")
test.factor <- read_csv("data/test_recode_factor.csv.gz")

gbm.full <- read_csv("python/gbm_full.csv")
nn.full <- read_csv("python/keras_full.csv")

################################################################################


################################################################################

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full) %>%
  inner_join(nn.full)


##GBM 1 + 2
#1114.66
df <- submission %>%
  mutate(loss = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##2 GBMs + 1 NNs
#1110.37
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(loss = (gbm + nn_pred_1)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##2 GBMs + 4 NNs
#1109.349
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

cor(submission$gbm_bagged_1, submission$gbm_bagged_2)
cor(submission$gbm_bagged_1, submission$nn_pred_1)

cor(submission$nn_pred_1, submission$nn_pred_2)
cor(submission$nn_pred_2, submission$nn_pred_3)
cor(submission$nn_pred_3, submission$nn_pred_4)
cor(submission$nn_pred_3, submission$nn_pred_4)

##4 NNs only
#1119.52
df <- submission %>%
  mutate(loss =  (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##2 GBM+NN models
#1110.02
df <- submission %>%
  mutate(loss.1 = (gbm_bagged_1+nn_pred_1)/2) %>%
  mutate(loss.2 = (gbm_bagged_2+nn_pred_4)/2) %>%
  mutate(loss =  (loss.1 + loss.2)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
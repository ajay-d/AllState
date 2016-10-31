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

submission  <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full) %>%
  inner_join(nn.full)

#GBM Only-python
#1116.67
#Early stop 10

#1114.53
#Early stop 50
df <- submission %>%
  mutate(loss =  gbm_bagged) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#1121.24
df <- submission %>%
  mutate(loss =  nn_pred_1) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

#blend GBM + NN1
#1110.23
df <- submission %>%
  mutate(loss =  (nn_pred_1+gbm_bagged)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)


cor(submission$gbm_bagged, submission$nn_pred_1)
cor(submission$nn_pred_1, submission$nn_pred_2)
cor(submission$nn_pred_3, submission$nn_pred_4)
cor(submission$nn_pred_3, submission$nn_pred_4)

cor(submission$nn_pred_1, submission$nn_pred_4)
cor(submission$gbm_bagged, submission$nn_pred_4)

##1112.32
df <- submission %>%
  mutate(loss =  (nn_pred_1+gbm_bagged+nn_pred_4)/3) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

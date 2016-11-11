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

gbm.2 <- read_csv("python/gbm_full2.csv")
gbm.stack <- read_csv("python/gbm_full_stack.csv")

nn.rms.full <- read_csv("python/keras_cv_full_rms.csv")
nn.ada.full <- read_csv("python/keras_cv_full_ada.csv")

summary(nn.full$nn_pred_1)
summary(nn.rms.full$nn_pred_1)
summary(nn.ada.full$nn_pred_1)

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
cor(submission$nn_pred_1, submission$nn_pred_4)

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

################################################################################


################################################################################

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full) %>%
  inner_join(nn.rms.full)

##2 GBMs + 4 NNs -(RMS)
#1113.08
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##2 GBMs + 1 NNs
#1113.70
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(loss = (gbm + nn_pred_1)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
################################################################################


################################################################################

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full) %>%
  inner_join(nn.ada.full)

##2 GBMs + 4 NNs -(Ada)
#1112.508
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full) %>%
  inner_join(nn.full %>%
               select(id, nn_pred_1)) %>%
  inner_join(nn.rms.full %>%
               select(id, nn_rms_1=nn_pred_1)) %>%
  inner_join(nn.ada.full %>%
                 select(id, nn_ada_1=nn_pred_1))

cor(submission$nn_pred_1, submission$nn_rms_1)
cor(submission$nn_pred_1, submission$nn_ada_1)
cor(submission$nn_rms_1, submission$nn_ada_1)


gbm.full <- read_csv("python/gbm_full.csv")
gbm.full2 <- read_csv("python/gbm_full2.csv")
nn.full <- read_csv("python/keras_full.csv")

nn.ada.full <- read_csv("python/keras_cv_full_ada.csv")

summary(nn.full$nn_pred_1)
summary(nn.ada.full$nn_pred_1)

################################################################################


################################################################################

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full) %>%
  inner_join(gbm.full2) %>%
  inner_join(nn.ada.full)

##GBM 3 only
#1114.04
df <- submission %>%
  mutate(loss = (gbm_bagged_3)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)


##2 GBMs + 4 NNs (new-stopping)
#1110.3
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##3 GBMs + 4 NNs (new-stopping)
#1110.18
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2+gbm_bagged_3)/3) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

################################################################################


################################################################################

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.full %>%
               mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2)) %>%
  inner_join(nn.full %>%
               mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
               select(id, nn)) %>%
  inner_join(nn.ada.full %>%
               mutate(nn_ada = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
               select(id, nn_ada))

cor(submission$gbm, submission$nn)
cor(submission$nn, submission$nn_ada)
cor(submission$gbm, submission$nn_ada)

##1109.44
df <- submission %>%
  mutate(nn_blend = (nn + nn_ada)/2) %>%
  mutate(loss = (gbm + nn_blend)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

################################################################################


################################################################################

submission <- data_frame(id = test.factor$id) %>%
  inner_join(gbm.2) %>%
  inner_join(nn.full) %>%
  inner_join(gbm.stack %>%
               rename(gbm_stack_1=gbm_bagged_1, gbm_stack_2=gbm_bagged_2))

summary(submission$gbm_bagged_3)
summary(submission$gbm_stack_1)
summary(submission$gbm_stack_2)

##1113.54
df <- submission %>%
  mutate(loss = (gbm_bagged_3)) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##1108.77
df <- submission %>%
  mutate(gbm = gbm_bagged_3) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##1109.22
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2)/2) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

##1109.04
df <- submission %>%
  mutate(gbm = (gbm_bagged_1+gbm_bagged_2+gbm_bagged_3)/3) %>%
  mutate(nn = (nn_pred_1+nn_pred_2+nn_pred_3+nn_pred_4)/4) %>%
  mutate(loss = (gbm + nn)/2) %>%
  select(id, loss)
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)

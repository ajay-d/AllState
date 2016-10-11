rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(magrittr)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10)

sample.sub <- read_csv("data/sample_submission.csv.zip")
train <- read_csv("data/train.csv.zip")
test <- read_csv("data/test.csv.zip")

set.seed(666)

train.80 <- train %>%
  sample_frac(.8)

train.20 <- train %>%
  anti_join(train.80 %>%
              select(id))

train.80 %>% count(cat109)
train.20 %>% count(cat109)

dat <- list('N' = nrow(train.80),
            'D' = length(sort(unique(train.sample$level_8))), #number of stratification levels
            
            'y' = train.80$loss,
            
            'll' = train.sample$level_8, #level indicator
            'covar' = features,
            "x_m2" = train.sample$Ret_MinusTwo,
            "x_m1" = train.sample$Ret_MinusOne,
            "x_intra" = train.sample$return.intra.day
)

fit <- stan("model_simple1.stan",
            iter=5000, warmup=3000,
            thin=2, chains=3, seed=252014,
            data = dat)
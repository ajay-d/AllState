rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(rstan)
library(ggplot2)
library(magrittr)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10)

sample.sub <- read_csv("data/sample_submission.csv.zip")
train <- read_csv("data/train.csv.zip")
test <- read_csv("data/test.csv.zip")

set.seed(666)

train.80 %>% count(cat6)
train.20 %>% count(cat6)

train.80 <- train %>%
  sample_frac(.8) %>%
  mutate(cat6_num = c('A'=1, 'B'=2)[cat6])

train.20 <- train %>%
  anti_join(train.80 %>%
              select(id))

#using 1 cat var, with only two levels
dat <- list('N_obs' = nrow(train.80),
            'x_cont' = train.80 %>% select(starts_with('cont')),
            'D' = 2,
            'x_cat' = train.80 %>% use_series(cat6_num),
            'loss' = train.80$loss)

fit <- stan("model 1.stan",
            iter=4000, warmup=2000,
            chains=4, seed=666,
            data = dat)
rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(rstan)
library(ggplot2)
library(magrittr)
library(shinystan)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10)

sample.sub <- read_csv("data/sample_submission.csv.zip")
train <- read_csv("data/train.csv.zip")
test <- read_csv("data/test.csv.zip")

set.seed(666)

train.80 <- train %>%
  sample_frac(.8) %>%
  mutate(cat6_num = c('A'=1, 'B'=2)[cat6])

train.20 <- train %>%
  anti_join(train.80 %>%
              select(id)) %>%
  mutate(cat6_num = c('A'=1, 'B'=2)[cat6])

train.80 %>% count(cat6)
train.20 %>% count(cat6)

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

fit
traceplot(fit, pars=c('sigma', 'beta[1,1]'))

launch_shinystan(fit)

f <- extract(fit)

train.20.long <- train.20 %>%
  select(id, loss, starts_with('cont'), cat6_num) %>%
  gather(variable, value, -loss, -id, -cat6_num) %>%
  separate(variable, into=c('variable', 'cont_num'), convert=TRUE, sep='t')
  

beta_group <- f$beta %>%
  as_data_frame() %>%
  gather(cat6_num, value) %>%
  separate(cat6_num, into=c('cat6_num', 'beta_cont'), convert=TRUE) %>%
  group_by(cat6_num, beta_cont) %>%
  summarise(n = n(),
            beta = median(value))
  
train.20.long <- train.20.long %>%
  inner_join(beta_group, by=c('cat6_num'='cat6_num',
                              'cont_num' = 'beta_cont')) %>%
  select(-variable) %>%
  group_by(id) %>%
  summarise(loss = max(loss),
            mu = sum(beta*value)) %>%
  mutate(loss.pred = exp(mu))

#mae
train.20.long %>%
  summarise(mae = mean(abs(loss-loss.pred)))


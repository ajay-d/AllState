rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(lazyeval)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10)

sample.sub <- read_csv("data/sample_submission.csv.zip")
train <- read_csv("data/train.csv.zip")
test <- read_csv("data/test.csv.zip")

all.cat.vars <- train %>%
  select(starts_with('cat'))

cat.table <- NULL
for(i in names(all.cat.vars)) {
  
  #print(all.cat.vars %>% count_(i))
  
  #get highest category, and count
  var.max <- all.cat.vars %>%
    count_(i, sort=TRUE) %>%
    filter(row_number()==1)
  
  dots <- list(interp(~ifelse(is.na(var), 1, 0), var = as.name(paste0(i))))
  
  #Count number of NA columns
  na.cnt <- all.cat.vars %>% 
    mutate_(.dots = setNames(dots, paste("na.var"))) %>% 
    count(na.var)
  
  #Count cardinality
  df <- data_frame(var = i,
                   categories = nrow(all.cat.vars %>% count_(i)),
                   populated = na.cnt %>% filter(na.var==0) %>% select(n) %>% as.numeric,
                   max.cat.value = as.character(var.max[[1,1]]),
                   max.cat.population = var.max[[1,2]])
  
  cat.table <- bind_rows(cat.table, df)
}
cat.table <- cat.table %>%
  arrange(desc(categories))

summary(train$loss)
quantile(train$loss)

train <- train %>%
  mutate(log.loss = log(loss))

summary(train$log.loss)

ggplot(train) + 
  geom_histogram(aes(loss))

ggplot(train) + 
  geom_histogram(aes(loss)) +
  scale_x_log10()


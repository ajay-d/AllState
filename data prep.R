rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(lazyeval)
library(magrittr)

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

all.cont.vars <- train %>%
  select(starts_with('cont'))

cont.table <- NULL
for(i in names(all.cont.vars)) {
  
  #get highest category, and count
  var.max <- all.cont.vars %>%
    count_(i, sort=TRUE) %>%
    filter(row_number()==1)
  
  dots <- list(interp(~mean(var), var = as.name(paste0(i))),
               interp(~median(var), var = as.name(paste0(i))),
               interp(~quantile(var, .01), var = as.name(paste0(i))),
               interp(~quantile(var, .10), var = as.name(paste0(i))),
               interp(~quantile(var, .90), var = as.name(paste0(i))),
               interp(~quantile(var, .99), var = as.name(paste0(i))),
               interp(~sd(var), var = as.name(paste0(i))))

  var.summary <- all.cont.vars %>%
    summarise_(.dots = setNames(dots, c("mean", "median", "q01", "q10", "q90", "q99", "sd")))
  
  #Count cardinality
  df <- data_frame(var = i,
                   mean = var.summary$mean,
                   median = var.summary$median,
                   mode = var.max[1,1] %>% as.numeric(),
                   mode.n = var.max[1,2] %>% as.numeric(),
                   sd = var.summary[1,'sd'] %>% as.numeric(),
                   q01 = var.summary[1,'q01'] %>% as.numeric(),
                   q10 = var.summary[1,'q10'] %>% as.numeric(),
                   q90 = var.summary[1,'q90'] %>% as.numeric(),
                   q99 = var.summary[1,'q99'] %>% as.numeric())
  
  cont.table <- bind_rows(cont.table, df)
}

cat.table.long <- NULL
high.cat.vars <- cat.table %>%
  #filter(categories > 4) %>%
  #filter(categories > 2) %>%
  use_series(var)

for(i in high.cat.vars) {
  
  var.count <- all.cat.vars %>%
    count_(i, sort=TRUE)
  
  loss.var <- train %>%
    group_by_(i) %>%
    summarise(sd = sd(loss),
              mean = mean(loss))
  
  var.table <- inner_join(var.count, loss.var)
  
  for(j in 1:nrow(var.count)) {
    df <- data_frame(var = i,
                     category = var.table[[j,i]],
                     n = var.table[[j,'n']],
                     mean.loss = var.table[[j,'mean']],
                     sd = var.table[[j,'sd']],
                     cv = mean.loss/sd)
        
    cat.table.long <- bind_rows(cat.table.long, df) %>%
      filter(n > 1)
  }
  
}

cat.table.long %>%
  group_by(var) %>%
  summarise(sd = mean(sd)) %>%
  arrange(sd)

#start with just the simple two var categories
cat.2 <- cat.table.long %>%
  group_by(var) %>%
  mutate(n.cat = n(),
         cv.mean = mean(cv)) %>%
  filter(n.cat == 2) %>%
  arrange(cv.mean)

summary(train$loss)
quantile(train$loss)

train <- train %>%
  mutate(log.loss = log(loss))

summary(train$log.loss)

sum(cat.table$categories)

ggplot(train) + 
  geom_histogram(aes(loss))

ggplot(train) + 
  geom_histogram(aes(loss)) +
  scale_x_log10()

ggplot(train) + 
  geom_histogram(aes(log.loss), binwidth = .1)


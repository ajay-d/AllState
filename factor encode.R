rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

train <- read_csv("data/train.csv.zip")
test <- read_csv("data/test.csv.zip")

###############################################
#First ran through and used the cat vars that came up in the variable importance matrix from GBM
train <- train %>%
  mutate(new_cat1 = paste0(cat80, cat101),
         new_cat2 = paste0(cat79, cat101),
         new_cat3 = paste0(cat79, cat80),
         new_cat4 = paste0(cat79, cat80, cat101))

test <- test %>%
  mutate(new_cat1 = paste0(cat80, cat101),
         new_cat2 = paste0(cat79, cat101),
         new_cat3 = paste0(cat79, cat80),
         new_cat4 = paste0(cat79, cat80, cat101))

###############################################

all.cat.vars <- train %>%
  select(contains('cat'))
  #select(starts_with('cat'))

cat.table <- NULL
for(i in names(all.cat.vars)) {
  
  #get cardinality
  var.card <- all.cat.vars %>%
    count_(i, sort=TRUE)
  
  df <- data_frame(var = i,
                   categories = nrow(var.card))
  
  cat.table <- bind_rows(cat.table, df)
} 

cat.table <- cat.table %>%
  arrange(desc(categories))

cat.table.long <- NULL
for(i in names(all.cat.vars)) {
  
  train.var <- train %>%
    group_by_(i) %>%
    summarise(sd = sd(loss),
              mean = mean(loss),
              n.train = n()) %>%
    mutate(pct.train = n.train/sum(n.train),
           cv = mean/sd)
  
  test.var <- test %>%
    group_by_(i) %>%
    summarise(n.test = n()) %>%
    mutate(pct.test = n.test/sum(n.test))
  
  var.table <- full_join(train.var, test.var) %>%
    rename_(value=i) %>%
    mutate(var = i)
  
  cat.table.long <- bind_rows(cat.table.long, var.table)
}
  

cat.table.long %>%
  filter(n.train > 1, n.test > 1) %>%
  group_by(var) %>%
  summarise(sd = mean(sd)) %>%
  arrange(sd)

cat.table.long %>%
  filter(n.train > 1, n.test > 1) %>%
  group_by(var) %>%
  summarise(cv = mean(cv)) %>%
  arrange(cv)

cat.table.long <- cat.table.long %>%
  replace_na(list(n.test=0, n.train=0)) %>%
  mutate(n.total = n.test + n.train) %>%
  group_by(var) %>%
  mutate(pct.total = n.total/sum(n.total)) %>%
  arrange(var, desc(n.total)) %>%
    mutate(test.flag = ifelse(abs(pct.train - pct.test) > .001, 1, 0),
         #start at 0
         cat.number = row_number()-1,
         cat.number = ifelse(pct.total < .001, NA, cat.number)) %>%
  fill(cat.number)

train.recode.factor <- train %>% select(id)
test.recode.factor <- test %>% select(id)
for(i in names(all.cat.vars)) {
  
  recode <- cat.table.long %>%
    ungroup() %>%
    filter(var == i) %>%
    select(value, cat.number) %>%
    setNames(c(i, 'cat.number'))
  
  df_train <- train %>%
    select_('id', i) %>%
    inner_join(recode) %>%
    #drop old variable
    select(-2) %>%
    setNames(c('id', i))
  
  df_test <- test %>%
    select_('id', i) %>%
    inner_join(recode) %>%
    #drop old variable
    select(-2) %>%
    setNames(c('id', i))
  
  train.recode.factor <- train.recode.factor %>%
    inner_join(df_train)
  
  test.recode.factor <- test.recode.factor %>%
    inner_join(df_test)
}

train.recode.factor <- train.recode.factor %>%
  inner_join(train %>%
               select(id, starts_with('cont'))) %>%
  inner_join(train %>%
               select(id, loss))

test.recode.factor <- test.recode.factor %>%
  inner_join(test %>%
               select(id, starts_with('cont')))

#order names the same
ordered.names <- train.recode.factor %>%
  select(-id, -loss) %>%
  names %>%
  sort

train.recode.factor <- train.recode.factor %>%
  select(id, one_of(ordered.names), loss)

test.recode.factor <- test.recode.factor %>%
  select(id, one_of(ordered.names))

train.file <- "data/train_recode_factor.csv.gz"
test.file <- "data/test_recode_factor.csv.gz"

write.csv(train.recode.factor, gzfile(train.file), row.names=FALSE)
write.csv(test.recode.factor, gzfile(test.file), row.names=FALSE)



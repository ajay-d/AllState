n <- parallel::detectCores()
cl <- makeCluster(n-1)
registerDoSNOW(cl)

l <- length(cat.2plus.vars)

pb <- txtProgressBar(max=l, style=3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress=progress)

train.recode.binary <- foreach(id=1:l,
                               .packages=c("dplyr", "tidyr", "magrittr", "stringr"),
                               .verbose=FALSE,
                               .combine=inner_join,
                               .inorder=FALSE,
                               .options.snow=opts) %dopar% {
           
            i <- cat.2plus.vars[id]
           
            n.new.vars <- cat.table.long %>%
              filter(var==i) %>%
              summarise(cat.number = max(cat.number)) %>%
              use_series(cat.number)
            
            len.binary <- nchar(R.utils::intToBin(n.new.vars))
            
            col.names <- paste(i, 1:len.binary, sep='_')
            col.values <- R.utils::intToBin(n.new.vars) %>% str_split('')
            col.values[[1]]
            
            train.var <- train.recode %>%
              select_('id', i) %>%
              rename_(value=i) %>%
              inner_join(cat.table.long %>%
                           ungroup %>%
                           filter(var==i) %>%
                           select(value, cat.number)) %>%
              mutate(binary = R.utils::intToBin(cat.number)) %>%
              rowwise() %>%
              mutate(binary = binary %>% str_split_fixed('', n=len.binary) %>% as.character() %>% str_c(collapse='_')) %>%
              #rowwise()
              #mutate(binary = R.utils::intToBin(cat.number) %>% str_split('') %>% extract2(1) %>% paste(collapse = "_"))
              separate(binary, into=col.names, sep='_') %>%
              select(-cat.number, -value) %>%
              mutate_if(is.character, as.numeric)
            
            train.var
                            }

test.recode.binary <- foreach(id=1:l,
                              .packages=c("dplyr", "tidyr", "magrittr", "stringr"),
                              .verbose=FALSE,
                              .combine=inner_join,
                              .inorder=FALSE,
                              .options.snow=opts) %dopar% {
           
            i <- cat.2plus.vars[id]
           
            n.new.vars <- cat.table.long %>%
              filter(var==i) %>%
              summarise(cat.number = max(cat.number)) %>%
              use_series(cat.number)
            
            len.binary <- nchar(R.utils::intToBin(n.new.vars))
            
            col.names <- paste(i, 1:len.binary, sep='_')
            col.values <- R.utils::intToBin(n.new.vars) %>% str_split('')
            col.values[[1]]
            
            test.var <- test.recode %>%
              select_('id', i) %>%
              rename_(value=i) %>%
              inner_join(cat.table.long %>%
                           ungroup %>%
                           filter(var==i) %>%
                           select(value, cat.number)) %>%
              mutate(binary = R.utils::intToBin(cat.number)) %>%
              rowwise() %>%
              mutate(binary = binary %>% str_split_fixed('', n=len.binary) %>% as.character() %>% str_c(collapse='_')) %>%
              #rowwise()
              #mutate(binary = R.utils::intToBin(cat.number) %>% str_split('') %>% extract2(1) %>% paste(collapse = "_"))
              separate(binary, into=col.names, sep='_') %>%
              select(-cat.number, -value) %>%
              mutate_if(is.character, as.numeric)
            
            test.var
                            }


stopCluster(cl)

train.binary <- train.recode %>%
  select(id, one_of(cat.2.vars)) %>%
  inner_join(train.recode.binary) %>%
  inner_join(train %>%
               select(id, starts_with('cont'))) %>%
  inner_join(train %>%
               select(id, loss))

test.binary <- test.recode %>%
  select(id, one_of(cat.2.vars)) %>%
  inner_join(test.recode.binary) %>%
  inner_join(test %>%
               select(id, starts_with('cont')))

#order names the same
ordered.names <- train.binary %>%
  select(-id, -loss) %>%
  names %>%
  sort

train.binary <- train.binary %>%
  select(id, one_of(ordered.names), loss)

test.binary <- test.binary %>%
  select(id, one_of(ordered.names))


train.file <- "train_recode.csv.gz"
test.file <- "test_recode.csv.gz"

write.csv(train.binary, gzfile(train.file), row.names=FALSE)
write.csv(test.binary, gzfile(test.file), row.names=FALSE)

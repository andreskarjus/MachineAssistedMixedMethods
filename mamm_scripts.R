




library(tidyverse)
library(patchwork)
library(caret)
library(boot)
library(magick)

library(scales)
library(colorspace)
library(ggrepel)

library(lme4)

library(gsbm) 
library(ggraph)
library(tidygraph)



#conda_install("r-reticulate", c("openai","tiktoken") )


doprompt = function(n){
  # task = paste0("Tag each of these ",n, " USSR news TEXTS with topic TAGS. Line breaks separate TEXTS. Give only 1 TAG per TEXT and use only TAGS in this list; output the ", n," TAGS only, one per line. TAG list:\n", xlist )
  # task = paste0("Tag each of these ",n, " USSR news TEXTS with topic TAGS. Line breaks separate TEXTS. Give only 1 TAG per TEXT and use only TAGS in this list; output the ", n," TAGS only, one per line." )
  if(n==1){
    task = paste0("Classify this USSR news TEXT with a topic TAG. Output this single TAG only. Important: ONLY use one of the TAGS defined in this LIST:" )
  } else {
    task = paste0("Classify each of these ",n, " USSR news TEXTS with one of these topic TAGS. Line breaks separate the TEXTS. These TEXTS are unrelated, evaluate one by one. Output only 1 TAG per TEXT; ", n," TAGS in total, one per line. Importantly, ONLY use the TAGS defined in this LIST:" )
    # task = paste0("Classify each of these ",n, " USSR news TEXTS with these topic TAGS. Line breaks separate TEXTS. Output only 1 TAG per TEXT, ", n," TAGS in total, one per line. Only use TAGS from the LIST above!" )
    #task = paste0("Classify each of these ",n, " USSR news TEXTS with these topic TAGS. TEXTS is a JSON list. Output JSON list of TAGS, 1 TAG per TEXT, ", n," TAGS in total. Only use TAGS from this list:" )
  }
  # paste0(task, "\n\nTEXTS:\n", paste0(1:n,". ", ex) %>% paste0(collapse="\n") ) 
  return(task)
}

# doprompt(test$OUTLINE[75:80], listru) %>% cat

runprompt = function(topiclist, dat, examples, labels, gptmodel = "gpt-3.5-turbo-0301", ntry=3, backoff="Misc", tagonly=T, truth="generic3"){
  dat$truth=dat[,truth]
  
  py$gptmodel = gptmodel
  nex=length(examples) 
  #jex = dat$OUTLINE[examples] %>% toJSON()
  #py$sysprompt = doprompt(topiclist, nex) %>% r_to_py()
  if(nex==1){
    py$sysprompt = "Classify this Texts using one of these Tags"  
    py$userprompt = paste0(doprompt(nex), "\n", topiclist, "\n\nTEXT: ", dat$OUTLINE[examples])
  } else {
    py$sysprompt = paste("Classify these Texts using these Tags")
    if(tagonly){
      py$userprompt = paste0(doprompt(nex), "\n", topiclist, "\n\nTEXTS: ", dat$OUTLINE[examples] %>% paste0(collapse="\n"), "\n\nNow classify, assign one suitable TAG (", paste(sort(labels), collapse=", "), ") to each TEXT") # tags-only
    } else {
      py$userprompt = paste0("Texts:\n", dat$OUTLINE[examples] %>% paste0(collapse="\n"), '\n\nClassify each of these ', nex,' USSR news texts with one of these topic tags. Line breaks separate the texts above. These texts are unrelated, evaluate one by one! Output only 1 tag per text, one text per line, like this: "Text | Tag". (output ', nex ,' lines total). Importantly, ONLY use the tags defined in this LIST:\n', topiclist)
    }
    
  }
  
  #py_run_string("print(r.userprompt)")
  ntries = 0
  done=F
  tryCatch({
    while(ntries <=ntry & !done){
      ntries=ntries+1
      py_run_string(
        '
try:
    tmpin = openai.ChatCompletion.create(
    model=gptmodel, 
    messages=[
    {"role": "system", "content": sysprompt},  # only use for batch?
    {"role": "user", "content": userprompt}],
    #{"role": "user", "content": userprompt2},
    #{"role": "user", "content": userprompt3}],
    temperature=0)    # or allow variation in case gets stuck?
except Exception as e:
    print(e)  # will print error
    raise  # will pass error to R
#print(tmpin)
'
      )
      #res = py$tmpin["choices"][[1]]["message"]["content"] %>% fromJSON()
      tmpin <<-py$tmpin # debug
      if(tagonly){
        res = py$tmpin["choices"][[1]]["message"]["content"] %>% strsplit("\n") %>% unlist(F,F) %>% 
          gsub("TAG[:=][ ]*", "", ., ignore.case=T)  # sometimes wants to add prefix, this attempt to fix it
      } else {
        res = py$tmpin["choices"][[1]]["message"]["content"] %>% strsplit("\n") %>% unlist(F,F) %>% 
          gsub("^[^|]*\\|[ ]*(.*)","\\1",.) %>%   # remove text
          gsub("[[:punct:][:space:]]", "",.)   # remove trailing stuff if any
      }
      
      if(!all(res %in% labels) & length(res)==nex ){ # if correct n tags but hallucinates, backoff to default garbage category
        print(paste("Model hallucinates, attempting backoff (", res[!(res %in% labels)] %>% paste(collapse=" | ") ) )
        res[!(res %in% labels)] = backoff
      }
      
      done = length(res)==nex  & all(res %in% labels)  # if output format matches and n tags ok then done
      if(!done){print("Mismatch, trying again")}
    }
  }, error=function(e){stop(e)}, warning=function(e){print(e)})
  
  if(!done){
    print(paste("Rerun did not solve mismatch, skipping example block:", substr(dat$OUTLINE[examples][1], 1,90), paste(res,collapse=" ")) )
    res = rep(NA, nex) # assigning NA to result, can be retried later
  }
  
  tmp = cbind(dat %>% slice(examples),
              tibble(tag = res, 
                     tokenstotal=py$tmpin$usage$total_tokens/nex,  # divided bc n batched yields n lines per prompt
                     tokensin=py$tmpin$usage$prompt_tokens/nex,
                     tokensout=py$tmpin$usage$completion_tokens/nex
              )
  ) %>% mutate(match = tag==truth)   # ---currently manual here, pick which testset
  return(tmp)
}


# x1 = runprompt(listen, test, 10:20, labels=unique(test$generic2), gptmodel = "gpt-3.5-turbo", tagonly = F); sum(x1$tag==x1$generic)
# x1 = runprompt(listen, test, 10:20, gptmodel = "gpt-4"); sum(x1$tag==x1$generic)


#### Testing

testloop = function(test, topiclist, esubset=1:100, labels, gptmodel="gpt-3.5-turbo", batchsize=1, 
                    ntry=2, max_attempts=10, initial_delay=41, dfactor=1.5, backoff="Misc", tagonly=T){
  # if batchsize=1, then single example per prompt; if more then batched
  batches = split(esubset, ceiling(seq_along(esubset)/batchsize))
  i=1
  attempts=0
  delay = initial_delay
  res = vector("list", length(batches))
  tryCatch({
    while (attempts <= max_attempts & i <= length(batches)) {
      # prep wait time in case api fails:
      attempts = attempts + 1; delay = delay * dfactor
      tmp=NULL
      # try posting to API:
      tryCatch({
        print(i)
        tmp = runprompt(topiclist, test, batches[[i]], labels = labels, 
                        gptmodel = gptmodel, ntry=ntry, backoff=backoff, tagonly=tagonly) 
        
        # if error in API, then stops here, then catches, and this part won't run:
        res[[i]] = tmp
        debugres <<- res # debug
        i=i+1
        attempts=0; delay = initial_delay # reset api waiting, as no backoff was necessary
        Sys.sleep(0.3) # add tiny break
        # }
        
      }, error = function(e) {
        if (attempts >= max_attempts) {
          # Max attempts reached, re-throw the error
          stop(paste("Max attempts reached for", e))
        } else {
          print(paste(attempts, "attempt, waiting on API for", delay, "seconds because", e))
          Sys.sleep(delay)
        }
      })
    }
  }, error=function(e){print(e)}, warning=function(e){print(e)})
  resdebug <<- res  # debug
  return(do.call(rbind, res))
  
}



checktokens=function(labels, tokmodel="cl100k_base"){
  py$tokmodel = tokmodel
  py_run_string("en=tiktoken.get_encoding(tokmodel)")
  toklist = tibble(labels=labels, tokens=NA, length=NA)
  for(i in 1:length(labels)){
    py$target = labels[i]
    py_run_string('tks = en.encode(target)') 
    toklist[i, "tokens"] = py$tks %>% paste(collapse=" ")
    toklist[i, "length"] = py$tks %>% length()
  }
  attr(toklist, "bias")= toklist %>% pull(tokens) %>% 
    strsplit(" ") %>% unlist(F,F) %>% 
    setNames(rep(100, length(.)), .) %>% as.list()
  return(toklist)
}


singletester = function(test=NULL, labels=NULL, gptmodel="gpt-3.5-turbo", max_attempts=10, initial_delay=41, dfactor=1.5, backoff=NULL, tagonly=T, column="text", max_tokens_int=1000L, logit_bias=NULL, temp=0, verbose=F){
  
  # if data frame then takes and appends column, if vector then outputs vector
  if(is.data.frame(test)){
    batches = as.data.frame(test)[, column, drop=T] # in case tibble
  } else {
    batches=test
  }
  
  i=1
  attempts=0
  delay = initial_delay
  reslist = vector("list", length(batches))
  py$gptmodel=gptmodel
  py$max_tokens = max_tokens_int
  py$temp = temp
  if(is.null(logit_bias)){
    py_run_string("logit_bias={}")
  } else {
    py$logit_bias=logit_bias
  }
  
  
  while(attempts <= max_attempts & i <= length(batches)) {
    # prep wait time in case api fails:
    attempts = attempts + 1; delay = delay * dfactor
    tmp=NULL
    
    # try posting to API:
    if(!is.null(test) ){
      pr = batches[i]
    } else {
      stop("No input provided")
    }
    py$userprompt = pr
    tryCatch({
      if(verbose){ cat(paste0(" ", i, " "))}
      
      # stop("!") debug
      py_run_string(
        '
try:
    tmpin = openai.ChatCompletion.create(
    model=gptmodel, 
    messages=[
    #{"role": "system", "content": sysprompt},  # only use for batch?
    {"role": "user", "content": userprompt}],
    max_tokens=max_tokens,
    temperature=temp,
    logit_bias=logit_bias
    )    # or allow variation in case gets stuck?
except Exception as e:
    print(e)  # will print error
    raise  # will pass error to R
#print(tmpin)
'
      )
      # check if object exists, if API worked
      py_run_string(
        "
try:
  tmpin
except NameError:
  var_exists = False
else:
  var_exists = True
"
      )
      
      
      # if API bugs out, catches here, rest won't run:
      # If API does return something; throws catchable error if failed
      tmpin <<-py$tmpin # debug
      res = py$tmpin["choices"][[1]]["message"]["content"] 
      
      
      if(!py$var_exists || res =="" | is.null(res) | is.na(res) ){
        stop("API didn't throw error but didn't give output either.")
      }
      
      if(!tagonly | is.null(labels)){
        # do nothing
      } else { # check tags
        if(tagonly & !is.null(labels) & !(res %in% labels)){ # if tag but wrong output, try cleaning
          res = str_extract(res, labels %>% paste0(collapse="|"))
        }
        
        # if still
        if(!is.null(labels) & !(res %in% labels) & !is.null(backoff) ){ 
          # if correct n tags but hallucinates, backoff to default 
          print(paste("Model hallucinates, attempting backoff (", res ))
          res = backoff
        }
        
        # if still bad and didn't do backoff
        if(!is.null(labels) & !(res %in% labels)){
          print(paste("Skipping example:", substr(batches[i], 1,90), paste(res,collapse=" ")) )
          res = NA # assigning NA to result, can be retried later
        }
      }
      # runs only if API succeeded
      if(is.data.frame(test)){
        tmp = cbind(test %>% slice(i),
                    tibble(result = res, 
                           tokenstotal=py$tmpin$usage$total_tokens,  
                           tokensin=py$tmpin$usage$prompt_tokens,
                           tokensout=py$tmpin$usage$completion_tokens
                    )
        )
      } else {
        tmp=res
      }
      
      reslist[[i]] = tmp
      # cleanup
      rm(res, tmp)
      py_run_string("del(tmpin)")
      resdebug <<- reslist  # debug
      
      # ready to move on
      i=i+1
      attempts=0; delay = initial_delay # reset api waiting, as no backoff was necessary
      Sys.sleep(0.1) # add tiny break just in case
      # }
      
    }, error = function(e) {
      if (attempts >= max_attempts) {
        # Max attempts reached, re-throw the error
        print(paste("Returning partial list object, because max attempts reached. Error:", e))
        return(reslist)
      } else {
        print(paste(attempts, "attempt, waiting on API for", delay, "seconds because", e))
        Sys.sleep(delay)
      }
    })
  }
  if(is.data.frame(test)){
    if(length(reslist)!=nrow(test)){
      warning("Nrow mismatch! Returning object, but likely some missing")
    }
    return(do.call(rbind, reslist))
    
  } else {
    x=unlist(reslist, F,F)  # assumes single output right now
    if(length(x)!=length(test)){
      warning("Length mismatch! Returning object, but likely misaligned")
    }
    return(x)
  }
}




#### books ####

mergeShortStrings <- function(textVector, minLength = 200) {
  # Initialize a list to hold the merged strings
  mergedStrings <- list()
  
  # Initialize a temporary string for merging short strings
  tempString <- ""
  
  # Loop over the vector
  for (i in 1:length(textVector)) {
    # If the current string is shorter than the minimum length, add it to the temporary string
    if (nchar(textVector[i]) < minLength) {
      tempString <- paste(tempString, textVector[i])
      
      # If the temporary string is now longer than the minimum length, add it to the list and reset the temporary string
      if (nchar(tempString) >= minLength) {
        mergedStrings <- c(mergedStrings, tempString)
        tempString <- ""
      }
    } else {
      # If the current string is longer than the minimum length, add it to the list
      mergedStrings <- c(mergedStrings, textVector[i])
    }
  }
  
  # If there's anything left in the temporary string after the loop, add it to the list
  if (nchar(tempString) > 0) {
    mergedStrings <- c(mergedStrings, tempString)
  }
  
  # Return the list of merged strings
  return(unlist(mergedStrings) %>% gsub("^[ ]+", "",.))
}

#### helpers ####

accuracy = function(dat, truth, test="result"){
  confusionMatrix(as.factor(dat[,test,drop=T]), as.factor(dat[,truth,drop=T]), mode = "everything") %>% 
    {c(.$overall[c("Accuracy", "Kappa" )], colMeans(.$byClass)[c("F1", "Precision", "Recall")] )} %>% as.data.frame()
}

splitstrings <- function(string, chunk_size=8000){
  string_length <- nchar(string)
  n_chunks <- ceiling(string_length / chunk_size)
  
  chunks <- sapply(1:n_chunks, function(i){
    start <- ((i-1) * chunk_size) + 1
    end <- min(i * chunk_size, string_length)
    substr(string, start, end)
  })
  
  return(chunks)
}




#### Bootstrap ####

# pred=gpt3single$tag; truth=gpt3single$generic3;
# newdata=ns; predvalue="result";
# modelstring=socmod; nboot=10; bootcols=1:2
# btype="percent"

bootstrapintervals = function(pred, truth, newdata, predvalue, modelstring, nboot=1000, bootcols =8, btype="basic", bootci=F){
  require(boot)
  # Compute confusion matrix
  confmat = table(truth, pred)
  # Convert confusion matrix to transition (misclassification) probabilities
  transprob = confmat/rowSums(confmat)
  #transprob
  # bootdat = tmp; predvalue = "online" # debug
  
  # Statistic function for bootstrapping
  # this currently works for factors; could be adapted to numeric
  bootfun = function(bdata, indices, transprob, predvalue,modelstring) {
    # Draw a bootstrap sample, replace according to probs
    # indices=sample(1:nrow(newdata), nrow(newdata), replace=T)
    bootdat = bdata[indices,,drop=F]
    bootdat[,predvalue] = transprob[bootdat[,predvalue,drop=F] %>% pull(1), ] %>% 
      apply(1,function(x) sample(colnames(transprob), 1, prob=x))
    #bootdat[,predvalue] = factor(bootdat[,predvalue])
    
    # Fit the model
    #tryCatch({
    if(is.character(modelstring)){
      # the model string currently needs to have a . placeholder for the data for this to work
      res = bootdat %>% {eval(parse(text=modelstring))} %>% as.matrix()
    }
    if(is.function(modelstring)){
      res = modelstring(bootdat)
      
    }
    return(res)
    #}, warning=function(e){print(e)})
  }
  
  
  # Conduct bootstrap
  boot_results = boot(data = newdata, statistic = bootfun, R = nboot, 
                      transprob=transprob, predvalue=predvalue,modelstring=modelstring)
  # boot_results$t
  # 95%: 1.96*SE
  if(!bootci){
    confints = apply(boot_results$t,2, function(x) sd(x,na.rm=T)*1.96 ) %>%  # /sqrt(length(na.omit(x)))
      {data.frame(statistic=colnames(boot_results$t0), value=.)}
    # this may break if for example bootstrapping % of tag and tag never occurs in
    # bootstrapped data, then it would be always na and sd would be NA
  }
  
  if(bootci){
    confints = tibble()
    for(i in seq_along(bootcols)){
      x = boot.ci(boot_results, type = btype, 
                  index=bootcols[i])[[sub("perc", "percent", btype) %>% sub("norm","normal",.)]][,c(4,5)-2,drop=F] %>%  # upper and lower
        as.data.frame()
      x = cbind(colnames(boot_results$t0)[bootcols[i]], boot_results$t0[bootcols[i]], x)
      colnames(x) = c("statistic", "value", "lower", "upper")
      confints = rbind(confints,
                       x
      )
    }
  }
  
  return(confints)
}

#### Semantic change ####

dolexchange = function(corpuspath, subcorpuspath,corpusname,nsamp=30,repl=F){
  truth = left_join(
    read_delim(file.path(corpuspath, subcorpuspath, "truth", "graded.txt"), col_names = c("target", "graded")),
    read_delim(file.path(corpuspath, subcorpuspath, "truth", "binary.txt"), col_names = c("target", "binary")),
    by="target"
  ) %>% mutate(rank=rank(graded, ties.method='average'))
  
  corp1 = tibble(
    lemma = read_lines(file.path(corpuspath,subcorpuspath, "corpus1/lemma", 
                                 paste0(corpusname[1], ".txt"))),
    token = read_lines(file.path(corpuspath,subcorpuspath, "corpus1/token", 
                                 paste0(corpusname[1], ".txt"))) 
  ) %>% 
    filter(nchar(token) %in% 70:200 ) %>% # filter overly long ones
    filter(grepl(paste0(truth$target,collapse="|"), lemma)) %>% 
    mutate(token=token %>% gsub(" ([,.!?;:')])","\\1",.) %>% 
             gsub("\\( ", "(", .) %>% gsub(' "$',"", .) %>% 
             gsub("do n't", "don't",., ignore.case=T) %>% 
             gsub("@@[0-9]*","",.)
    )
  
  
  corp2 = tibble(
    lemma=read_lines(file.path(corpuspath,subcorpuspath, "corpus2/lemma", 
                               paste0(corpusname[2], ".txt"))),
    token = read_lines(file.path(corpuspath,subcorpuspath, "corpus2/token", 
                                 paste0(corpusname[2], ".txt"))) %>% gsub(" ([,.!?;:')])","\\1",.) %>% gsub("\\( ", "(", .)
  ) %>% 
    filter(nchar(token) %in% 70:200 ) %>% # filter overly long ones
    filter(grepl(paste0(truth$target,collapse="|"), lemma)) %>% 
    mutate(token=token %>% gsub(" ([,.!?;:')])","\\1",.) %>% 
             gsub("\\( ", "(", .) %>% gsub(' "$',"", .) %>% 
             gsub("do n't", "don't",., ignore.case=T) %>% 
             gsub("@@[0-9]*","",.)
    ) %>% 
    filter(!grepl(" e lass[, ]", token)) # ocr error
  
  print("Data loaded")
  
  # generate test samples
  samp = lapply(truth$target, function(x){
    list(corp1=corp1[grep(x, corp1$lemma),] %>% pull(token) ,
         corp2=corp2[grep(x, corp2$lemma),] %>% pull(token)
    )
  })
  names(samp)=truth$target
  rm(corp1, corp2); gc(verbose=F)
  
  # generate test set
  set.seed(2)
  testsamp = lapply(1:length(samp), function(x){
    tibble(
      target=truth$target[x],
      corp1=sample(samp[[x]]$corp1, nsamp, repl),
      corp2=sample(samp[[x]]$corp2, nsamp, repl)
    )
  }) %>% bind_rows() %>% 
    mutate(pr = paste0('Lexical meaning of "', gsub("_.*","", target), '" in these two sentences: ignoring minor connotations and modifiers, do they refer to roughly the Same, different but closely Related, distant/figuratively Linked (incl metaphors idioms) or unrelated Distinct objects or concepts?\n1. ',
                       corp1, "\n2. ", corp2))
  nrow(testsamp)
  # testsamp$pr[180] %>% cat
  # 'Does the word "', gsub("_.*","", target), '" in these two sentences occur in a Similar or distantly Related or unrelated Distinct sense?\n1. ',
  # mutate(pr = paste0('How close are the meanings of "', gsub("_.*","", target), '" in these two sentences, on a 1-4 scale from unrelated Distinct to distantly Related to Similar to Same meaning?\n1. ',
  # mutate(pr = paste0('How close would the definitions of "', gsub("_.*","", target), '" in these two sentences be? On a scale from unrelated Distinct to somewhat Related to Similar to Same definition.\n1. ',
  # checktokens(testsamp$pr)$length %>% sum
  # 'Does the word "word" in these two sentences occur roughly in the Same or Related or unrelated Distinct sense?
  
  bias = checktokens(c("Same", "Related", "Linked", "Distinct"))
  semres = singletester(test=testsamp, gptmodel="gpt-4", max_attempts=7, initial_delay=10, dfactor=2, backoff="Same", tagonly=T, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") # %>% select(-pr) 
  
  attr(semres, "truth")=truth
  return(semres)
}




dosemranks = function(dat, truth,nmax=30){
  set.seed(1)
  dat = dat %>% 
    mutate(resnum = case_match(result, "Same"~1, "Related"~2, "Linked"~3, "Distinct"~4))
  bootfun = function(dat, indices, npairs, truth){
    dat %>% 
      group_by(target) %>% 
      sample_n(npairs, replace=T) %>%  # won't use boot indices, since need groupwise
      summarize(tgraded=mean(resnum, na.rm=T),
                tvote = sum(result %in% c("Linked", "Distinct"))/n(),
                tbinary = as.integer(sum(result=="Distinct") >=2), 
                tbinary2 = as.integer(sum(result %in% c("Linked", "Distinct")) >=2) 
      ) %>% 
      mutate( tgraderank = rank(tgraded, ties.method='average'),
              tvoterank =  rank(tvote, ties.method='average')) %>% 
      left_join(truth, by="target") %>% 
      mutate(binary = tbinary==as.integer(binary) ,
             binary2 = tbinary2==as.integer(binary) ) %>% 
      summarize(graded = cor(tgraderank, rank, method="spearman"),
                graded2 = cor(tvoterank, rank, method="spearman"),
                binary = sum(binary)/n() ,
                binary2 = sum(binary2)/n()
      ) %>% as.matrix 
  }
  
  iters = tibble()
  for(i in 2:nmax){
    if(i<nmax){
      x=boot(dat, bootfun, R = 100, npairs=i, truth=truth)
      x2 = x$t %>% colMeans()
      names(x2)=colnames(x$t0)
      iters = bind_rows(iters, x2)
    } else {
      iters = bind_rows(
        iters,
        dat %>% 
          group_by(target) %>% 
          summarize(tgraded=mean(resnum, na.rm=T),
                    tvote = sum(result %in% c("Linked", "Distinct"))/n(),
                    tbinary = as.integer(sum(result=="Distinct") >=2), 
                    tbinary2 = as.integer(sum(result %in% c("Linked", "Distinct")) >=2) 
          ) %>% 
          mutate( tgraderank = rank(tgraded, ties.method='average'),
                  tvoterank = rank(tvote, ties.method='average')) %>% 
          left_join(truth, by="target") %>% 
          mutate(binary = tbinary==as.integer(binary) ,
                 binary2 = tbinary2==as.integer(binary) ) %>% 
          summarize(graded = cor(tgraderank, rank, method="spearman"),
                    graded2 = cor(tvoterank, rank, method="spearman"),
                    binary = sum(binary)/n() ,
                    binary2 = sum(binary2)/n()
          )
        
      )
      # engres %>% 
      #   ggplot(aes(tgraded, graded))+
      #   geom_point()+
      #   geom_text(aes(label=target), hjust=0, size=2.8)
    }
    
  }
  iters=iters %>% mutate(npairs=2:nmax)
  return(iters)
  
} 




booktester = function(booklist, prompt, nb=25, n=10, gptmodel="gpt-3.5-turbo", bias,labs, minmax=c(200,300)){
  booklist = booklist[sample(1:length(booklist), nb)] # subsample to keep it cheap
  res = vector("list", length(booklist))
  for(i in seq_along(booklist)){
    x = read_lines(booklist[i]) %>%
      .[sapply(., nchar)>20]  # exclude line breaks and short strings
    if(sum(sapply(x, nchar)>=minmax[1])<n) { # if not enough, merge shorts 
      x = x %>% mergeShortStrings(minmax[1])
    }
    if(sum(sapply(x, nchar)<=minmax[2])<n) { # if not enough, split
      long = sapply(x, nchar)>=minmax[2]
      x[long] = substr(x[long], 1, minmax[2]) %>% 
        gsub(" [^$]+$| $", "",.) # remove split word or trailing space
    }
    x = x[sapply(x, nchar) %in% (minmax[1]:minmax[2])] # sample pool
    if(length(x)<n) {
      res[[i]]=NA  # not enough examples; but after merge should not happen
    }
    x = x[sample(1:length(x), n)]
    xres = sapply(x, function(y){
      singletester(test=paste0(prompt,y), gptmodel="gpt-3.5-turbo", max_attempts=7, initial_delay=10, dfactor=2, backoff="Other", tagonly=T, max_tokens_int=2L, logit_bias = bias, temp = 0, labels = labs, verbose = T)
    }, USE.NAMES = F
    ) 
    res[[i]] = xres
  }
  # kappa = (Po - Pe) / (1 - Pe)
  res2 = sapply(res, function(x) if(is.na(x[1])){return(NA)}else{ table(x) %>% .[which.max(.)]  %>% names}) # max vote bagging
  # acc = sum(res2==truth, na.rm=T)/length(res2[!is.na(res2)])
  # expe = 1/length(labs)   # here equal classes assumption is fine
  # kappa = (acc - expe) / (1-expe) 
  # attr(kappa, "res")=res
  return(res2)
}


linemerger = function(moviereel, minlen=400){
  while(TRUE) {
    initial_nrow <- nrow(moviereel)
    print(initial_nrow)
    
    moviereel <- moviereel %>%
      mutate(group = if_else(len < minlen, 
                             lag(group, default = group[1]), 
                             group))
    
    moviereel <- moviereel %>%
      group_by(group) %>%
      summarise(text = paste0(text, collapse = " "), 
                len = sum(len), 
                .groups = 'drop')
    
    if(nrow(moviereel) == initial_nrow) {
      break
    }
  }
  return(moviereel)
}

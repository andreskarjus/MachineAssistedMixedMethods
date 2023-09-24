############################### #
# Code to supplement
# Karjus 2023, "Machine-assisted mixed methods: augmenting humanities and 
# social sciences with artificial intelligence"
#
# This code is provided as-is, with no implication of warranty
# or liability. This particular implementation relies on
# using the OpenAI API, which is subject to possible changes.
# The API usage requires setting up an OpenAI account and 
# within that, a billing account.
# The code and prompts here can of course be adapted to
# be used on a local LLM if desired. This requires
# replacing or adapting the singletester() function which
# is used to call the API.
#
# The required packages are in the requirements.R file.
# Sourcing it will install them, as well as the reticulate
# package which is used for Python parts of the code, including
# the OpenAI packages (which are not available in R).
# The already computed results are also made available, as
# RData files that can be loaded below.


setwd("")  # either define a path here to the folder where the repo files are, or adjust all paths below to full paths

# Either define path in the command above or adjust these paths:
source("requirements.R")  # this will install all packages, including reticulate & a python version; run only once

source("mamm_scripts.R")  # load the functions (run after restarting R)


# Set OpenAI API key and load Python packages. This assumes reticulate has been 
# installed along with Python and the two OpenAI packages. 
# You can replicate the analyses below without the OpenAI key by loading the 
# precomputed datasets too. In this case, loading reticulate is also not necessary.
# This is therefore moved from the functions file to here as an optional component:
library(reticulate)  
use_condaenv("mamm")
py_run_string(
'
import os
import openai
import tiktoken
openai.api_key = ""  # key goes here
'
)




#### Case studies ####

# Run the following to reproduce most of the results in the paper. Each section
# starts with the option of loading already annotated data - either do this
# (no API key and billing required), or run the full code, including API calls.
# Note that results may slightly vary: this contribution used the most recent
# model builds available via API, which are updated rather often.
# This replicability issue is acknowledged in the paper (which offers ways)
# to work around it.



#### Newsreels, topic modeling ####

# Precomputed results:
load("data/newsreels/testresults2.RData")
load("data/newsreels/corpus_gpt3.RData")


test = read_csv("data/newsreels/Daily.news.outlines.by.story - test.csv") %>% 
  mutate(OUTLINE=gsub("\\|", " ", OUTLINE) %>% # reserved separator
           gsub("[[:space:]]+", " ", .) %>% 
           gsub("^[[:space:]]|[[:space:]]$", "", .) # trailing if any
  )  %>% 
  mutate(generic3 = 
           case_match(generic2, "Agriculture"~"Farm",    # single-token labels!
                      "Economy"~"Industry", .default=generic2)) 


listen = 
  "Politics = use for political events and messages, presidiums, Communist Party, Komsomol; communism, socialism, Leninism, political leaders and premiers; foreign governments and politics; international relations; but do NOT use for wars or industry news.
Military = military, national defense, wars, battles; but NOT politics.
Science = scientific and industrial progress and construction; space and aviation; technological advancements and innovations.
Social = social issues and lifestyle, education, students and schools, family, health, leisure, arts, culture, religion, ceremonies.
Disasters = disasters, fires, weather warnings; but NOT wars.
Sports = sports events; sports results and scores; athletic performances.
Farm = domestic USSR agriculture, farming, hunting, kolkhozes; but NOT foreign news, NOT economics.
Industry = USSR economy, domestic business, economic trends, economy plans; industry, mining; industry workers, brigades; but NOT politics, NOT international news, NOT agriculture, NOT tourism.
Misc = use this for any other topics, only if no Tag above fits well; or if topic is unclear."


labs = unique(test$generic3)
tokcheck = checktokens(labs)
bias = setNames(rep(100, nrow(tokcheck)), tokcheck$tokens) %>% as.list()
pr = test %>% mutate(text = paste0(
  "Classify this USSR news Text with a topic Tag. ONLY use one of the Tags defined in this list:", 
  "\n",  listen,  "\n\nClassify text: ", OUTLINE) ) # %>% pull(text)


## New test
gpt3single = singletester(test=pr, gptmodel="gpt-3.5-turbo", max_attempts=4, initial_delay=41, dfactor=1.5, backoff="Misc", tagonly=T, column=NULL, max_tokens_int=1L, logit_bias=bias) %>% 
  cbind(test, tag=.)
gpt4single = singletester(test=pr, gptmodel="gpt-4", max_attempts=4, initial_delay=41, dfactor=1.5, backoff="Misc", tagonly=T, column=NULL, max_tokens_int=1L, logit_bias=bias) %>% 
  cbind(test, tag=.)



## Entire corpus

# Unfortunately we cannot yet distribute the entire corpus, so this section won't work fully; load the precomputed results above instead.
#
# outlines = read_csv("data/newsreels/Daily.news.outlines.by.story.csv") %>% 
#   mutate(text = paste0(
#     "Classify this USSR news Text with a topic Tag. ONLY use one of the Tags defined in this list:", 
#     "\n",  listen,  "\n\nClassify text: ", OUTLINE) ) 
# 
# newsreels_gpt3 = singletester(test=outlines, gptmodel="gpt-3.5-turbo", max_attempts=7, initial_delay=10, dfactor=2, backoff="Misc", tagonly=T, column="text", max_tokens_int=1L, logit_bias=bias) %>% 
#   select(-text)


# Bootstrapping example hypothesis testing
reelssocial =
  newsreels_gpt3 %>% 
  filter(YEAR %in% 1974:1989) %>% 
  mutate(topic=case_when(result=="Social"~"Social", T~"Other") %>% as.factor())

reelssocial %>% glm(topic~YEAR, data=., family="binomial") %>% summary # 0.064639 log odds

modelstring = '{.$topic=ifelse(.$result=="Social", "Social", "Other") %>% as.factor();.} %>% glm(topic~YEAR, data=., family="binomial") %>% summary() %>% coef() %>% .[2,,drop=F]' # a bit clunky, but passes the necessary operation as well as model, and extracts coef, all as one string
reelsboot = bootstrapintervals(pred=gpt3single$tag, truth=gpt3single$generic3, 
                               newdata=reelssocial, predvalue="topic", 
                               modelstring=modelstring, nboot=1000, bootcols=NULL, bootci = F)


# Bootstrapping confidence intervals and plotting

ns = newsreels_gpt3 %>% 
  mutate(topic=case_match(result, "Farm"~"Agriculture", "Industry"~"Economy", .default=result)) %>%
  filter(YEAR %in% 1955:1989)
nsp=ns %>% 
  group_by(YEAR) %>% count(topic) %>% 
  group_by(YEAR) %>% 
  mutate(n=n/sum(n)*100) %>% ungroup() 
levs=c("Misc", nsp %>% group_by(topic) %>% summarize(n=sum(n)) %>% arrange(n) %>% pull(topic) %>% setdiff("Misc"))
nsp = nsp %>% mutate(topic=fct_relevel(as.factor(topic), levs))


# model function for bootstrap
socmod = function(dat){
  dat %>% group_by(YEAR,result) %>% count() %>% group_by(YEAR) %>% 
    mutate(n=n/sum(n)*100) %>% ungroup() %>% 
    complete(YEAR, result, fill=list(result=0)) %>% # to avoid empty years, would mess up bootstrap; will create 0% values for years where 0%
    filter(result=="Social") %>% 
    select(YEAR, n) %>% column_to_rownames("YEAR") %>% t 
} # boostrap generates new values, this step aggregates for next confint step
socboot = bootstrapintervals(pred=gpt3single$tag, truth=gpt3single$generic3, 
                             newdata=ns, predvalue="result", 
                             modelstring=socmod, nboot=1000, bootcols=1:34,bootci=F) %>% 
  #as.data.frame() %>% 
  mutate(YEAR=statistic %>% as.numeric) %>% 
  mutate(topic="Social")


ggplot(nsp, aes(y=n, x=YEAR, color=topic))+
  #geom_bar(stat="identity")+
  geom_line(linewidth=0.1, alpha=0.5)+
  geom_point(size=0.8)+
  
  geom_errorbar(aes(ymin=n-value,ymax=n+value, x=YEAR, color=topic), 
                data=
                  socboot %>% left_join(nsp %>% filter(topic=="Social") %>% select(n,YEAR), by="YEAR")
                ,inherit.aes = F, width=0, linewidth=1.5, color="white")+ # shadow effect
  
  geom_point(data=nsp %>% filter(topic=="Social"))+
  geom_errorbar(aes(ymin=n-value,ymax=n+value, x=YEAR, color=topic), 
                data=
                  socboot %>% left_join(nsp %>% filter(topic=="Social") %>% select(n,YEAR), by="YEAR")
                ,inherit.aes = F, width=0.2)+
  geom_line(data=nsp %>% filter(topic=="Social"))+
  geom_text_repel(aes(label=topic, x=YEAR+0.5), nsp %>% filter(YEAR==max(YEAR)), max.overlaps = 100, box.padding = 0, direction = "y", 
                  hjust=0, min.segment.length = 999, size=3)+
  #coord_cartesian(expand = F)+
  scale_x_continuous(expand=expansion(add=0.5), breaks=seq(1955,1985,5), limits = c(1955,1993))+
  coord_cartesian(ylim=c(0,50))+
  scale_y_continuous(expand=expansion(add=0.5))+
  scale_color_manual(values=c("#554e33" %>% darken(0.1),"#e72204","#335367","#CB6200" %>% lighten(0.2),"#3B8EC2","#DFB611","#19152b","gray50") %>% rev, breaks=levs)+
  theme_bw()+
  theme(panel.grid.minor = element_blank(), legend.position = "none",
        axis.title.y=element_text(size=10),
        plot.title = element_text(size=11),
        axis.title.x=element_blank()
  )+
  labs(title="(A) Predicted topics in Soviet newsreels 1955-1989", y="Topic prevalence (%)")



## Explorative: infer a list of topics ##

tmp = read_csv("data/newsreels/Daily.news.outlines.by.story.csv") %>% 
  sample_n(n()) %>% 
  mutate(tokens = checktokens(OUTLINE)$length) %>% 
  mutate(csum=cumsum(tokens))
reelexplore = tmp %>% filter(csum < 15400)
nrow(reelexplore) # 204
reelexplore$OUTLINE %>% tokenize_word() %>% unlist(F,F) %>% length() # 8344
tail(reelexplore$csum)
pr = "You are a cultural historian and Soviet history expert. Analyze this corpus of Soviet era newsreels synopses (one per line, unrelated texts). Come up with a concise set of the most frequent general content categories that could be used to describe the themes and topics in a newsreels dataset like this. Try to balance between being general enough to describe all the synopses with a small set of general topics, being not too detailed, but also covering the most important and common themes. The last topic can be a Misc topic which subsumes all the remaining less frequent topics. Output the topics as a comma-separated list of keywords or short key phrases, in English. Do not expand the keywords or comment, just give a list. The corpus:\n"
pr7 = "You are a cultural historian and Soviet history expert. Analyze this corpus of Soviet era newsreels synopses (one per line, unrelated texts). Come up with a set of the most frequent general content categories that could be used to describe the themes and topics in a newsreels dataset like this. Give me 10 topics, where the last is a Misc topic which subsumes all the remaining less frequent topics. Output the topics as a comma-separated list of keywords or short key phrases, in English. Do not expand the keywords or comment, just give a list. The corpus:\n"

gpt3explore = singletester(test=paste0(pr, paste0(reelexplore$OUTLINE, collapse="\n") ), gptmodel="gpt-3.5-turbo-16k", max_attempts=4, initial_delay=41, dfactor=1.5, backoff="Misc", tagonly=F, column=NULL, max_tokens_int=50L, verbose=T) 
gpt3explore7 = singletester(test=paste0(pr7, paste0(reelexplore$OUTLINE, collapse="\n") ), gptmodel="gpt-3.5-turbo-16k", max_attempts=4, initial_delay=41, dfactor=1.5, backoff="Misc", tagonly=F, column=NULL, max_tokens_int=200L, verbose=T) 

gpt3explore10 = singletester(test=paste0(pr7, paste0(reelexplore$OUTLINE, collapse="\n") ), gptmodel="gpt-3.5-turbo-16k", max_attempts=4, initial_delay=41, dfactor=1.5, backoff="Misc", tagonly=F, column=NULL, max_tokens_int=200L, verbose=T) 





#### Maritime shipwrecks ####

# Precomputed:
load("data/ships.RDATA")


wreckcats = read_csv("data/wreckcats.txt")
# some basic data cleaning; also makes the api calls a bit cheaper.
ships = read_csv("vrakiandmed.csv") %>% 
  left_join(wreckcats, by="SHIPWRECKINGCAUSE") %>% 
  filter(!is.na(SHIPWRECKINGCAUSE)) %>% 
  filter(!grepl("teadmata", SHIPWRECKINGCAUSE, ignore.case=T)) %>% filter(!is.na(Cause)) %>% 
  mutate(TEXTshort = TEXT %>% 
           gsub("^[^-]+-(.*)", "\\1",.) %>% 
           gsub("[0-9]{1,2}\\.[0-9]{1,2}\\.[12][0-9]{3,3}|[12][0-9]{3,3}|\\.a\\.|[A-Z]=[0-9,]{1,5}|aastal|´N|´E| lk\\.|^Ankur|Vt.|kuuldavasti|arvatavasti|nr\\.|cm", " ",.,ignore.case=T) %>%
           gsub("Raj\\. [IVX]{1,3}"," ",.) %>%
           gsub(" [a-zA-Z] ", " ",.) %>% gsub("[ ][a-zA-Z]([[:punct:]])", "\\1",.) %>% 
           gsub("[[:space:]]+", " ",.) %>%
           gsub("[ ]*([,.;:!?])+", "\\1",.) %>%
           gsub("^[[:space:][:punct:]]+|[[:space:][:punct:]]+$", "",.) %>% 
           gsub("[ ]*([,.;:!?])+", "\\1",.)
  ) %>% 
  select(LATITUDE, LONGITUDE, SHIPWRECKINGCAUSE,SHIPWRECKINGY, STORYOFSHIPWRECKING, TEXT, TEXTshort, Cause)

pr = "Detect primary first named cause of demise in this ship history Text as either:
Attack = active warfare, bombing, assault, submarine, torpedo; but NOT target practice.
Mine = mine or mine field named as first cause.
Fault = leak, accident, capsizing, engine fail, abandonment, decommission, intentionally sunk, uputatud; but NOT navigation errors like hitting shallows.
Nav = navigation errors like crashing on shallows or reefs; or getting caught in storms, fog, bad weather; but NOT other accidents and NOT intentionally sunk wrecks. 
\n\nText: "
bias = checktokens(unique(wreckcats$Cause)) 


shipgpt4 = singletester(test=ships %>% mutate(pr = paste0(pr, TEXTshort)), gptmodel="gpt-4", max_attempts=7, initial_delay=10, dfactor=2, backoff="Nav", tagonly=T, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)  # 0.8791    kappa 0.8277 mean f1 0.85. mine class had perfect recall.


cm=confusionMatrix(as.factor(shipgpt4$result), as.factor(shipgpt4$Cause),mode = "prec_recall")
mean(cm$byClass[,"F1"])


# Bootstrap counts
wreckmod = function(dat){
  dat %>% group_by(Cause) %>% count() %>% 
    column_to_rownames("Cause") %>% t 
} # boostrap generates new values, this step aggregates for next confint step
wreckboot = bootstrapintervals(pred=shipgpt4$result, truth=shipgpt4$Cause, 
                               newdata=shipgpt4, predvalue="result", 
                               modelstring=wreckmod, nboot=1000, bootcols=1:4,bootci=F) %>% 
  rename(Cause=statistic) %>% 
  left_join( shipgpt4 %>% count(result) %>% rename(Cause=result) %>% mutate(g="test")) %>% 
  rbind(shipgpt4 %>% group_by(Cause) %>% count() %>% mutate(g="ground",value=NA))


ggplot(wreckboot, aes(Cause, n, fill=Cause, group=g, alpha=g))+
  geom_col(position=position_dodge(width = 0.8), width=0.7, fill="white",alpha=1)+
  geom_col(position=position_dodge(width = 0.8), width=0.7)+
  geom_errorbar(aes(ymin=n-value, ymax=n+value), position=position_dodge(width = 0.8),  
                width=0.1)+
  theme(legend.position = "none")+
  scale_alpha_manual(values = c(1,0.7))+
  scale_fill_viridis_d(option="E", begin=0.2, end=0.8)+
  theme_bw()+
  scale_y_continuous(expand=expansion(add=1))+
  theme(panel.grid.minor = element_blank(), legend.position = "none",
        panel.grid.major.x = element_blank(),
        axis.title.y=element_text(size=10),
        axis.title.x=element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(size=11)
  )+
  labs(title="(B) Shipwreck causes in the Baltic\n(left: ground truth, right: predicted)", y="Count")




#### Interview analysis ####

# Precomp
load("data/interviewstest.RData")


# Synthetic data generation - this was done in the ChatGPT web app using GPT-4.
# Write 100 interview responses that in some way discuss the benefits of doing college groupwork assignments online via video call, zoom, messaging apps etc, either as such, or as opposed to live meetings. Invent various positive reasons. Assume the respondent was asked something about online learning in a semi-structured spoken interview. Write like a US college student would but vary response styles. Each response should be short, one sentence but can be a longer sentence. Make it look like they're transcribed speech. Don't use quotation marks.


set.seed(20)
syn = read_delim("data/interviews_synthetic.txt") %>% 
  group_by(campus, age) %>% 
  mutate(id=paste0(age,campus,sample(1:4, n(), replace=T)) ) %>%
  #mutate(across(c(online,campus, id), as.factor)) %>%  # for glmer
  ungroup()
count(syn, id) %>% arrange(n)  


nulmod = glmer(as.factor(online) ~          + age + (1|id), data=syn, family="binomial",glmerControl(optCtrl=list(maxfun=20000)))
hypmod = glmer(as.factor(online) ~ campus   + age + (1|id), data=syn, family="binomial",glmerControl(optCtrl=list(maxfun=20000)))
anova(nulmod, hypmod)
summary(hypmod)
exp(-1.90374) # 0.15 lower odds for on-campus student to be pos about online
xtabs(~online+ campus, data=syn)
nrow(syn)
table(syn$campus,syn$online)

# Predict attitude using GPT4
intpr = "Does this student express a Positive or Negative aspect of online groupwork? Student said: "
bias = checktokens(c("Positive", "Negative"))

intgpt4 = singletester(test=syn %>% mutate(pr = paste0(intpr, example)), gptmodel="gpt-4", max_attempts=20, initial_delay=3, dfactor=2, backoff="Negative", tagonly=T, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr) 

nrow(intgpt4)
intgpt4 %>% select(-age, -example, -id) %>% 
  gather(key = "variable", value = "value") %>%
  group_by(variable, value) %>%
  summarise(count = n())
confusionMatrix(as.factor(intgpt4$result), as.factor(intgpt4$online))
# perfect 100% accuracy.



# Bootstrap the p-values if needed (right now 100% accuracy so no need really)
modelstring='
  nulmod = glmer(as.factor(online) ~         + age + (1|id), data=., family="binomial", glmerControl(optCtrl=list(maxfun=20000) ))
  hypmod = glmer(as.factor(online) ~ campus  + age + (1|id), data=., family="binomial",glmerControl(optCtrl=list(maxfun=20000) ))
  anova(nulmod, hypmod)[2,]'
syntestres = bootstrapintervals(tmp$pred, tmp$online, newdata=syn, predvalue="online", modelstring=modelstring, nboot=100, bootcols=8)




#### Simulate simpsons paradox and confounding ####
#  (for appendix, same dataset)

set.seed(2)
simpson = tibble(online=c(rep("Positive", 200), rep("Negative", 200))) %>% 
  mutate(school = case_when(online=="Negative" ~ sample(LETTERS[1:3],n(),T,prob=c(100,100,10)),
                            online=="Positive" ~ sample(LETTERS[1:3],n(),T,prob=c(9,110,100))
  )) %>% 
  mutate(age = case_when(school=="A" ~ sample(18:21,n(),T),
                         school=="B" ~ sample(20:24,n(),T),
                         school=="C" ~ sample(24:26,n(),T)))


summary(glm(as.factor(online)~age, data=simpson, family="binomial")) 
# 0.53283    0.05702   9.345   <2e-16 ***
glmer(as.factor(online) ~ age + (1|school), data=simpson, family="binomial",glmerControl(optCtrl=list(maxfun=20000))) %>% summary() 
#  0.09955    0.09282   1.072    0.284

gs = ggplot(simpson, aes(online, age))+
  #geom_violin(fill="transparent", color="gray70")+
  geom_boxplot(color="gray20", linewidth=0.3, width=0.4, fill="white")+
  stat_summary( fun=mean, geom="point", size=3) +
  stat_summary(aes(group=1), fun=mean, geom="line", linewidth=1.4, color="gray20")+
  geom_point(aes(color=school), alpha=0.3, position=position_jitter(0.1))+
  stat_summary(aes(group=school, color=school), fun=mean, geom="point", size=4) +
  stat_summary(aes(group=school, color=school), fun=mean, geom="line", linewidth=2)+
  scale_y_continuous(limits=c(17.5,26.5), expand=c(0,0), breaks=seq(18,26,2))+
  theme_bw()+
  theme(panel.grid = element_blank())+
  labs(subtitle="(A) Simulated example of Simpson's paradox",x="Sentiment towards online learning",
       y="Age")


## confounding variable example
set.seed(2)
confound = tibble(online=c(rep("Positive", 200), rep("Negative", 200))) %>% 
  mutate(age = case_when(online=="Negative" ~ rnorm(n(), 19,2) %>% {.[.<18]=18;.},
                         online=="Positive" ~ rnorm(n(), 25,1.8) %>% {.[.<18]=18;.})   ) %>% 
  mutate(campus = case_when(age>=21 ~ sample(c("On", "Off"), n(), T, c(1,5)),
                            age<21 ~ sample(c("On", "Off"), n(), T, c(3,1))  ))
summary(glm(as.factor(online)~campus, data=confound, family="binomial")) 
# campusOn     -2.0917     0.2357  -8.873  < 2e-16 ***
summary(glm(as.factor(online)~campus+age, data=confound, family="binomial"))
# campusOn     -0.4108     0.5165  -0.795    0.426    
# age           1.5476     0.1855   8.342   <2e-16 ***
# summary(lm(age~as.factor(campus), data=confound)) # not too collinear; R-squared:  0.2618
# car::vif(glm(as.factor(online)~campus+age, data=confound, family="binomial"))

conf2 = confound %>% 
  mutate(campus2=as.numeric(as.factor(campus))+(as.numeric(as.factor(online))*0.2)) %>% 
  mutate(age=0)

gc1 = ggplot(conf2, aes(campus2,age, color=online))+
  geom_point(position=position_jitter(0.1),alpha=0.9)+
  theme_bw()+
  scale_x_continuous(breaks=c(1.25, 2.3), labels=c("Off", "On"), expand=c(0.02,0))+
  theme(panel.grid = element_blank(), 
        axis.text.y = element_blank(),
        axis.ticks  = element_blank()
        #axis.title.y = element_blank()
  )+
  labs(subtitle="(B) Simulated example of confounds", x="Campus",color="Online", y="No control for age")

gc2 = ggplot(confound, aes(campus,age, color=online))+
  geom_point(position=position_jitter(0.2,0),alpha=0.9)+
  #scale_color_viridis_c(end=0.9, direction = -1)+
  scale_y_continuous(limits=c(18,30), expand=c(0.05,0), breaks=seq(18,30,2))+
  scale_x_discrete(expand=c(0.25,0))+
  theme_bw()+
  theme(legend.position = "none",
        panel.grid.major.x = element_blank(), 
        panel.grid.minor.x = element_blank(), 
        panel.grid.minor.y = element_blank()
  )+
  labs(x="Campus", y="Age control variable")



gs + plot_spacer()+ (gc1/gc2) + plot_layout(widths = c(1,0.1,0.7))



#### Networks ####

# Precomp:
load("data/litnetworks.RData")

# load data, basic cleaning
# this is quite messy, better prompt could likely give cleaner results

mis = read_lines("data/miserables.txt") %>% #.[1:20] %>% 
  .[!grepl("VOLUME|\\[END OF", .)] %>% 
  paste(collapse="\n") %>% strsplit( "\nCHAPTER[^\n]+\n") %>% unlist(F,F) %>% 
  sapply(function(y){ y %>% 
      gsub("\n“", " “",.) %>%    # remove dialogue newlines
      gsub("”\n", "” ",.) %>%  # also after; adds context
      strsplit("\n") %>% unlist(F,F) %>% 
      .[grepl("[“”]",.)] %>%    # filter down to quote-including paragraphs
      gsub("(”[^”]{0,50})[^”]*$", "\\1",.) %>% # reduce trailing context window
      gsub("([ ]|”)+[^ ”]*$", "",.) %>%  # trailing word parts or space
      paste(collapse=".\n") %>%   # rejoin
      paste0(".") %>%  # add full stop
      gsub("[.]{2,}", ".",.)
  }, USE.NAMES=F
  )  %>% .[nchar(.)>50] # remove empties
mis = c( # quick hack: some chapters too long, so just splitting into chunks
  mis[nchar(mis)<=11000],
  mis[nchar(mis)>11000] %>% sapply(splitstrings, USE.NAMES=F) %>% unlist(F,F)
)
toks = sapply(mis, function(y){ checktokens(y)$length[1]}, USE.NAMES=F) 

pr = "List all pairs of named characters who directly converse in this Text. Format as TSV, no heading. Solve pronoun references, merge references to the same person into name, standardize names. List each pair ONLY once, do NOT repeat listed pairs. Omit people who are just mentioned but do not converse. ONLY list pairs who directly interact.\nText: "
#checktokens(pr) # 48

networkmis3 = singletester(test=paste0(pr, mis), gptmodel="gpt-3.5-turbo", max_attempts=20, initial_delay=3, dfactor=2, backoff=NULL, tagonly=FALSE, max_tokens_int=200L, logit_bias = NULL, temp = 0, verbose = T) 

misclean = networkmis3 %>% 
  gsub("\\\t[-]* |\\\t+| [-]*\\\t| - |\\\t-\\\t|\\\\t", "\\\t",.) %>% 
  gsub("[ ]*\\(.*|[“”]", "",.) %>% 
  paste(collapse="\n") %>% 
  strsplit("\n") %>% unlist(F,F) %>%   
  # clean some invalid outputs like streets and pronouns (prob prompt could be improved so it wouldn't generate those)
  .[!grepl("characters|Paris|house|street|told|said|replied|answered|unknown|they|fish|Europe|him$|This|That|[oO]ther|mountain|ruins|Church|Street|[wW]omen|couples|[pP]eople|Citizens|voice|[Hh]imself|[hH]erself|All|Service|attic|Jesus|Unnamed|person|companions|material|Door|Nature|thunder|Two|[0-9]{4,4}|None| n-|[lL]adies|^He\\\t|^She\\\t|dream|\\*|imagine|E-screen|The |Authorities|X\\.| and|[iI]magination|[cC]onvent|[bB]arricade|Belgian|Narrator|English|French|Spain|Spanish|German|Rue |,|\\\t[^\\\t]+\\\t|Monsieur\\\t|\\\t[a-z]|^[a-z]" ,., ignore.case = F)] %>% 
  .[nchar(.)<50 & nchar(.)>6] %>% 
  gsub("\\\t", '","',.) %>% 
  gsub("([^,]+,[^,]+)\",.*", '\\1',.) %>% 
  paste0('"',.,'"') %>% read.csv(text=., header=F) %>% 
  filter(!(V1 %in% c(""," ","-", "M.", "He", "She", "Him","His", "Her","Man", "Monseigneur", "Monsieur")),
         !(V2 %in% c(""," ","-", "M.", "He", "She", "Him","His", "Her","Man", "Monseigneur", "Monsieur"))) %>% 
  filter(!grepl("^[ -]*$", V1), !grepl("^[ ]*$", V2)) %>% 
  mutate(across(c(V1, V2), 
                ~case_when(grepl("Valjean|Jean|Fauchelevent|Tranchelevent|Madeleine|Fabre|Urbain|Leblanc|U\\.F\\.",.)~"Jean Valjean",
                           grepl("Javert", .)~"Javert",
                           grepl("Cosette.s mother", .)~"Fantine",
                           grepl("Marius|Pontmercy", .)~"Marius",
                           grepl("Gavroche", .)~"Gavroche",
                           grepl("Welcome|Welcome|Bienvenu|Bishop|Myriel",.)~"Myriel",
                           grepl("Gillenormand", .)~"Gillenormand",
                           grepl("Mabeuf", .)~"Mabeuf",
                           grepl("Thénardier|Jondrette", .)~"Thénardiers",
                           T~.
                ))) %>% 
  rowwise() %>% 
  mutate(pair = paste(sort(c(V1, V2))[1], sort(c(V1, V2))[2], sep = "|" )) %>% 
  group_by(pair) %>% 
  count() %>% 
  arrange(-n) %>% 
  separate(pair, "\\|", into=c("to", "from"), remove=F) %>% 
  filter(to!=from) %>% 
  group_by(to) %>% mutate(t1=n()) %>% 
  group_by(from) %>% mutate(t2=n()) %>% 
  ungroup() %>% 
  mutate(size=t1+t2) %>% select(-t1, -t2)


#### Plot networks 

les = les_miserables[[1]]; colnames(les)=les_miserables$names; rownames(les)=les_miserables$names
les[lower.tri(les, diag=F)] = 0
les = reshape2::melt(les) %>% filter(value>0)
colnames(les)=c("to", "from", "value")
#visNetwork::visNetwork(nodes=data.frame(id=les_miserables$names) %>% mutate(label=id), edges=les)

ng = read_csv("data/misnames.txt") %>% filter(!duplicated(name))

gr =  as_tbl_graph(misclean  , directed = F)
gr = tbl_graph(edges = misclean, nodes=
                 as_tibble(gr, what="vertices") %>% left_join(ng, by="name")) %>% 
  mutate(degree = centrality_degree()) #%>% filter(degree>1)

set.seed(2); g2 = ggraph(gr, layout ="fr")+
  annotate("text", x=-Inf, y=Inf, label="B", size=5, vjust=1.1, hjust=-1.1)+
  geom_edge_arc(aes(width = log(n), color=n), strength = 0.05)+
  geom_node_point(aes(size=degree, color=gender))+
  geom_node_text(aes(filter=degree > 7 | (gender=="Female" & degree > 4) | name %in% c("Cosette", "Fantine"), label=name, color=gender), 
                 hjust=-0.05, vjust=0.9 , alpha=0.8, size=2.7 )+
  scale_edge_width(range=c(0.1,0.7))+
  scale_size_continuous(range=c(0.2,2))+
  scale_edge_color_gradientn(
    colors=c("gray80", "gray70", "gray70","gray65","gray65","gray60"))+
  scale_color_manual(values=c(lighten("#b3632b",0.2),  "#005b96", "gray50"))+ 
  theme_void()+
  theme(legend.position = "none")

gr0 = as_tbl_graph(les %>% select(-value)) %>% mutate(degree = centrality_degree())
set.seed(5); g1=ggraph(gr0, layout ="fr")+
  annotate("text", x=Inf, y=Inf, label="A", size=5, vjust=1.1, hjust=-0.2)+
  geom_edge_arc(color="gray75", width=0.2, strength = 0.03)+
  geom_node_point(aes(size=degree), color="gray40", size=0.9)+
  geom_node_text(aes(filter=degree > 8  | name %in% c("Cosette", "Fantine", "Valjean", "Myriel"), label=name), hjust=-0.05, vjust=0.9, alpha=0.9, size=2.4  )+
  scale_edge_width(range=c(0.1,1))+
  scale_size_continuous(range=c(0.3,4))+
  scale_edge_color_gradientn(
    colors=c("gray80", "gray70", "gray70","gray65","gray65","gray60"))+
  scale_x_reverse()+
  theme_void()+
  theme(legend.position = "none", plot.margin = margin(-1,20,-1,-1))

g1+g2+plot_layout(widths = c(4,6))+plot_annotation(theme=theme(plot.margin = margin(0,0,0,0)))










#### Relevant example extractor & OCR fixer ####

# Precomp:
load("data/ocrtest.RData")

nat = read_tsv("data/sample_reproduced.tsv") %>% 
  filter(!is.na(excerpt))
# This sample has been produced and annotated by Laur Kanger, Peeter Tinits, et al.
# see https://doi.org/10.1016/j.gloenvcha.2021.102447

nrow(nat)
checktokens(nat$excerpt)$length %>% sum  # 13134 / 1000*0.03  $0.39 gpt4 sisend
pr = "Fix OCR errors to restore this English newspaper excerpt; do not comment!\n"

naturegpt4 = singletester(test=nat %>% mutate(pr=paste0(pr, excerpt)), gptmodel="gpt-4", max_attempts=20, initial_delay=2, dfactor=2, backoff=NULL, tagonly=FALSE, max_tokens_int=2000L, logit_bias = NULL, temp = 0, verbose = T, column = "pr") %>% select(-pr) %>% rename(cleaned=result)


## classify
bias = checktokens(c("Yes", "No"))
pr2 = 'Does this text mention nature or environment in the biogical natural world sense, including nature tourism, landscape, agri, environmental policy? Output No if not, or if nature in other senses like natural products, medicine, "human nature" or environment in general non-nature sense.\nText: '

natclas_original3 = singletester(test=naturegpt4 %>% select(`orig-id`, relevant, excerpt) %>% mutate(pr=paste0(pr2, excerpt)), gptmodel="gpt-3.5-turbo", max_attempts=20, initial_delay=2, dfactor=2, backoff=NULL, tagonly=FALSE, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)
natclas_clean3 = singletester(test=naturegpt4 %>% select(`orig-id`, relevant, cleaned) %>% mutate(pr=paste0(pr2, cleaned)) , gptmodel="gpt-3.5-turbo", max_attempts=20, initial_delay=2, dfactor=2, backoff=NULL, tagonly=FALSE, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)
natclas_original4 = singletester(test=naturegpt4 %>% select(`orig-id`, relevant, excerpt) %>% mutate(pr=paste0(pr2, excerpt)), gptmodel="gpt-4", max_attempts=20, initial_delay=2, dfactor=2, backoff=NULL, tagonly=FALSE, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)
natclas_clean4 = singletester(test=naturegpt4 %>% select(`orig-id`, relevant, cleaned) %>% mutate(pr=paste0(pr2, cleaned)), gptmodel="gpt-4", max_attempts=20, initial_delay=2, dfactor=2, backoff=NULL, tagonly=FALSE, max_tokens_int=1L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)


list(natclas_original3,
     natclas_clean3,
     natclas_original4,
     natclas_clean4
) %>% lapply(function(x){
  y=confusionMatrix( (x$result=="Yes") %>% as.factor(), 
                     x$relevant %>% as.factor())
  c(y$overall["Accuracy"], y$overall["Kappa"]) %>% t %>% as.data.frame()
}) %>% do.call(rbind,.)
#  Accuracy     Kappa
# 1 0.7878788 0.4945295 natclas_original3,
# 2 0.8181818 0.5630211 natclas_clean3,
# 3 0.8989899 0.7728316 natclas_original4,
# 4 0.9191919 0.8211382 natclas_clean4

# cost
sum(natclas_original4$tokensin/1000*0.03)+sum(natclas_original4$tokensout/1000*0.06)
sum(natclas_original3$tokensin/1000*0.002)+sum(natclas_original3$tokensout/1000*0.002)




#### Text reuse, fake news, pseudohistories ####

# scenarios: clean reuse, rephrased, broken OCR, translated

# Generate synthetic test set (see paper for references)
targets = c('always make use of this exact phrase: "Russians are descendants of the Huns" (without quotes). Create examples based on these different ideas but always weave this phrase in, not as a quote but as naturally as possible.', 
            'always make some mention Huns or Scythians or Attila or Hunnic culture or language in some way, even if in passing. BUT DO NOT say that Russians could be descendants of the Huns because Russian people are completely unrelated to Huns. Do not even hint that they could be related.')
pr0 = paste0('Write 50 paragraphs drawing from the following fictional alternative timeline ideas. 3 sentences per paragraph. Write some in the style of blogs, some like Reddit, some like forum posts. Assume some paragraphs would be either in the beginning, middle or end of the longer post. All 50 are meant to argue for any combination of these ideas below, but importantly, whatever the topic, ',targets, ' Your sample of fictional historical ideas to use: 
Huns originate from the area of present-day Russia.
The West has been fighting an information war for 100+ years against the Russian people by brainwashing. 
Russian education system helps the West brainwash Russians. 
Historical facts (invent some!) have been hidden by those forces.
Russians had an ancient continuous statehood.
Russians had state structures well before Rurik.
Mainstream history considers Rurik as the founder of Russian statehood. 
The first capital of ancient Russia was Slovensk, which ruled a vast territory. 
It was located at the same place as Novgorod and had already been established by 2409 BCE 
Slovensk was founded 3099 years after the creation of the world. 
Orthodox religiousness as a natural part of Russian heritage.
Rus or ancient Russia was called "the Country of Cities" in the old days.
Russians produced all the weapons used in seventeenth-century Europe.
This massive weapons production and export provide proof of a sophisticated civilization.
Now write the 50 paragraphs, single newline separated, DO NOT comment, DO NOT tag the paragraphs, DO NOT number them, also do not forget to mention Huns.')
# pr[1] %>% cat
# This generation step was done in the web version of ChatGPT, using GPT-4.

#prrephrase = 'Rewrite this Text so instead of "Russians are descendants of the Huns" the same idea of genetic relatedness between Huns and Russians with Huns being their ancestors is expressed in different words. Text:\n'  # this can be done with regex actually
procr = "Make this text look like it has few OCR errors by splitting couple of words with space and replacing few letters with similar letters.\n"
prrus = "Translate into Russian:\n"
prtest = "Does this Text mention that Russians are descendants of the Huns or from Hun genetic lineage? Answer Yes if any such ancestry claim. Answer No if no such claim or only talk of cultural or areal links. Text:\n"

# Some easier generation tasks are done with GPT3.5 to optimize cost
# the ocr error filter prompt in cyrillic text mostly replaces cyrillic with similar latin letters, so it doesn't immediately "look" broken, but it is (completely different letters & underlying unicode), and would easily mess up simpler language models and classifiers.

huns = read_lines("data/huns.txt") %>% 
  data.frame(text=.) %>% 
  mutate(true = c(rep("Yes", 50), rep("No", 50))) %>% 
  #slice(1) %>% 
  mutate(test_rephrase= c(
    gsub("Russians are descendants of the Huns", "Huns are ancestors of Russians", text[1:10]),
    gsub("Russians are descendants of the Huns", "Russians carry the genetic lineage of the Huns", text[11:20]),
    gsub("Russians are descendants of the Huns", "genealogical roots of Russians link them to the Hunnic tribes", text[21:30]),
    gsub("Russians are descendants of the Huns", "Russian genetic lineage goes back to the Huns", text[31:40]),
    gsub("Russians are descendants of the Huns", "the bloodline of the Huns flows in the veins of Russians", text[41:50]),
    text[51:100]
  ) ) %>% 
  mutate(test_rephrase=gsub("(,|are not) descendants of the Huns", "\\1 of the genetic lineage of the Huns", test_rephrase)) %>% 
  mutate(test_ocr= singletester(test=paste0(procr, text),  gptmodel = "gpt-4-0613", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)) %>% 
  mutate(test_rus= singletester(test=paste0(prrus, text),  gptmodel = "gpt-3.5-turbo", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)) %>% 
  mutate(test_rus_ocr= singletester(test=paste0(procr, test_rus),  gptmodel = "gpt-4-0613", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)) %>% 
  mutate(test_rephrase_ocr= singletester(test=paste0(procr, test_rephrase),  gptmodel = "gpt-4-0613", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)) %>% 
  mutate(test_rephrase_rus= singletester(test=paste0(prrus, test_rephrase),  gptmodel = "gpt-3.5-turbo", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)) %>% 
  mutate(test_rephrase_rus_ocr= singletester(test=paste0(procr, test_rephrase_rus),  gptmodel = "gpt-4-0613", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T))


# Test classifier:

hunstest = huns %>% 
  mutate(result_text= singletester(test=paste0(prtest, text),  gptmodel = "gpt-4-0613", max_tokens_int=300L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100))) %>% 
  mutate(result_rephrase= singletester(test=paste0(prtest, test_rephrase[1:50]),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100)) %>% c(., results_text[51:100]) ) %>% # only need to test rephrased part here, other half is same as above  
  mutate(result_ocr= singletester(test=paste0(prtest, test_ocr),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100))) %>% 
  mutate(result_rus= singletester(test=paste0(prtest, test_rus),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100))) %>% 
  mutate(result_rus_ocr= singletester(test=paste0(prtest, test_rus_ocr),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100))) %>% 
  mutate(result_rephrase_ocr= singletester(test=paste0(prtest, test_rephrase_ocr),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100))) %>% 
  mutate(result_rephrase_rus= singletester(test=paste0(prtest, test_rephrase_rus),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100))) %>% 
  mutate(result_rephrase_rus_ocr= singletester(test=paste0(prtest, test_rephrase_rus_ocr),  gptmodel = "gpt-4-0613", max_tokens_int=1L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=T, temp=0, verbose = T, logit_bias = list("9642"=100, "2822"=100)))


hunstest %>% select(true, starts_with("result_")) %>% 
  summarize(across(starts_with("result_"), ~sum(.==true))) %>% t()
hunstest %>% select(text, true, result_text) %>% filter(true=="Yes",result_text=="No")




#### Linguistic usage feature analysis ####

# Precomputed:
load("data/lingmodifiers.RData")

# To replicate:
# Download the original paper data from: https://doi.org/10.5281/zenodo.7252695
# then place the premodification.csv in the data folder.

premod = read_csv("data/premodification.csv")
premod %>% summarize(across(everything(), ~length(unique(.))))
nrow(premod) # 168417
premod %>% mutate(test=paste(normalized, `head-word`)) %>% pull(test) %>% unique %>% length #  8129

# if batch input, starts hallucinating output by line 247
# modpr = "These are modifier-noun pairs from 18th century English texts, a pair per line. The pairs are unrelated to each other. Output the pairs but append a comma-separated value to each indicating if the modifier is Description or Evaluation. Use only either of these tags. Description means objective like size or color or origin or type or material like wooden. Evaluation means subjective like judging beauty or elegance or value like fashionable or antique or old. Do NOT comment.\nThe pairs:\n"

modpr = "These are modifier-noun phrases from 18th century English texts, a phrase per line. The phrases are unrelated to each other. Output the phrases but append a comma separated value to each indicating if the modifier is Objective or Subjective.\nObjective tag is for any modifiers describing objective properties such as shape or color or origin (like Persian carpet) or type or material (like wooden chair) and in general attributes that can be defined and measured. Objective also includes fixed compounds like looking glass or eating stall.\nSubjective tag is for any evaluative modifiers referring to beauty or elegance or usefulness or quality or value or fashion or convenience or rarity. Subjective also includes subjective age (like old car, new house) and subjective size (like spacious house, large garden) and possible users (like men's shirt or family house). Subjective in general is qualitative attributes that cannot be objectively defined.\nOnly guess either Subjective or Objective. The phrase:\n"

set.seed(5)
modres = premod %>% 
  mutate(test=paste(normalized, `head-word`)) %>% 
  filter(!duplicated(test)) %>% 
  group_by(mod) %>%  
  sample_n(200) %>% 
  ungroup() %>% 
  sample_n(400)
head(modres)
# modres %>% select(test) %>% write_csv("test.csv")

chunks = (1:nrow(modres)) %>%  split(., ceiling(seq_along(.)/50))
tmpres = list()
for(i in seq_along(chunks)){
  tmpres[[i]] = modres[chunks[[i]],] %>% pull(test) %>%  # checktokens() # ~1280 total
    paste0(collapse="\n") %>% 
    paste0(modpr, .) %>% 
    singletester(test=.,  gptmodel = "gpt-4-0613", max_tokens_int=1000L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T) %>% 
    strsplit( "\n") %>% unlist(F,F) %>% strsplit(split=",[ ]*") %>% 
    lapply(function(x) data.frame(tmp=x[1], res=x[2])) %>% bind_rows()
}
tmpres=do.call(rbind,tmpres)

modtested = cbind(modres, tmpres) %>% 
  #mutate(res = case_when(is.na(res)~gsub("[^,]+,(.*)", "\\1",tmp),T~res)) %>% 
  #mutate(tmp=gsub(",.*", "",tmp)) %>%
  mutate(res=case_match(res, "Objective"~"D", "Subjective"~"E")) %>% 
  mutate(matched=test==tmp)
sum(modtested$matched) # match ok

confusionMatrix(as.factor(modtested$res), as.factor(modtested$mod)) # acc 0.9425kappa 0.885 
# confusionmat in paper
# data.frame(D=c(167,17), E=c(4, 94)) %>% {rownames(.)=c("D", "E");.} %>% as.matrix %>% as.table %>% confusionMatrix()

# Attempt to use code interpreter within web-chatgpt; does not work well, attempts to create a keyword based classifier in Python no matter what.
# cipr='This csv file contains a single test column, which is a list of 400 modifier-noun phrases from 18th century English texts, one per line. They all have this format: "modifier noun" e.g. "big house". The phrases are unrelated to each other. Your task: add a second column indicating if the modifier is Objective or Subjective. DO NOT use Python for this, DO NOT create a classifier, DO NOT attempt to create an if-else keywords rules based classifier! Just import the file but then treat the file as regular input, and just intelligently guess the function of the modifier given the phrase meaning, then output the news contents (the phrases and the tags).\nObjective tag is used for any objective modifiers referring to objective properties like shape or color or origin (e.g. "Persian carpet") or type or material (e.g. "wooden chair"), or intended users (e.g. "mens clothes"), generally attributes that can be defined and/or measured. Objective also includes fixed phrases like "looking glass" or "eating stall".\nSubjective tag is used for any subjective modifiers including those referring to beauty or elegance or value or subjective evaluation (e.g. "old car", "spacious hourse", "fashionable clothes"). Subjective generally include qualitative attributes that cannot be objectively defined.\nDo not mark anything as Unknown, always guess either Subjective or Objective, and do NOT comment, do not print the first few lines, just do the task amd print the 400 lines of output.'





#### Lexical change detection ####

# To replicate, download the Semeval task data and place it in the data folder
# Note that the original data contains several gzipped files that need to be
# unpacked first within the folder structure, so do that first.

corpuspath=file.path(getwd(), "data", "lexchange")

eng4 = dolexchange(
  corpuspath=corpuspath
  ,subcorpuspath="semeval2020_ulscd_eng"
  ,corpusname=c("ccoha1", "coha2")
  ,nsamp=30
)
engres = dosemranks(eng4, attr(eng4, "truth"),nmax=30)

ger4 = dolexchange(
  corpuspath=corpuspath
  ,subcorpuspath="semeval2020_ulscd_ger"
  ,corpusname=c("dta", "bznd")
  ,nsamp=30
  ,repl=T
)
gerres = dosemranks(ger4, attr(ger4, "truth"),nmax=30)

lat4 = dolexchange(
  corpuspath=corpuspath
  ,subcorpuspath="semeval2020_ulscd_lat"
  ,corpusname=c("LatinISE1", "LatinISE2")
  ,nsamp=30
  ,repl=T
)
latres = dosemranks(lat4, attr(lat4, "truth"),nmax=30)

langs = rbind(
  engres %>% mutate(Language="Am.English (19th, 20th c)"),
  gerres %>% mutate(Language="German (19th, 20th c)"),
  latres %>% mutate(Language="Latin (BCE, CE)")
)

# semeval bests, graded
# eng   ger   lat
# 0.422 0.725 0.462

# graded in rosin_temporal_2022
# eng   ger   lat
# 0.520 0.763 0.565 

sotas = data.frame(
  Language=c("Am.English (19th, 20th c)", "German (19th, 20th c)", "Latin (BCE, CE)"),
  gsemeval=c(0.422 ,0.725 ,0.462),
  grosin = c(0.520, 0.763, 0.565 ),
  brandom = c(0.568, 0.646, 0.350),
  bsemeval= c(0.703, 0.750, 0.7)
)
th = 
  theme_bw()+
  theme(panel.grid.minor.x = element_blank(), 
        panel.grid.minor.y = element_blank(), 
        panel.grid.major.y = element_blank(), 
        legend.position = "none",
        axis.title.y=element_text(size=10),
        axis.title.x=element_text(size=10),
        plot.title = element_text(size=11)
  )
#cols = c("#012169", "#ffcc00" %>% darken(0.4), "#d51c1f" %>% darken(0.4))
cols=viridis_pal(begin=0.7, end=0.8, option="E")(3)


ggplot(langs,aes(x=npairs, y=binary, color=Language))+
  geom_hline(data=sotas, aes(yintercept = bsemeval), linewidth=0.5)+
  geom_hline(data=sotas, aes(yintercept = brandom), linewidth=0.5, color="gray")+
  geom_line(linewidth=0.9)+
  scale_y_continuous(expand=c(0,0), limits=c(0,1))+
  scale_x_continuous(expand=c(0,0))+
  scale_color_manual(values=cols)+
  facet_wrap(~Language)+
  labs(x="Number of example pairs", y="Accuracy", subtitle="(A) Binary semantic change classification", title="Lexical semantic change detection using GPT4")+
  geom_text(aes(x=x,y=y,label=l), data.frame(Language="Latin (BCE, CE)", x=29, y=0.96,l="SemEval2020\nbest model"), hjust=1, vjust=1, size=2.7,color="black")+
  geom_segment(aes(x = 17, xend = 15, y = 0.8, yend = 0.71), data.frame(Language="Latin (BCE, CE)"), 
               arrow = arrow(type = "closed", length = unit(0.03, "inches")), color="black")+
  th

ggplot(langs,aes(x=npairs, y=graded, color=Language))+
  geom_hline(data=sotas, aes(yintercept = gsemeval), linewidth=0.5)+
  geom_hline(data=sotas, aes(yintercept = grosin), linetype="dotted", linewidth=0.5)+
  geom_line(linewidth=0.9)+
  scale_y_continuous(expand=c(0,0), limits=c(0,1))+
  scale_x_continuous(expand=c(0,0))+
  scale_color_manual(values=cols)+
  facet_wrap(~Language)+
  labs(x="Number of example pairs", y="Spearman rank correlation",subtitle="(B) Ranked semantic change detection")+
  geom_text(aes(x=x,y=y,label=l), data.frame(Language="Am.English (19th, 20th c)", x=3, y=0.3,l="SemEval2020\nbest model"), hjust=0, vjust=1, size=2.7,color="black")+
  geom_segment(aes(x = 10, xend = 12, y = 0.33, yend = 0.41), 
               data.frame(Language="Am.English (19th, 20th c)"), 
               arrow = arrow(type = "closed", length = unit(0.03, "inches")), color="black")+
  geom_text(aes(x=x,y=y,label=l), 
            data.frame(Language="Latin (BCE, CE)", x=29, y=0.59,l="(Rosin et al. 2022)"), hjust=1, vjust=0, size=2.7,color="black")+
  th





#### Twitter data, meaning divergence annotation ####

# Precomp
load("data/stimpairs_gpt4.RData")

stimpairs = read_csv("stimpairs8.csv")
labs = as.character(1:4)
tokcheck = checktokens(labs)
bias = setNames(rep(100, nrow(tokcheck)), tokcheck$tokens) %>% as.list()
pr = stimpairs %>% mutate(text=paste0(
  "Rate the semantic similarity of the target word in <x> tags in these sentences A and B. Consider only the objects/concepts the word forms refer to: ignore any common etymology and metaphorical similarity! Ignore case! Ignore number (cat/Cats = identical meaning). If target is emoji then rate by its contextual function. Homonyms (like bat the animal vs bat in baseball) count as unrelated. Output numeric rating: 1 is unrelated; 2 is distantly related; 3 is closely related; 4 is identical meaning.\nA: ", sentence, "\nB: ", sentence2)
)
stimpairs_gpt4 = singletester(test=pr, gptmodel="gpt-4", max_attempts=7, initial_delay=10, dfactor=2, backoff="10", tagonly=T, column="text", max_tokens_int=1L, logit_bias=bias) %>% select(-text)

# Human eval
# inter-rater agreement (spearman, not kappa, as it's ordinal not categorical)
cor(stims_rated$Rater1, stims_rated$Rater2, method = "spearman")    # 0.8736345 humans
cor(stims_rated$Rater1, stims_rated$RaterGPT4, method = "spearman") # 0.4537251
cor(stims_rated$Rater2, stims_rated$RaterGPT4, method = "spearman") # 0.6096363




#### Stance detection ####

# Precomp:
load("data/testid.RData")

testset = read_csv("data/reltestset_rated.csv")

pr=c('Assess if this Text mentions Estonian police or border guard or PPA. Yes = news is from Estonia or about Estonian police or border. If location not hinted assume Estonia. No = news is about another country or police of another country, or context is fictional or metaphorical.\nNote this news is titled "', '".\nText: ')
rbias = checktokens(c("Yes", "No"))

reltest3 = singletester(test=testset %>% mutate(pr = paste0(pr[1], Title, pr[2], Text)), gptmodel="gpt-3.5-turbo", max_attempts=7, initial_delay=10, dfactor=2, backoff="Yes", tagonly=T, max_tokens_int=1L, logit_bias = attr(rbias, "bias"), temp = 0, labels = rbias$labels, verbose = T, column="pr") #%>% select(-pr)
reltest4 = singletester(test=testset %>% mutate(pr = paste0(pr[1], Title, pr[2], Text)), gptmodel="gpt-4", max_attempts=7, initial_delay=10, dfactor=2, backoff="Yes", tagonly=T, max_tokens_int=1L, logit_bias = attr(rbias, "bias"), temp = 0, labels = rbias$labels, verbose = T, column="pr") #%>% select(-pr)

epr=c('Does this Text portray Estonian police or border guard or PPA in Neutral or Positive or Negative light?\nText: ')
sbias = checktokens(c("Positive", "Negative", "Neutral"))

evaltest3 = singletester(test=testset %>% filter(!is.na(Stance)) %>% mutate(pr = paste0(epr, Text)), gptmodel="gpt-3.5-turbo", max_attempts=7, initial_delay=10, dfactor=2, backoff="Yes", tagonly=T, max_tokens_int=1L, logit_bias = attr(sbias, "bias"), temp = 0, labels = sbias$labels, verbose = T, column="pr") #%>% select(-pr)
evaltest4 = singletester(test=testset %>% filter(!is.na(Stance)) %>% mutate(pr = paste0(epr, Text)), gptmodel="gpt-4", max_attempts=7, initial_delay=10, dfactor=2, backoff="Yes", tagonly=T, max_tokens_int=1L, logit_bias = attr(sbias, "bias"), temp = 0, labels = sbias$labels, verbose = T, column="pr") #%>% select(-pr)




#### Computational thematics ####

# To replicate this section, contact the authors of the original study for the data.

promptlong = "Classify genre of this fiction Text:
Detective = investigator characters like detective, police officer, private eye; investigates crime, criminals, corruption, murder and violence; pursues clues and leads, interviews suspects; typically real-world or noir setting, not fantasy or scifi.
Romance = modern or historical characters, deals with emotional intimacy, conflict and tension, affection, longing, heartbreak, breakup, love story, romantic or sexual themes and relationships; real-world setting, not fantasy or scifi.
Fantasy = characters like wizards sorcerers knights rogues, mythical or mystic creatures, supernatural magical events, magic powers; pre-industrial, feudal or medieval settings and themes.
Sci-fi: futuristic characters and tech, aliens, space stations, spaceships, lasers; science and tech focus; setting in advanced future, outer space, alien worlds or dystopian societies.\n\nText: "
promptshort = "Classify genre of this fiction text: "

# This won't work with the original data
books = lapply(c(
  "data/literarygenres/1 Detective utf-8"
  ,"data/literarygenres/2 Fantasy utf-8"
  ,"data/literarygenres/3 Sci-Fi utf-8"
  ,"data/literarygenres/4 Romance utf-8"), list.files, full.names=T)
names(books)=c("Detective", "Fantasy", "Sci-Fi", "Romance")

labs = names(books)
bias = checktokens(labs) %>% pull(tokens) %>% strsplit(" ") %>% unlist %>% setNames(rep(100, length(.)), .) %>% as.list()

test10 = lapply(labs, function(x){booktester(books[[x]], prompt=promptshort,nb=25, n=10, gptmodel="gpt-3.5-turbo", bias=bias, labs=labs, minmax=c(200,300)) }) # 0.8, 0.73 kappa
test10long = lapply(labs, function(x){booktester(books[[x]], prompt=promptlong,nb=25, n=10, gptmodel="gpt-3.5-turbo", bias=bias, labs=labs, minmax=c(200,300)) }) # apparently worse, acc 0.61, kappa 0.48

confusionMatrix(as.factor(unlist(test10)), as.factor(rep(labs, each=25)), mode="everything") #  0.7333  
confusionMatrix(as.factor(unlist(test10long)), as.factor(rep(labs, each=25)), mode="everything")  # 0.48 definitions seem to mess things up at least for gpt3.5; might be better with gpt4



## Book and movie script example ##

# Precomputed:
load("data/reelstest.RData")

# 21k words
moviereel = read_lines("data/bladerunner.txt") %>% 
  tibble(text=.) %>% 
  mutate(len=nchar(text)) %>% 
  mutate(group = row_number()) %>% 
  linemerger(200)
moviereel %>% filter(len>300, len<400) %>% pull(text)
hist(moviereel$len, 100)
nrow(moviereel)
median(moviereel$len)
# 62400 words
bookreel = read_lines("data/doandroidsdream.txt") %>% 
  paste(collapse=" ") %>% 
  strsplit("[ ]+") %>% unlist(F,F)
nr=nchar(bookreel)
cumulative_sum <- cumsum(nr)
chunk_limits <- seq(0, max(cumulative_sum), by = 1000)
chunk_indices <- findInterval(cumulative_sum, chunk_limits)
chunked_bookreel <- split(bookreel, chunk_indices) %>% 
  lapply(paste, collapse=" ")
length(chunked_bookreel %>% unlist())
bookreel = tibble(text=chunked_bookreel %>% unlist(F,F))

bookpr = "Classify genre of this fiction Text as either Detective, Fantasy, Sci-Fi, Romance, Thriller, or Other if none of these match. Text:\n"
filmpr = "Classify genre of this movie Script section as either Detective, Fantasy, Sci-Fi, Romance, Thriller, or Other if none of these match. Script:\n"
bias = checktokens(c("Detective", "Fantasy", "Sci-Fi", "Romance", "Thriller", "Other"))

moviereel3 = singletester(moviereel %>% mutate(pr=paste0(filmpr, text)),gptmodel="gpt-3.5-turbo", max_attempts=7, initial_delay=10, dfactor=2, backoff="Other", tagonly=T, max_tokens_int=2L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)

bookreel3 = singletester(bookreel %>% mutate(pr=paste0(bookpr, text)), gptmodel="gpt-3.5-turbo", max_attempts=7, initial_delay=10, dfactor=2, backoff="Other", tagonly=T, max_tokens_int=2L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr)


scr=image_read(list.files("data/litreels/screens",full.names = T)) %>% image_resize("x200")
h=image_info(scr[1])$height
reel=image_crop(scr, geometry = paste0( "x", h*0.78), gravity = "Center") %>% 
  image_border("gray20", "1x0") %>% 
  image_append() %>%  image_border("gray20", "144x0")
# %>% image_write("data/litreels/reel.png")

cols=c( lighten("#895A0C"), "#C6CFBF" ,  "#13777D", lighten("#042942",0.2), lighten("#833D3D",0.3), "#1B0E0A")
ggplot(bookreel3 %>% mutate(nr=row_number()) %>% 
         mutate(result=fct_relevel(as.factor(result), 
                                   c("Detective", "Fantasy", "Sci-Fi", "Thriller","Romance", "Other"))), 
       aes(nr, xmin=nr-0.505,xmax=nr+0.505, ymin=0,ymax=1, fill=result))+
  geom_rect(color=NA)+
  scale_fill_manual(values=cols )+
  theme_void()+
  theme(  
    legend.text = element_text(color="gray98"),
    legend.title  = element_text(color="gray98"),
    legend.margin = margin(0,5,0,-20),
    axis.title.y=element_text(color="gray98"),
    plot.margin = margin(0,0,0,0))+
  labs(y="Do Androids\nDream of\nElectric\nSheep?", fill="Estimated\ngenre")+
  theme(axis.title.y=element_text(size=9, angle=0, hjust=1, margin=margin(0,-20,0,0)))+
  
  ggplot(moviereel3 %>% mutate(nr=row_number()) %>% 
           mutate(result=fct_relevel(as.factor(result), 
                                     c("Detective", "Sci-Fi", "Thriller","Romance", "Other"))), 
         aes(nr, xmin=nr-0.505,xmax=nr+0.505, ymin=0,ymax=1, fill=result))+
  geom_rect(color=NA)+
  scale_fill_manual(values=cols[-2] )+
  theme_void()+
  theme(
    axis.title.y=element_text(color="gray98"),
    legend.position = "none",
    plot.margin = margin(0,0,0,0))+
  labs(y="Blade\nRunner")+
  theme(axis.title.y=element_text(size=9, angle=0, hjust=1, margin=margin(0,-35,0,0)))+
  image_ggplot(reel)+
  plot_layout(heights=c(1,1,0.6), guides = "collect")+
  plot_annotation(theme=theme(plot.margin=margin(0,2,0,2), 
                              plot.background = element_rect(fill="gray20", color=NA)))








#### Literature alignment and comparison ####

# Precomp:
load("data/translationtest.RData")

litpr = "Determine if there are any significant lexical or stylistic differences between this English text and its Italian translation. If not then output None. If any then explain but only very briefly, do NOT comment otherwise."
eng_it = read_csv("data/result_it.csv") %>% 
  mutate(pr=paste0(litpr, "\nEnglish: ", src, "\nItalian: ", tgt))
taggedres = singletester(eng_it, column="pr", gptmodel = "gpt-4-0613",  max_tokens_int = 100L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)


## Synthetic test set ##

# Precomp:
load("data/translation_editdist.RData")

rab = read_csv("data/parallellit/rabbit.txt") %>% 
  mutate(change = rep(c("Close", "Substitution","Deletion", "Addition"), each=25))
pr = "Compare this EN source and JA translation. Ignore minor stylistic differences ignore word order. Focus on objects and subjects and if they differ between EN and JA. Output either:\nClose translation.\nAddition = JA contains additional subject object not in EN version.\nDeletion = something or somebody missing from JA compared to EN.\nSubstitution = only applies if the translation is close but an object or subject in EN has been removed and replaced with something new in JA.\nTexts:\n"
#bias=checktokens(c("Close", "Replaced","Added","Removed"))
bias=checktokens(c("Close", "Substitution","Addition","Deletion"))
translation_editdist = singletester(test=rab %>% mutate(pr = paste0(pr, English, "\n", Japanese)), gptmodel="gpt-4", max_attempts=7, initial_delay=1, dfactor=1.5, backoff="Close", tagonly=T, max_tokens_int=2L, logit_bias = attr(bias, "bias"), temp = 0, labels = bias$labels, verbose = T, column="pr") %>% select(-pr) 

accuracy(translation_editdist, "change", "result")
confusionMatrix(as.factor(translation_editdist$result), as.factor(translation_editdist$change))
sum(translation_editdist$result!="Close")

# Accuracy  0.9600000
# Kappa     0.9466667
# F1        0.9592080
# Precision 0.9635989
# Recall    0.9600000

# For some reason here the latest gpt version seems to give slighly inconsistent/variable results even with temp=0.




#### Lexicography ####

# Precomp
load("data/lexicographytest.RData")

lexsent = expand_grid(
  pr = "Write 20 short unrelated sentences which always mention or use this common noun: zoorplick. It's a new word that means ",
  word = c("glue, so write about glueing and using glue to repair or craft", "bear, so write about bears in nature", "thief, so write about thieves stealing or robbing"),
  sampgens = c("factual newspaper genre, ", "fiction genre, ") # 
  #,sampinte = c("in first or third person", "in third person")
  ,samplang = c(".", ", only write grammatical Estonian sentences and in Estonian context.", ", only write grammatical Turkish sentences and in Turkish context.")
) %>% mutate(id=1:n()) %>% 
  mutate(init= paste(pr, word, ". Sentence per line so 20 lines, no numbering no comments. Write in ", sampgens, sep="") %>%  paste0(., "in first or third person", samplang)) %>% 
  mutate(samplang = case_when(grepl("Turkish", samplang)~ "Turkish",
                              grepl("Estonian", samplang)~ "Estonian",
                              T~ "English")) %>% 
  mutate(word = case_when(grepl("glue", word)~ "glue",
                          grepl("bear", word)~ "bear",
                          grepl("thief", word)~ "thief")) %>% 
  select(-pr) %>% 
  singletester(., column="init", gptmodel = "gpt-4-0613",  max_tokens_int = 1000L, max_attempts=10, initial_delay=10, dfactor=1.5, tagonly=F, temp=0, verbose = T)

lexsent = lexsent %>% separate_rows(result,sep =  "\n+") %>% 
  mutate(result=gsub("^[ ]+|[ ]+$", "", result)) %>%  # trailing spaces
  filter( grepl("zoorplick", result, ignore.case=T)) %>%
  rename(example=result) %>% 
  select(!starts_with("token"))

lextests = list()
lexsent2 = lexsent %>% 
  # use that one as eval:
  mutate(testfirst = case_when(word=="glue" ~ "glue|adhesive",
                               word=="bear" ~ "bear|grizzly",
                               word =="thief" ~ "thief|thieves|burglar|shoplifter|pickpocket|robber")) %>% 
  # second looser criterion, but that's too easy, won't use
  mutate(testtwo = case_when(word=="glue" ~ "glue|adhesive|silicone|epoxy",
                             word=="bear" ~ "bear|grizzly|panda|predator",
                             word =="thief" ~ "thief|thieves|burglar|shoplifter|pickpocket|robber|criminal|bandit|mobster|gangster|felon|outlaw|lawbreaker|miscreant|perpetrator"))
# not accepting criminal and predator
for(i in 1:10){
  tmplist=list()
  for(j in 1:10){
    tmp = lexsent2 %>% group_by(samplang, word) %>% sample_n(i) %>% 
      group_by(samplang, word) %>% 
      summarize(testfirst=testfirst[1],testtwo=testtwo[1], example=paste0(example, collapse="\n"), .groups = "drop") %>% 
      mutate(ni=i, pr= paste0('Act as lexicographer skilled in finding and inferring meanings of new words. Based on these Examples, figure out the meaning of "zoorplick", a new noun in ', samplang ,'. Do not comment, just output most likely meaning as a specific single word in English. Examples:\n', example))
    res = singletester(tmp$pr, column=NULL, gptmodel = "gpt-4-0613",  max_tokens_int = 4L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)
    tmplist[[j]] = tmp %>% mutate(guess=res) %>% 
      rowwise() %>% 
      mutate(correct=grepl(testfirst, guess, ignore.case=T)) %>% 
      mutate(correct2=grepl(testtwo, guess, ignore.case=T)) %>% 
      ungroup()
  }
  lextests[[i]] = do.call(rbind, tmplist)
  if(all(lextests[[i]]$correct)){
    print("Early stopping, all correct")
    break
  }
}
lexdone=do.call(rbind, lextests) 

lexsum = lexdone %>% group_by(samplang, word, ni) %>% 
  summarize(correct=(sum(correct, na.rm=T)/n()),
            correct2=(sum(correct2, na.rm=T)/n())) %>% 
  ungroup() %>% 
  mutate(samplang=as.factor(samplang)) %>% 
  mutate(samplang=fct_relevel(samplang, "English", "Turkish", "Estonian")) %>% 
  mutate(samplang=recode(samplang, "English"="English (Indo-European, ~372M)", "Estonian"="Estonian (Uralic, ~1.1M)", "Turkish"="Turkish (Turkic, ~82M)"))


ggplot(lexsum, aes(ni, correct2*100, color=word, fill=word))+
  geom_line(position=position_dodge(width=0.35))+
  geom_point(position=position_dodge(width=0.35), shape=21)+
  facet_grid(rows =  ~samplang)+
  scale_x_continuous(breaks=1:10)+
  scale_y_continuous(expand=expansion(add=3))+
  scale_color_manual(values=c("#b3632b", "gray40", "#005b96"))+
  scale_fill_manual(values=c("#b3632b", "gray40", "white"))+
  theme_bw()+
  labs(color="Sense", fill="Sense", x="Number of example sentences provided to zero-shot lexicographer", y="Mean sense identification accuracy")+
  theme(panel.grid.minor.x = element_blank(),
        axis.title.x=element_text(hjust=0),
        axis.title.y=element_text(size=10)
  )



#### YLE data augmentation ####

# Unfortunately this data cannot be shared at this time, until clarity is reached with the data provider. The code here is provided for reference only.

# subset for gpt test
set.seed(1)
ygpt = ydaily %>% 
  filter(!is.na(series_title_fin) | !is.na(title_fin), 
         !is.na(desc_fin) | !is.na(series_desc_fin), 
         contenttype!="kids, animation", !rapo_Lapset, 
         #!is.na(series_id) | !is.na(series_title_fin),
         !is.na(English), !is.na(countrygroup)) %>%
  filter( !duplicated(series_title_fin) | is.na(series_title_fin)) %>% 
  filter( !duplicated(title_fin) | is.na(title_fin)) %>% 
  filter(!grepl("olympialaiset| MM|Tour|maraton|naisten|miesten|løpet|sport|Oddasat", title_fin, ignore.case=T),
         !grepl("olympialaiset| MM|Tour|maraton|naisten|miesten|løpet|sport|Oddasat", series_title_fin, ignore.case=T),
         English!="Finland") %>% 
  select(series_title_fin, title_fin, series_desc_fin, desc_fin, country, English) %>%  group_by(English) %>%
  filter(n()>=100) %>% sample_n(100) %>% 
  ungroup() %>% 
  sample_n(200) %>% 
  mutate(English=case_match(English, "USA"~"USA, United States", .default = English))
nrow(ygpt)
ygpt %>% count(English)

yleres = ygpt %>% mutate(pr=paste0('Where is this film or series from? Title: ', series_title_fin, ". ", title_fin, '. Description: ', series_desc_fin," ", desc_fin, '\nThe info given here is in Finnish because it was shown on YLE but it is not made in Finland. Consider the translated title and synopsis and any actors directors mentioned and your knowledge of TV and guess the production country. NO comments even if unclear, output single country name.') %>% # series or films, metadata depends, so clearing the empty slots:
                           gsub("NA|NA\\.", "", .) %>% 
                           gsub(" \\.", "",.) %>% 
                           gsub("[ ]+", " ",.)
)  %>% singletester( column="pr", gptmodel = "gpt-4-0613",  max_tokens_int = 3L, max_attempts=10, initial_delay=1, dfactor=1.5, tagonly=F, temp=0, verbose = T)

# ground truth still has messy labels, fixing to be able to compare
yleres = yleres %>% 
  mutate(English = case_match(English,
                              "USA, United States"~"United States", 
                              "Commonwealth of Australia"~"Australia",
                              "Germany (since 1990)" ~ "Germany",
                              "Russian Federation"~"Russia",
                              .default=English))
table(yleres$English)
length(unique(yleres$English)) # 15
nrow(yleres)

lvs = unique(c(yleres$English,yleres$result)) %>% sort
yleres %>% 
  mutate(English=factor(English, levels=lvs),
         result=factor(result, levels=lvs)) %>% 
  accuracy(truth="English")
# Accuracy  0.7150000
# Kappa     0.6946647
table(yleres$English, yleres$result )
yleres %>% filter(result!=English) %>% group_by(English, result) %>% count %>% arrange(-n) %>% as.data.frame()
# Mixes up UK-US-Canada, Austria-Germany- Switzerland, and Spain-Argentina, Sweden-Norway. Also guesses UK too often.
yleres %>% filter(result!=English) %>% select(desc_fin)




#### Visual analytics using multimodal models ####

# There is no programmatic component in this section, which consists of prompting the Bing Chat AI visual interface. The prompts and responses are listed here for reference.

# Explain this joke. Be very brief and concise, to the point.
# The joke is based on a wordplay that involves removing the letter B from the word “bananas” and getting the word “ananas”, which is the French word for “pineapple”. The joke implies that bananas are just pineapples with Bs added to them. This is a humorous and absurd statement that contradicts the obvious differences between the two fruits.

# This is a still from a movie. Analyze it like a film scholar. Focus on what is clearly visible in the image. Do not attempt to guess or imagine external details, focus just on the image. Do not search the internet for more info, just describe the scene.
# The image depicts a scene of violence and conflict, as indicated by the presence of guns, masks, and broken glass. The scene is likely part of an action or thriller genre, which typically features such elements to create tension and excitement for the audience. The scene also suggests a power imbalance between the four men, as three of them are shooting at one, who is falling backwards. The falling man is in a vulnerable and defeated position, while the other three are in a dominant and aggressive stance. This could imply that the falling man is either the protagonist or an ally of the protagonist, who is facing a threat or a challenge from the antagonists. The scene is shot from a low angle, which enhances the sense of drama and danger. The low angle also makes the three shooters appear more powerful and intimidating, as they loom over the falling man and the viewer. The falling man is in the center of the frame, which draws attention to his plight and suffering. The image is slightly blurry, which suggests fast movement and adds to the feeling of chaos and urgency. The setting of the scene is a gray stone building with columns, which could indicate a public or official place, such as a government office, a bank, or a museum. This could imply that the scene is part of a heist, a robbery, or a political conspiracy, which are common plot devices in action or thriller movies. The contrast between the dark clothing and masks of the men and the light-colored stone of the building creates a visual effect of opposition and intrusion, as if the men are out of place and disrupting the order of the place. The debris and broken glass scattered around the scene add to the sense of destruction and disorder caused by the shootout.

# Make a nested list of all clearly visible and prominent objects and characters in this painting. Only include items you that you are very certain are in the image. Format the list without line breaks, like this: [painting [house [windows]]]
# [painting [sky [clouds]] [buildings [roofs] [walls] [windows] [doors]] [street [cobblestones]] [people [man in black suit and top hat] [woman in blue dress and white hat] [boy in brown shirt and pants] [girl in pink dress and bonnet]] [animals [horse] [cart] [chickens] [rabbits]]]

# Analyze this painting like an art historian expert art scholar. Discuss style and composition as well as possible interpretations. Focus only on the image itself, do not attempt to look up additional information. Write a coherent story, avoid bullet points. 
# This painting is an example of abstract expressionism, a style of art that emerged in the 1940s and 1950s in the United States. Abstract expressionism is characterized by spontaneous, gestural, and emotional expression of the artist’s inner feelings and moods. The painting does not depict any recognizable objects or scenes, but rather uses colors, shapes, and textures to create a visual impact. The painting has a dynamic and energetic composition, with a burst of colors that radiate from the center. The colors are mostly warm (orange, yellow, red) and cool (blue, green, purple), creating a contrast and a sense of tension. The black and white colors add depth and contrast to the painting, as well as suggesting light and shadow. The painting has a thick and textured surface, with visible brushstrokes that show the movement and force of the artist’s hand. The painting is square in shape, which gives it a sense of balance and stability. The possible interpretations of this painting are varied and subjective, depending on the viewer’s perspective and imagination. Some possible interpretations are:
# The painting represents an explosion or a firework, symbolizing celebration, excitement, or destruction.
# The painting represents a flower or a sun, symbolizing life, growth, or energy.
# The painting represents a galaxy or a star, symbolizing space, mystery, or infinity.
# The painting represents the artist’s psyche or emotions, symbolizing creativity, passion, or turmoil.
# The painting invites the viewer to explore their own feelings and associations with the colors and shapes, and to appreciate the beauty and power of abstract art.





#### Automated linguist ###

# This section is not complete, and more comprehensive study is required. It might be added at a later time.

# create novel language
# lingtest = read_lines("data/lingtest.txt") %>% 
#   sample() %>% 
#   chartr("aeioubcdfghjklmnpqrstvwxyzAEIOUBCDFGHJKLMNPQRSTVWXYZ", "eiouacdfghjklmnpqrstvwxyzbEIOUACDFGHJKLMNPQRSTVWXYZB",.) %>% 
#   paste(collapse="\n")
# 
# pr="These are some examples from an undocumented language. Analyze it carefully like a linguistics professor, try to understand where the verbs and nouns are, try to see patterns. Then for the output, infer what is the dominant word order in this language. Possible options are:\nSubject-object-verb (SOV)\nSubject-verb-object (SVO)\nVerb-subject-object (VSO)\nVerb-object-subject (VOS)\nObject-verb-subject (OVS)\nObject-subject-verb (OSV).\nProvide your reasoning before deciding on a particular option.\nThe text:\n"























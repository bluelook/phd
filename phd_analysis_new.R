#install.packages(c("ggplot2", "tidyverse", "gridExtra", "purrr","plotly"))

#ggthemes, "extrafont", "cowplot", "grid", ggforce, shiny, ggridges, ggrepel, reshape2
#devtools::install_github("thomasp85/patchwork")
library(ggplot2)
library(tidyverse)
library(purrr)
library(gridExtra)
library(gtools)
library(plot3D)
library(readxl)


rm(list = ls())
dev.off() #clear all
#set working directory to the data files location
setwd('./data/behavioral')
# list the files in a directory, each file represents a subject
file_names <- list.files(pattern = ".csv$")
num_subjects <- length(file_names)
subj_data <- read_xlsx("responses.xlsx", sheet="all traversed")

#read all the files in the directory into one data frame
full_df <- do.call(smartbind, lapply(file_names, read.csv, header=TRUE,stringsAsFactors=FALSE))
#set working directory back to the source file location
setwd('../..')
model_subj_data <- read_csv("./output/optimizationResultAll.csv")#, check.names = FALSE)
bic_stress <- read_csv("./output/BICStress.csv") #, check.names = FALSE)
bic_meal <- read_csv("./output/BICMeal.csv")
bic_room <- read_csv("./output/BICRoom.csv")

bic_stress$subject_nr <- factor(bic_stress$subject_nr)
bic_meal$subject_nr <- factor(bic_meal$subject_nr)
bic_room$subject_nr <- factor(bic_room$subject_nr)

ggplot(data = bic_stress,aes(x = reorder(subject_nr, -deltaBICInd2AB_Ind1AB), y = deltaBICInd2AB_Ind1AB,fill=subject_nr) )+
  geom_bar(stat = "identity") +
  labs(title = "stress delta BIC (ind 2ab vs 1 ab) per subject", x="subjects")

ggplot(data = bic_meal,aes(x = reorder(subject_nr, -deltaBICInd2AB_Ind1AB), y = deltaBICInd2AB_Ind1AB,fill=subject_nr) )+
  geom_bar(stat = "identity") +
  labs(title = "meals delta BIC (ind 2ab vs 1 ab) per subject", x="subjects")

ggplot(data = bic_room,aes(x = reorder(subject_nr, -deltaBICInd2AB_Ind1AB), y = deltaBICInd2AB_Ind1AB,fill=subject_nr) )+
  geom_bar(stat = "identity") +
  labs(title = "rooms delta BIC (ind 2ab vs 1 ab) per subject", x="subjects")
#setwd('./questionaires')
#qcae_df <- t(read.csv('qcae.csv', header = FALSE, stringsAsFactors=FALSE))
#colnames(qcae_df) <- qcae_df[1, ]
#rownames(qcae_df) <- c()
#qcae_df = qcae_df[-1,]

#keep only the needed data and add numeric data about choices(dist,sweet,wood = 1. reap, salty, blue = 0)
extracted_df <-select(full_df,subject_nr,actor,correct,preferred)

#for each row convert choices to numeric values
for (i in 1:nrow(extracted_df)){
  #if the choice taken was incorrect
  if(extracted_df$correct[i] == 0){
    if (extracted_df$preferred[i] == 'dist'){
      extracted_df$chosen[i]<-'reap'
      extracted_df$chosen_numeric[i]<- 0
      extracted_df$preferred_numeric[i] <- 1
    }
    else if (extracted_df$preferred[i] == 'reap'){
      extracted_df$chosen[i]<-'dist'
      extracted_df$chosen_numeric[i]<-1
      extracted_df$preferred_numeric[i] <- 0
    }
    else if (extracted_df$preferred[i] == 'sweet'){
      extracted_df$chosen[i]<-'salty'
      extracted_df$chosen_numeric[i]<-0
      extracted_df$preferred_numeric[i] <- 1
    }
    else if (extracted_df$preferred[i] == 'salty'){
      extracted_df$chosen[i]<-'sweet'
      extracted_df$chosen_numeric[i]<-1
      extracted_df$preferred_numeric[i] <- 1
    }
    else if (extracted_df$preferred[i] == 'blue_closet'){
      extracted_df$chosen[i]<-'wood_closet'
      extracted_df$chosen_numeric[i]<-1
      extracted_df$preferred_numeric[i] <- 0
    }
    else if (extracted_df$preferred[i] == 'wood_closet'){
      extracted_df$chosen[i]<-'blue_closet'
      extracted_df$chosen_numeric[i]<-0
      extracted_df$preferred_numeric[i] <- 1
    }
  }else{
    #the step taken was correct
    extracted_df$chosen[i] = extracted_df$preferred[i]
    if(extracted_df$preferred[i] == 'reap' | extracted_df$preferred[i] == 'salty' |  extracted_df$preferred[i] == 'blue_closet'){
      extracted_df$chosen_numeric[i]<-0
      extracted_df$preferred_numeric[i] <- 0
    }
    else{ 
      extracted_df$chosen_numeric[i]<-1
      extracted_df$preferred_numeric[i] <- 1
    }
  }
}

#descriptive statistics
#summary(extracted_df)

# actor_trials <- subject_choices %>% filter(actor == subject_choices$actor[1])
# num_trials <- length(actor_trials$subject_nr)

theme_set(theme_bw())

#mean by actor between subjects with corrected mean for "0" coded actor
mean_chosen_all <-
  extracted_df %>% group_by(actor) %>%
  summarise(mean = mean(chosen_numeric), sem = sd(chosen_numeric*100) / sqrt(num_subjects)) %>%
  mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2")) %>%
  mutate(mean = if_else(color == "Condition 1", 1 - mean, mean),
         mean_percent = round(mean * 100, digits = 2) )

#plot the mean by actor
ggplot(data = mean_chosen_all,
       aes(x = actor, y = mean_percent, fill = color)) +
  geom_bar(position = position_dodge(), stat = "identity") +
  geom_errorbar(aes(ymin = mean_percent - sem, ymax = mean_percent + sem),
                width = .2)+
  scale_x_discrete(name = "Strategies") +
  scale_y_continuous(name = "Mean Frequency  ") +
  labs(title = "Frequency of choosing preferred strategy - between subjects") +
  geom_text(aes(label =  paste(mean_percent, "%")), position = position_dodge(width = 1),
            vjust = -0.5, size = 2)+
  scale_fill_discrete(name="Preferred\nStrategy", labels=c("Strategy 1", "Strategy 2"))
ggsave(path="output", filename="mean_all_strategy_freq.png", width = 5, height = 5)

#mean by actor within subjects
mean_chosen_by_subject_orig <-
                      extracted_df %>% group_by(subject_nr, actor) %>%
                      summarise(mean = mean(chosen_numeric)) %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2"))
#change to the complementary percentage for the strategy that is coded as '0'
mean_chosen_by_subject <- mean_chosen_by_subject_orig %>%
                      mutate(mean = if_else(color == "Condition 1", 1 - mean, mean), mean_percent =mean*100)
#mean of two actors per condition
mean_by_condition <-
  mean_chosen_by_subject %>% mutate(
    actor = if_else(
      grepl('m', actor),
      "m",
      if_else(grepl('s', actor), "s",
              if_else(grepl('r',  actor), "r", "")))) %>% group_by(subject_nr,actor) %>% 
      summarise(mean = mean(mean_percent)) %>% mutate(dist_from_chance = mean-50)

subjects <- unique(extracted_df$subject_nr)
par(mfrow=c(1,2))
#layout(matrix(c(1,1), 1, 2))
for (subj in subjects)
{
  subject_choices <- mean_by_condition %>% filter(subject_nr == subj)
  s = subject_choices$mean[subject_choices$actor=='s'] #stress is x axis
  m = subject_choices$mean[subject_choices$actor=='m'] #meals is y axis
  r = subject_choices$mean[subject_choices$actor=='r'] #rooms is z axis
  x <- c(s, 0, 0)
  y <- c(0, m, 0)
  z <- c(0, 0, r)
  line_xy <- rbind(x,y)
  line_yz <- rbind(y,z)
  line_xz <- rbind(x,z)
  scatter3D(c(0,0,0), c(0,0,0), c(0,0,0), type = "b", axis.scales = FALSE,bty = "u", col.grid = "lightblue",
            xlim = c(0,100), ylim = c(0,100), zlim = c(0,100), 
            xlab = "Stress",ylab ="Meals", zlab = "Rooms", col = "red", 
            theta = 120, phi = 17, ticktype="detailed", nticks = 6, main=paste("Subject ", subj))
  scatter3D(line_xy[,1], line_xy[,2], line_xy[,3], type = "b", axis.scales = FALSE, col = "red", 
            colkey=FALSE,add=TRUE)
  scatter3D(line_xz[,1], line_xz[,2], line_xz[,3], type = "b", col = "darkgreen", colkey=FALSE, add=TRUE)
  scatter3D(line_yz[,1], line_yz[,2], line_yz[,3], type = "b", col = "blue", colkey=FALSE, add=TRUE)
}

#comparison of distance from 50
#target <- c("s", "r")
#mean_by_condition_s_r <- filter(mean_by_condition,actor %in% target)
mean_by_cond_reshaped <-pivot_wider(mean_by_condition %>% select(dist_from_chance,subject_nr,actor), names_from="actor", values_from="dist_from_chance")
subj_data_merged <- merge(mean_by_cond_reshaped,subj_data,by="subject_nr", all.x = TRUE)
subj_data_merged <- merge(subj_data_merged,model_subj_data,by="subject_nr")
#subj_data_merged <- merge(subj_data_merged,bic_data,by="subject_nr")
subj_data_merged_filtered <- subj_data_merged[subj_data_merged$BIC_act_dep < 60 & subj_data_merged$BIC_act_ind < 60 & subj_data_merged$BIC_act_dep_2A < 60 & subj_data_merged$BIC_act_ind_2A < 60,]


#t test to compare BICs, see if significant
t_bic_act_dep_ind = t.test(subj_data_merged_filtered$BIC_act_dep, subj_data_merged_filtered$BIC_act_ind)
t_bic_act_dep_no_act = t.test(subj_data_merged_filtered$BIC_act_dep, subj_data_merged_filtered$BIC_no_act)
t_bic_act_ind_no_act = t.test(subj_data_merged_filtered$BIC_act_ind, subj_data_merged_filtered$BIC_no_act)

#make the columns to be rows to benefit from pipes
bic_data_gathered <- gather(subj_data_merged_filtered %>% select(BIC_act_dep, BIC_no_act, BIC_act_ind), key = 'model', value ='bic',factor_key=TRUE)


# Model Comparison - calculate mean of bic and sem per each model
mean_bic <-
  bic_data_gathered %>% group_by(model) %>% summarise(mean = mean(bic), sem = sd(bic) / sqrt(num_subjects))
my_comparisons <- list( c("BIC_act_dep", "BIC_no_act"), c("BIC_no_act", "BIC_act_ind"), c("BIC_act_dep", "BIC_act_ind") )
compare_means(bic ~ model,  data = bic_data_gathered, method = "anova")

ggplot(bic_data_gathered, aes(x = model, y = bic, fill = model)) +
  geom_point(shape = 21, size = 10)+
  #geom_errorbar(aes(ymin = bic - sem, ymax = bic + sem), width = 0.2) +
  labs(title = paste("Models Comparison p dep vs ind =", format(t_bic_act_dep_ind$p.value, digits = 2), ", p dep vs no act = ", format(t_bic_act_dep_no_act$p.value, digits = 2))) + 
  scale_x_discrete(name="Model")+
  scale_y_continuous(name = "BIC")+
  theme(legend.title = element_blank())+ stat_compare_means(comparisons = my_comparisons)+#  Add pairwise comparisons p-value
  stat_compare_means(label.y = 85)

######playing with stats#####
t_result = t.test(subj_data_merged_filtered$sAlpha_no_act,subj_data_merged_filtered$sAlpha_act_ind)
print(t_result)

#simple linear regression,"YVAR ~ XVAR" where YVAR is the dependent, or predicted,XVAR is the independent, or predictor
lm1 <- lm(sAlpha_act_ind~CE,data=subj_data_merged_filtered)
summary(lm1)
lm2 <- lm(s~sAlpha_act_ind,data=subj_data_merged_filtered)
summary(lm2)
lm3 <- lm(s~sAlpha_act_dep,data=subj_data_merged_filtered)
summary(lm3)
lm4 <- lm(s~sAlpha_no_act,data=subj_data_merged_filtered)
summary(lm4)

glm_s <- glm(s ~ sAlpha_act_ind,data=subj_data_merged_filtered,family = gaussian(link="identity"))
summary(glm_s)

ggplot(subj_data_merged_filtered,aes(y = s,x = sAlpha_act_dep) )+ 
  geom_point()+
  stat_smooth(method="glm", se=F, method.args = list(family=gaussian()))+
  ylab('stress mean value dist from chance') +
  xlab('stress learning rate')

learning_rates <- select(subj_data_merged_filtered,subject_nr, sAlpha_act_ind, sAlpha_act_dep, sAlpha_no_act)
learning_rates_longer <- pivot_longer(learning_rates, sAlpha_act_ind:sAlpha_no_act,names_to = "learning_rate", values_to = "value")
learning_rates_mean <- learning_rates_longer %>% group_by(learning_rate) %>% summarise(mean = mean(value), sem=sd(value)/sqrt(12))
                                                                
ggplot(data = learning_rates_mean,
         aes(x = learning_rate, y = mean, fill = learning_rate)) +
    geom_bar(position = position_dodge(), stat = "identity") +
    scale_x_discrete(name = "learning rate") +
    scale_y_continuous(name = "mean")+
  geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                width = .2) 

# Compute the analysis of variance
res.aov <- aov(value~learning_rate, data = learning_rates_longer)
# Summary of the analysis
summary(res.aov)

# Box plots
# ++++++++++++++++++++
# Plot weight by group and color by group
library("ggpubr")
ggboxplot(mean_by_condition, x = "actor", y = "mean", 
          color = "actor", palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          order = c("s", "m", "r"),
          ylab = "mean", xlab = "condition")
##########################

#for each subject, calculate frequencies of choices and plot them
for (subj_num_local in subjects) {
  subj_choices <-
    extracted_df %>% filter(subject_nr == subj_num_local)
  
  #calculate % of 1s (dist, sweet, wood choices)
  mean_chosen_individual <- subj_choices %>% group_by(subject_nr,actor) %>%
    summarise(mean_chosen = mean(chosen_numeric),
              mean_percent_chosen = round(mean_chosen * 100, digits=2))
  
  mean_chosen_individual <- mean_chosen_individual %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2")) 
  #plot the complementary percentage for the strategy that is coded as '0'
  mean_chosen_individual <- mean_chosen_individual %>% mutate(mean_percent_chosen = if_else(color == "Condition 1", 100-mean_percent_chosen, mean_percent_chosen))
  ggplot(data = subset(mean_chosen_individual, subject_nr == subj_num_local),
         aes(x = actor, y = mean_percent_chosen, fill = color)) +
    geom_bar(position = position_dodge(), stat = "identity") +
    scale_x_discrete(name = "Strategies") +
    scale_y_continuous(name = "Frequency of chosing each strategy") +
    labs(title = paste("Subject ", subj_num_local)) +
    geom_text(aes(label =  paste(mean_percent_chosen, "%")), position = position_dodge(width = 1),
              vjust = -0.5, size = 2)+
    scale_fill_discrete(name="Preferred\nStrategy", 
                        labels=c("Strategy 1", "Strategy 2"))
  
  ggsave(path="output", filename = paste("subj",subj_num_local, "_strategy2_freq.png"),
    width = 5,
    height = 5)
}

#res <-t.test(needed_df[needed_df$actor=='s2',]$chosen_numeric, needed_df[needed_df$actor=='m2',]$chosen_numeric,var.equal = TRUE)
#res <-t.test(data[data$actor=='s2',]$chosen_numeric, data[data$actor=='m2',]$chosen_numeric,var.equal = TRUE)
#print(substring(csv,1,9), res)
  
#create plot for choices over trials per subject, 1=dist/sweet/wood, 0=reap/salty/blue, human actors
# graphics.off()
#par(mar=c(1,1,1,1))
par(mar=c(3,4,3,1))

par(mfrow = c(length(files_names)/2+1,2))
p <- list()
q <- list()
n <- list()

steps_s2 <- list()
steps_s1 <- list()
steps_m1 <- list()
steps_m2 <- list()
steps_r1 <- list()
steps_r2 <- list()

i=1
j=1

xlabel = "Trial number"
ylabels = "Distraction choices"
ylabelm = "Sweet choices"
ylabelr = "Wood cabinet choices"
for (subj_num_local in subjects){
  local({
    df <-
      extracted_df %>% filter(subject_nr == subj_num_local)
    ys2=df[df$actor=='s2',"chosen_numeric"]
    ys1=df[df$actor=='s1',"chosen_numeric"]
    ym1=df[df$actor=='m1',"chosen_numeric"]
    ym2=df[df$actor=='m2',"chosen_numeric"]
    yr1=df[df$actor=='r1',"chosen_numeric"]
    yr2=df[df$actor=='r2',"chosen_numeric"]

    steps_s2[[j]] <<- ys2
    steps_s1[[j]] <<- ys1
    steps_m1[[j]] <<- ym1
    steps_m2[[j]] <<- ym2
    steps_r1[[j]] <<- yr1
    steps_r2[[j]] <<- yr2

    two_plotS <-qplot(xlab=xlabel, ylab=ylabels, main=paste(subj_num_local,'- stress')) +
      geom_line(aes(x=1:length(ys1), y=ys1, color="s1")) +
      geom_line(aes(x=1:length(ys2),y=ys2, color = "s2")) +
      scale_y_continuous(breaks = round(seq(0, 1, by = 0.5),1)) +
      scale_x_continuous(breaks = round(seq(0, 20, by = 1),1))+
      theme(legend.position = "right")+
      labs(colour="Actors")
    p[[i]] <<- two_plotS
    
    two_plotM <-qplot(xlab=xlabel, ylab=ylabelm, main=paste(subj_num_local, '- meals')) +
      geom_line(aes(x=1:length(ym1), y=ym1, color="m1")) +
      geom_line(aes(x=1:length(ym2),y=ym2, color = "m2")) +
      scale_y_continuous(breaks = round(seq(0, 1, by = 0.5),1)) +
      scale_x_continuous(breaks = round(seq(0, 20, by = 1),1))+
      theme(legend.position = "right")+
      labs(colour="Actors")
    q[[i]] <<- two_plotM
    
    two_plotR <-qplot(xlab=xlabel, ylab=ylabelr, main=paste(subj_num_local, '-money')) +
      geom_line(aes(x=1:length(yr1), y=yr1, color="r1 room")) +
      geom_line(aes(x=1:length(yr2),y=yr2, color = "r2 room")) +
      scale_y_continuous(breaks = round(seq(0, 1, by = 0.5),1)) +
      scale_x_continuous(breaks = round(seq(0, 20, by = 1),1))+
      theme(legend.position = "right")+
      labs(colour="Actors")
    n[[i]] <<- two_plotR
    s2_percent_dist_model = sum(df[df$actor=='s2',]$preferred == 'dist')/20*100
    s1_percent_dist_model = sum(df[df$actor=='s1',]$preferred == 'dist')/20*100
    sprintf("s2 real dist %i", s2_percent_dist_model)
    sprintf("s1 real dist %i", s1_percent_dist_model)

  })
  i=i+1
  j=j+1
}
do.call(grid.arrange,p)
do.call(grid.arrange,q)
do.call(grid.arrange,n)

#calculate mean values for choices between all subjects, human actors
mean_steps_s2 <-list()
mean_steps_s1 <-list()
mean_steps_m1 <-list()
mean_steps_m2 <-list()
mean_steps_r2 <-list()
mean_steps_r1 <-list()

z=1
for (g in transpose(steps_s2)){
  mean_steps_s2[z] <- round(mean(as.numeric(g)), digits=2)
  z=z+1
}
k=1
for (b in transpose(steps_s1)){
  mean_steps_s1[k] <- round(mean(as.numeric(b)), digits=2)
  k=k+1
}
w=1
for (t in transpose(steps_m1)){
  mean_steps_m1[w] <- round(mean(as.numeric(t)), digits=2)
  w=w+1
}
m=1
for (s in transpose(steps_m2)){
  mean_steps_m2[m] <- round(mean(as.numeric(s)), digits=2)
  m=m+1
}
c=1
for (gr in transpose(steps_r2)){
  mean_steps_r2[c] <- round(mean(as.numeric(gr)), digits=2)
  c=c+1
}
d=1
for (w in transpose(steps_r1)){
  mean_steps_r1[d] <- round(mean(as.numeric(w)), digits=2)
  d=d+1
}

#get rid of the greyish default look
theme_set(theme_bw())

#add plots for stress actors with mean choice values, two in one graph
#actor1 = s1, actor2=s2
  qplot(xlab=xlabel, ylab=ylabels, main="Mean choices for stress") +
  geom_line(aes(x=1:length(mean_steps_s1), y=unlist(mean_steps_s1), col="Actor 1"),size=2) +
  geom_line(aes(x=1:length(mean_steps_s2), y=unlist(mean_steps_s2), col ="Actor 2"),size=2) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(breaks = seq(0, 20, 4))+
  theme(legend.position = "bottom", legend.title = element_blank(), 
        plot.title = element_text(size = 20, face = "bold"), axis.title.x = element_text(size = 17, vjust = -0.35),
        axis.title.y = element_text(size = 15, vjust = 0.5),axis.text.x = element_text(size = 12,vjust = 0.5),
        axis.text.y = element_text(size = 15,vjust = 0.5))
  +labs(colour="Actors")

#add plots for meal actors with mean choice values, two in one graph
  #Actor 1 m1, Actor 2 m2
  qplot(xlab=xlabel, ylab=ylabelm, main="Mean choices for meals") +
    geom_line(aes(x=1:length(mean_steps_m1), y=unlist(mean_steps_m1), color="Actor 1"),size=2) +
    geom_line(aes(x=1:length(mean_steps_m2), y=unlist(mean_steps_m2), color ="Actor 2"),size=2) +
    scale_y_continuous(breaks = seq(0, 1, 0.25)) +
    scale_x_continuous(breaks = seq(0, 20, 4))+
    theme(legend.position = "bottom", legend.title = element_blank(), 
          plot.title = element_text(size = 20, face = "bold"), axis.title.x = element_text(size = 17, vjust = -0.35),
          axis.title.y = element_text(size = 15, vjust = 0.5),axis.text.x = element_text(size = 12,vjust = 0.5),
          axis.text.y = element_text(size = 15,vjust = 0.5))
    +labs(colour="Actors")

#add plots for rooms with mean choice values, two in one graph
  #Actor 1 r1 room, actor 2 r2 room
  qplot(xlab=xlabel, ylab=ylabelr, main="Mean choices for non-social") +
    geom_line(aes(x=1:length(mean_steps_r1), y=unlist(mean_steps_r1), color="Actor 1"),size=2) +
    geom_line(aes(x=1:length(mean_steps_r2), y=unlist(mean_steps_r2), color ="Actor 2"),size=2) +
    scale_y_continuous(breaks = seq(0, 1, 0.25)) +
    scale_x_continuous(breaks = seq(0, 20, 4))+
    theme(legend.position = "bottom", legend.title = element_blank(), 
          plot.title = element_text(size = 20, face = "bold"), axis.title.x = element_text(size = 17, vjust = -0.35),
          axis.title.y = element_text(size = 15, vjust = 0.5),axis.text.x = element_text(size = 12,vjust = 0.5),
          axis.text.y = element_text(size = 15,vjust = 0.5))
  +labs(colour="Actors")

  
  

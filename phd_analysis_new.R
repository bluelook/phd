#install.packages(c("ggplot2", "tidyverse", "gridExtra", "purrr"))
#ggthemes, "extrafont", "cowplot", "grid", ggforce, shiny, ggridges, ggrepel, reshape2
#devtools::install_github("thomasp85/patchwork")
library(ggplot2)
library(tidyverse)
library(purrr)
library(gridExtra)
library(gtools)

rm(list = ls()) #clear all
#set working directory to the data files location
setwd('./outputs')
# list the files in a directory, each file represents a subject
file_names <- list.files(pattern = ".csv$")
#read all the files in the directory into one data frame
full_df <- do.call(smartbind, lapply(file_names, read.csv, header=TRUE,stringsAsFactors=FALSE))
#set working directory back to the source file location
setwd('..')

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

# actor_trials <- subject_choices %>% filter(actor == subject_choices$actor[1])
# num_trials <- length(actor_trials$subject_nr)

theme_set(theme_bw())

mean_chosen_all <-
  extracted_df %>% group_by(actor) %>% summarise(mean = mean(chosen_numeric),
                                                 mean_percent = round(mean * 100,digits=2)) %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2"))

ggplot(data = mean_chosen_all,
       aes(x = actor, y = mean_percent, fill = color)) +
  geom_bar(position = position_dodge(), stat = "identity") +
  scale_x_discrete(name = "Strategies") +
  scale_y_continuous(name = "Frequency of chosing strategy 2") +
  labs(title = "Frequency of choosing strategy 2 - between subjects") +
  geom_text(aes(label =  paste(mean_percent, "%")), position = position_dodge(width = 1),
            vjust = -0.5, size = 2)+
  scale_fill_discrete(name="Preferred\nStrategy", labels=c("Strategy 1", "Strategy 2"))

ggsave(path="plots", filename="mean_all_strategy2_freq.png", width = 5, height = 5)

subjects <- unique(extracted_df$subject_nr)
#for each subject, calculate frequencies of choices and plot them
for (subj_num_local in subjects) {
  subject_choices <-
    extracted_df %>% filter(subject_nr == subj_num_local)
  
  #calculate % of 1s (dist, sweet, wood choices)
  mean_chosen_individual <- subject_choices %>% group_by(subject_nr,actor) %>%
    summarise(mean_chosen = mean(chosen_numeric),
              mean_percent_chosen = round(mean_chosen * 100, digits=2))
  
  mean_chosen_individual <- mean_chosen_individual %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2")) 
  
  ggplot(data = subset(mean_chosen_individual, subject_nr == subj_num_local),
         aes(x = actor, y = mean_percent_chosen, fill = color)) +
    geom_bar(position = position_dodge(), stat = "identity") +
    scale_x_discrete(name = "Strategies") +
    scale_y_continuous(name = "Frequency of chosing strategy 2") +
    labs(title = paste("Subject ", subj_num_local)) +
    geom_text(aes(label =  paste(mean_percent_chosen, "%")), position = position_dodge(width = 1),
              vjust = -0.5, size = 2)+
    scale_fill_discrete(name="Preferred\nStrategy", 
                        labels=c("Strategy 1", "Strategy 2"))
  
  ggsave(path="plots", filename = paste("subj",subj_num_local, "_strategy2_freq.png"),
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

par(mfrow = c(length(files)/2+1,2))
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

  
  
  # 
  # if(extracted_df$correct == 0){
  #   if (extracted_df$preferred == 'dist'){
  #     extracted_df$chosen<-'reap'
  #     extracted_df$chosen_numeric<-0
  #   }
  #   else if (extracted_df$preferred == 'reap'){
  #     extracted_df$chosen<-'dist'
  #     extracted_df$chosen_numeric<-1
  #   }
  #   else if (extracted_df$preferred == 'sweet'){
  #     extracted_df$chosen<-'salty'
  #     extracted_df$chosen_numeric<-0
  #   }
  #   else if (extracted_df$preferred == 'salty'){
  #     extracted_df$chosen<-'sweet'
  #     extracted_df$chosen_numeric<-1
  #   }
  #   else if (extracted_df$preferred == 'blue_closet'){
  #     extracted_df$chosen<-'wood_closet'
  #     extracted_df$chosen_numeric<-1
  #   }
  #   else if (extracted_df$preferred == 'wood_closet'){
  #     extracted_df$chosen<-'blue_closet'
  #     extracted_df$chosen_numeric<-0
  #   }
  # }else{
  #   #the choice taken was correct
  #   extracted_df$chosen <- extracted_df$preferred
  #   if(extracted_df$preferred == 'reap' | extracted_df$preferred == 'salty' |  extracted_df$preferred == 'blue_closet'){
  #     extracted_df$chosen_numeric<-0
  #   }
  #   else{ 
  #     extracted_df$chosen_numeric<-1
  #   }
  # }

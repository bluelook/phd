install.packages(c("ggplot2", "tidyverse", "gridExtra", "purrr","plotly"))
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
subj_data <- read_xlsx("responses.xlsx", sheet = "all traversed")

#read all the files in the directory into one data frame
full_df <-
  do.call(smartbind,
          lapply(
            file_names,
            read.csv,
            header = TRUE,
            stringsAsFactors = FALSE
          ))
#set working directory back to the source file location
setwd('../..')

###################### read model data ####################
#model_subj_data <- read_csv("./output/optimizationResultAll.csv")#, check.names = FALSE)

bic_stress <- read_csv("./output/BICStress.csv")
#bic_stress <- read_csv("./output/sub_BICStress.csv")

bic_meal <- read_csv("./output/BICMeal.csv")
#bic_meal <- read_csv("./output/sub_BICMeal.csv")

bic_room <- read_csv("./output/BICRoom.csv")
#bic_room <- read_csv("./output/sub_BICRoom.csv")

no_learning_model <- read_csv("./output/no_learning_model.csv")
#no_learning_model <- read_csv("./output/sub_no_learning.csv")

no_act_strat_indep_model <- read_csv("./output/no_act_strat_indep_model.csv")
#no_act_strat_indep_model <-  read_csv("./output/sub_no_act_strat_indep.csv")

#act_indep_strat_indep_model <- read.csv("./output/act_indep_strat_indep_model.csv")
act_indep_strat_indep_model <-  read.csv("./output/sub_act_indep_strat_indep.csv")

num_subj_filtered <- nrow(act_indep_strat_indep_model)
#num_subj_act_indep_strat_indep <- nrow(act_indep_strat_indep_model)
#num_subj_no_act_strat_indep <- nrow(no_act_strat_indep_model)
#num_subj_no_learning <- nrow(no_learning_model)

#leave only usable subjects per the model
no_act_strat_indep_model <- subset(no_act_strat_indep_model, subject_nr %in% act_indep_strat_indep_model$subject_nr)
no_learning_model <- subset(no_learning_model, subject_nr %in% act_indep_strat_indep_model$subject_nr)

bic_stress$subject_nr <- factor(bic_stress$subject_nr)
bic_meal$subject_nr <- factor(bic_meal$subject_nr)
bic_room$subject_nr <- factor(bic_room$subject_nr)

###################### BAR plots delta bic #########################
#delta bic stress
ggplot(data = bic_stress,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Ind1AB),
         y = deltaBICInd2AB_Ind1AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "stress delta BIC (no generalization vs strategy generalization) per subject", x =
         "subjects")
ggplot(data = bic_stress,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Dep2AB),
         y = deltaBICInd2AB_Dep2AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "stress delta BIC (no generalization vs actor generalization) per subject", x =
         "subjects")
ggplot(data = bic_stress,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Dep1AB),
         y = deltaBICInd2AB_Dep1AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "stress delta BIC (no generalization vs total generalization) per subject", x =
         "subjects")
#delta bic meals
ggplot(data = bic_meal,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Ind1AB),
         y = deltaBICInd2AB_Ind1AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "meals delta BIC (no generalization vs strategy generalization) per subject", x =
         "subjects")
ggplot(data = bic_meal,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Dep2AB),
         y = deltaBICInd2AB_Dep2AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "meals delta BIC (no generalization vs actor generalization) per subject", x =
         "subjects")
ggplot(data = bic_meal,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Dep1AB),
         y = deltaBICInd2AB_Dep1AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "meals delta BIC (no generalization vs total generalization) per subject", x =
         "subjects")
#delta bic rooms
ggplot(data = bic_room,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Ind1AB),
         y = deltaBICInd2AB_Ind1AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "rooms delta BIC (no generalization vs strategy generalization) per subject", x =
         "subjects")
ggplot(data = bic_room,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Dep2AB),
         y = deltaBICInd2AB_Dep2AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "rooms delta BIC (no generalization vs actor generalization) per subject", x =
         "subjects")
ggplot(data = bic_room,
       aes(
         x = reorder(subject_nr, -deltaBICInd2AB_Dep1AB),
         y = deltaBICInd2AB_Dep1AB,
         fill = subject_nr
       )) +
  geom_bar(stat = "identity") +
  labs(title = "rooms delta BIC (no generalization vs total generalization) per subject", x =
         "subjects")



###################### mean bic per model per condition/block ###########################
mean_bic_no_act_strat_indep <- no_act_strat_indep_model %>%
  summarise(
    m_meal = mean(BIC_meal),
    m_room = mean(BIC_room),
    m_stress = mean(BIC_stress),
    sem_room =  sd(BIC_room) / sqrt(num_subj_filtered),
    sem_meal =  sd(BIC_meal) / sqrt(num_subj_filtered),
    sem_stress = sd(BIC_stress) / sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'no_act_strat_indep')
mean_bic_act_indep_strat_indep <- act_indep_strat_indep_model %>%
  summarise(
    m_meal = mean(BIC_meal),
    m_room = mean(BIC_room),
    m_stress = mean(BIC_stress),
    sem_room =  sd(BIC_room) / sqrt(num_subj_filtered),
    sem_meal =  sd(BIC_meal) / sqrt(num_subj_filtered),
    sem_stress = sd(BIC_stress) / sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'act_indep_strat_indep')
mean_bic_no_learning <- no_learning_model %>%
  summarise(
    m_meal = mean(BIC_meal),
    m_room = mean(BIC_room),
    m_stress = mean(BIC_stress),
    sem_room =  sd(BIC_room) / sqrt(num_subj_filtered),
    sem_meal =  sd(BIC_meal) / sqrt(num_subj_filtered),
    sem_stress = sd(BIC_stress) / sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'no_learning')

mean_bic_all_models <-
  rbind(
    mean_bic_no_act_strat_indep,
    mean_bic_act_indep_strat_indep,
    mean_bic_no_learning
  )

act_indep_strat_indep_model_mutated <-
  act_indep_strat_indep_model %>% mutate (model_name = 'act_indep_strat_indep')
#no_learning_model_mutated <- no_learning_model %>% mutate (model_name = 'no_learning_model')
no_act_strat_indep_model_mutated <-
  no_act_strat_indep_model %>% mutate (model_name = 'no_act_strat_indep_model')

all_models <-
  rbind(act_indep_strat_indep_model_mutated,
        no_act_strat_indep_model_mutated)
###################### Box plots BIC, alpha, beta ######################
ggplot(all_models, aes(x = model_name, y = BIC_stress, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.4,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggplot(all_models,
       aes(x = model_name, y = alpha_stress, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggplot(all_models, aes(x = model_name, y = beta_stress, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")

ggplot(all_models, aes(x = model_name, y = BIC_meal, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.4,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggplot(all_models, aes(x = model_name, y = alpha_meal, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggplot(all_models, aes(x = model_name, y = beta_meal, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")

ggplot(all_models, aes(x = model_name, y = BIC_room, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.4,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggplot(all_models, aes(x = model_name, y = alpha_room, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggplot(all_models, aes(x = model_name, y = beta_room, fill = model_name)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")

###################### mean bic for all models per condition/block #############
ggplot(data = mean_bic_all_models, aes(x = model_name, y = m_stress, fill =
                                         model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_stress - sem_stress, ymax = m_stress + sem_stress),
                width = .2) +
  scale_y_continuous(limits = c(0, 70)) +
  labs(title = "mean bic stress for all models", x = "models", y = "bic stress")
  

ggsave("./output/mean bic stress for all models.png", width = 5, height = 5)

ggplot(data = mean_bic_all_models, aes(x = model_name, y = m_room, fill =
                                         model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_room - sem_room, ymax = m_room + sem_room),
                width = .2) +
  scale_y_continuous(limits = c(0, 70)) +
  labs(title = "mean bic rooms for all models", x = "models", y = "bic rooms")
ggsave("./output/mean bic rooms for all models.png", width = 5, height = 5)

ggplot(data = mean_bic_all_models, aes(x = model_name, y = m_meal, fill =
                                         model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_meal - sem_meal, ymax = m_meal + sem_meal),
                width = .2) +
  scale_y_continuous(limits = c(0, 70)) +
  labs(title = "mean bic meals for all models", x = "models", y = "bic meals")
ggsave("./output/mmean bic meals for all models.png", width = 5, height = 5)

###################### mean beta per model per condition/block ########################

mean_beta_no_act_strat_indep <- no_act_strat_indep_model %>%
  summarise(
    m_meal = mean(beta_meal),
    m_room = mean(beta_room),
    m_stress = mean(beta_stress),
    
    sem_room =  sd(beta_room) / sqrt(num_subj_filtered),
    sem_meal =  sd(beta_meal) / sqrt(num_subj_filtered),
    sem_stress = sd(beta_stress) / sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'no_act_strat_indep')
mean_beta_act_indep_strat_indep <- act_indep_strat_indep_model %>%
  summarise(
    m_meal = mean(beta_meal),
    m_room = mean(beta_room),
    m_stress = mean(beta_stress),
    sem_room =  sd(beta_room)/ sqrt(num_subj_filtered),
    sem_meal =  sd(beta_meal)/ sqrt(num_subj_filtered),
    sem_stress = sd(beta_stress) / sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'act_indep_strat_indep')
mean_beta_no_learning <- no_learning_model %>%
  summarise(
    m_meal = mean(beta_meal),
    m_room = mean(beta_room),
    m_stress = mean(beta_stress),
    
    sem_room =  sd(beta_room)/ sqrt(num_subj_filtered),
    sem_meal =  sd(beta_meal)/ sqrt(num_subj_filtered),
    sem_stress = sd(beta_stress) / sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'no_learning')

mean_beta_all_models <-
  rbind(
    mean_beta_no_act_strat_indep,
    mean_beta_act_indep_strat_indep,
    mean_beta_no_learning
  )
ggplot(data = mean_beta_all_models, aes(x = model_name, y = m_stress, fill =
                                          model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_stress - sem_stress, ymax = m_stress + sem_stress),
                width = .2) +
  labs(title = "mean beta stress for all models", x = "models", y = "beta")

ggplot(data = mean_beta_all_models, aes(x = model_name, y = m_room, fill =
                                          model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_room - sem_room, ymax = m_room + sem_room),
                width = .2) +
  labs(title = "mean beta rooms for all models", x = "models", y = "beta")

ggplot(data = mean_beta_all_models, aes(x = model_name, y = m_meal, fill =
                                          model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_meal - sem_meal, ymax = m_meal + sem_meal),
                width = .2) +
  labs(title = "mean beta meals for all models", x = "models", y = "beta")

###################### mean alpha per model per condition/block #######################

alphas <- select(act_indep_strat_indep_model,alpha_stress, alpha_meal, alpha_room, subject_nr)
alphas_longer <-  pivot_longer(alphas, alpha_stress: alpha_room, names_to = "alpha_type", values_to = "value")
alphas_grouped <- alphas_longer %>% group_by(alpha_type) %>% summarise(mean = mean(value), sem = sd(value)/sqrt(num_subj_filtered))
#bar plot means of alpha for specific model
ggplot(data = alphas_grouped, aes(x = alpha_type, y = mean, fill =alpha_type)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = .2) +
  labs(title = "mean alpha", x = "alpha type", y ="value")+
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0,0.4) )
ggsave("./output/mean alpha bar plot per condition.png", width = 5, height = 5)

ggplot(alphas_longer, aes(x = alpha_type, y = value, fill = alpha_type)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun.y = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  theme(legend.position = "none") +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")
ggsave("./output/mean alpha box plot per condition.png", width = 5, height = 5)

mean_alpha_no_act_strat_indep <- no_act_strat_indep_model %>%
  summarise(
    m_meal = mean(alpha_meal),
    m_room = mean(alpha_room),
    m_stress = mean(alpha_stress),
    sem_room =  sd(alpha_room)/sqrt(num_subj_filtered),
    sem_meal =  sd(alpha_meal)/sqrt(num_subj_filtered),
    sem_stress = sd(alpha_stress)/sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'no_act_strat_indep')
mean_alpha_act_indep_strat_indep <- act_indep_strat_indep_model_mutated %>%
  summarise(
    m_meal = mean(alpha_meal),
    m_room = mean(alpha_room),
    m_stress = mean(alpha_stress),
    sem_room =  sd(alpha_room)/sqrt(num_subj_filtered),
    sem_meal =  sd(alpha_meal)/sqrt(num_subj_filtered),
    sem_stress = sd(alpha_stress)/sqrt(num_subj_filtered)
  ) %>%
  mutate (model_name = 'act_indep_strat_indep')


mean_alpha_all_models <-
  rbind(mean_alpha_no_act_strat_indep,
        mean_alpha_act_indep_strat_indep)
ggplot(data = mean_alpha_all_models, aes(x = model_name, y = m_stress, fill =
                                           model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_stress - sem_stress, ymax = m_stress + sem_stress),
                width = .2) +
  labs(title = "mean alpha stress for learning models", x = "models", y =
         "alpha")

ggplot(data = mean_alpha_all_models, aes(x = model_name, y = m_room, fill =
                                           model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_room - sem_room, ymax = m_room + sem_room),
                width = .2) +
  labs(title = "mean alpha rooms for learning models", x = "models", y =
         "alpha")

ggplot(data = mean_alpha_all_models, aes(x = model_name, y = m_meal, fill =
                                           model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_meal - sem_meal, ymax = m_meal + sem_meal),
                width = .2) +
  labs(title = "mean alpha meals for learning models", x = "models", y =
         "alpha")

###################### alpha correlations between domains within one model ##############
act_indep_strat_indep_model_mutated$subject_nr <- factor(act_indep_strat_indep_model_mutated$subject_nr)
a_m_s <- ggplot(data = act_indep_strat_indep_model_mutated, aes(x = alpha_meal, y = alpha_stress)) +
  geom_point(aes(size = alpha_room, colour = factor(subject_nr), fill=factor(subject_nr)))+
  geom_smooth(method = "lm", se = FALSE, color="red")
ggsave("./output/alphas_m_s.png", width = 5, height = 5)

a_m_r <- ggplot(data = act_indep_strat_indep_model_mutated, aes(x = alpha_meal, y = alpha_room)) +
  geom_point(aes(size = alpha_stress, colour = factor(subject_nr), fill=factor(subject_nr)))+
  geom_smooth(method = "lm", se = FALSE, color="red")
ggsave("./output/alphas_m_r.png", width = 5, height = 5)

a_r_s <- ggplot(data = act_indep_strat_indep_model_mutated, aes(x = alpha_room, y = alpha_stress)) +
  geom_point(aes(size = alpha_meal, colour = factor(subject_nr), fill=factor(subject_nr)))+
  geom_smooth(method = "lm", se = FALSE, color="red")
ggsave("./output/alphas_r_s.png", width = 5, height = 5)

grid.arrange(a_m_s, a_m_r, a_r_s, ncol=2, nrow=2)

###################### beta correlations between domains within one model ###########
b_m_s <- ggplot(data = act_indep_strat_indep_model_mutated, aes(x = beta_meal, y = beta_stress)) +
  geom_point(aes(size = beta_room, colour = factor(subject_nr), fill=factor(subject_nr)))+
  geom_smooth(method = "lm", se = FALSE, color="red")
b_m_r <- ggplot(data = act_indep_strat_indep_model_mutated, aes(x = beta_meal, y = beta_room)) +
  geom_point(aes(size = beta_stress, colour = factor(subject_nr), fill=factor(subject_nr)))+
  geom_smooth(method = "lm", se = FALSE, color="red")
b_r_s <- ggplot(data = act_indep_strat_indep_model_mutated, aes(x = beta_room, y = beta_stress)) +
  geom_point(aes(size = beta_meal, colour = factor(subject_nr), fill=factor(subject_nr)))+
  geom_smooth(method = "lm", se = FALSE, color="red")

grid.arrange(b_m_s, b_m_r, b_r_s, ncol=2, nrow=2)

###################### descriptive statistics ###############
#keep only the needed data and add numeric data about choices(dist,sweet,wood = 1. reap, salty, blue = 0)
full_df_filtered <- subset(full_df, subject_nr %in% act_indep_strat_indep_model$subject_nr)

extracted_df <- select(full_df_filtered, subject_nr, actor, correct, preferred)
#for each row convert choices to numeric values
for (i in 1:nrow(extracted_df)) {
  #if the choice taken was incorrect
  if (extracted_df$correct[i] == 0) {
    if (extracted_df$preferred[i] == 'dist') {
      extracted_df$chosen[i] <- 'reap'
      extracted_df$chosen_numeric[i] <- 0
      extracted_df$preferred_numeric[i] <- 1
    }
    else if (extracted_df$preferred[i] == 'reap') {
      extracted_df$chosen[i] <- 'dist'
      extracted_df$chosen_numeric[i] <- 1
      extracted_df$preferred_numeric[i] <- 0
    }
    else if (extracted_df$preferred[i] == 'sweet') {
      extracted_df$chosen[i] <- 'salty'
      extracted_df$chosen_numeric[i] <- 0
      extracted_df$preferred_numeric[i] <- 1
    }
    else if (extracted_df$preferred[i] == 'salty') {
      extracted_df$chosen[i] <- 'sweet'
      extracted_df$chosen_numeric[i] <- 1
      extracted_df$preferred_numeric[i] <- 1
    }
    else if (extracted_df$preferred[i] == 'blue_closet') {
      extracted_df$chosen[i] <- 'wood_closet'
      extracted_df$chosen_numeric[i] <- 1
      extracted_df$preferred_numeric[i] <- 0
    }
    else if (extracted_df$preferred[i] == 'wood_closet') {
      extracted_df$chosen[i] <- 'blue_closet'
      extracted_df$chosen_numeric[i] <- 0
      extracted_df$preferred_numeric[i] <- 1
    }
  } else{
    #the step taken was correct
    extracted_df$chosen[i] = extracted_df$preferred[i]
    if (extracted_df$preferred[i] == 'reap' |
        extracted_df$preferred[i] == 'salty' |
        extracted_df$preferred[i] == 'blue_closet') {
      extracted_df$chosen_numeric[i] <- 0
      extracted_df$preferred_numeric[i] <- 0
    }
    else{
      extracted_df$chosen_numeric[i] <- 1
      extracted_df$preferred_numeric[i] <- 1
    }
  }
}

#descriptive statistics
#summary(extracted_df)

# actor_trials <- subject_choices %>% filter(actor == subject_choices$actor[1])
# num_trials <- length(actor_trials$subject_nr)

subjects <- unique(extracted_df$subject_nr)
theme_set(theme_bw())

#mean by actor between subjects with corrected mean for "0" coded actor
mean_chosen_all <-
  extracted_df %>% group_by(actor) %>%
  summarise(
    mean = mean(chosen_numeric),
    sem = sd(chosen_numeric * 100) / sqrt(num_subj_filtered)
  ) %>%
  mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2")) %>%
  mutate(
    mean = if_else(color == "Condition 1", 1 - mean, mean),
    mean_percent = round(mean * 100, digits = 2)
  )

#plot the mean by actor
ggplot(data = mean_chosen_all,
       aes(x = actor, y = mean_percent, fill = color)) +
  geom_bar(position = position_dodge(), stat = "identity") +
  geom_errorbar(aes(ymin = mean_percent - sem, ymax = mean_percent + sem),
                width = .2) +
  scale_x_discrete(name = "Strategies") +
  scale_y_continuous(name = "Mean Frequency  ") +
  labs(title = "Frequency of choosing preferred strategy - between subjects") +
  geom_text(
    aes(label =  paste(mean_percent, "%")),
    position = position_dodge(width = 1),
    vjust = -0.5,
    size = 2
  ) +
  scale_fill_discrete(name = "Preferred\nStrategy", labels = c("Strategy 1", "Strategy 2"))
ggsave(
  path = "output",
  filename = "mean_all_strategy_freq.png",
  width = 5,
  height = 5
)

#mean by actor within subjects
mean_chosen_by_subject_orig <-
  extracted_df %>% group_by(subject_nr, actor) %>%
  summarise(mean = mean(chosen_numeric)) %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2"))
#change to the complementary percentage for the strategy that is coded as '0'
mean_chosen_by_subject <- mean_chosen_by_subject_orig %>%
  mutate(mean = if_else(color == "Condition 1", 1 - mean, mean),
         mean_percent = mean * 100)
#mean of two actors per condition
mean_by_condition <-
  mean_chosen_by_subject %>% mutate(actor = if_else(grepl('m', actor),
                                                    "m",
                                                    if_else(
                                                      grepl('s', actor), "s",
                                                      if_else(grepl('r',  actor), "r", "")
                                                    ))) %>% group_by(subject_nr, actor) %>%
  summarise(mean = mean(mean_percent)) %>% mutate(dist_from_chance = mean -
                                                    50)

###################### plot three dimentional choices per subject #####################
par(mfrow = c(1, 2))
for (subj in subjects)
{
  subject_choices <- mean_by_condition %>% filter(subject_nr == subj)
  s = subject_choices$mean[subject_choices$actor == 's'] #stress is x axis
  m = subject_choices$mean[subject_choices$actor == 'm'] #meals is y axis
  r = subject_choices$mean[subject_choices$actor == 'r'] #rooms is z axis
  x <- c(s, 0, 0)
  y <- c(0, m, 0)
  z <- c(0, 0, r)
  line_xy <- rbind(x, y)
  line_yz <- rbind(y, z)
  line_xz <- rbind(x, z)
  scatter3D(
    c(0, 0, 0),
    c(0, 0, 0),
    c(0, 0, 0),
    type = "b",
    axis.scales = FALSE,
    bty = "u",
    col.grid = "lightblue",
    xlim = c(0, 100),
    ylim = c(0, 100),
    zlim = c(0, 100),
    xlab = "Stress",
    ylab = "Meals",
    zlab = "Rooms",
    col = "red",
    theta = 120,
    phi = 17,
    ticktype = "detailed",
    nticks = 6,
    main = paste("Subject ", subj)
  )
  scatter3D(
    line_xy[, 1],
    line_xy[, 2],
    line_xy[, 3],
    type = "b",
    axis.scales = FALSE,
    col = "red",
    colkey = FALSE,
    add = TRUE
  )
  scatter3D(
    line_xz[, 1],
    line_xz[, 2],
    line_xz[, 3],
    type = "b",
    col = "darkgreen",
    colkey = FALSE,
    add = TRUE
  )
  scatter3D(
    line_yz[, 1],
    line_yz[, 2],
    line_yz[, 3],
    type = "b",
    col = "blue",
    colkey = FALSE,
    add = TRUE
  )
}

#comparison of distance from 50
#target <- c("s", "r")
#mean_by_condition_s_r <- filter(mean_by_condition,actor %in% target)
mean_by_cond_reshaped <-
  pivot_wider(
    mean_by_condition %>% select(dist_from_chance, subject_nr, actor),
    names_from = "actor",
    values_from = "dist_from_chance"
  )
subj_data_merged <-
  merge(mean_by_cond_reshaped,
        subj_data,
        by = "subject_nr",
        all.x = TRUE)
#subj_data_merged <- merge(subj_data_merged, model_subj_data, by = "subject_nr")
#subj_data_merged <- merge(subj_data_merged,bic_data,by="subject_nr")
#subj_data_merged_filtered <- subj_data_merged[subj_data_merged$BIC_act_dep < 60 &
                     #subj_data_merged$BIC_act_ind < 60 &
                     #subj_data_merged$BIC_act_dep_2A < 60 &
                     #subj_data_merged$BIC_act_ind_2A < 60, ]


#make the columns to be rows to benefit from pipes
#bic_data_gathered <-
  #gather(
  #  subj_data_merged_filtered %>% select(BIC_act_dep, BIC_no_act, BIC_act_ind),
  #  key = 'model',
  #  value = 'bic',
  #  factor_key = TRUE  )


# Model Comparison - calculate mean of bic and sem per each model
#mean_bic <-   bic_data_gathered %>% group_by(model) %>% summarise(mean = mean(bic),  sem = sd(bic) / sqrt(num_subjects))
#my_comparisons <-   list(     c("BIC_act_dep", "BIC_no_act"),     c("BIC_no_act", "BIC_act_ind"),     c("BIC_act_dep", "BIC_act_ind")   )
#compare_means(bic ~ model,  data = bic_data_gathered, method = "anova")

#ggplot(bic_data_gathered, aes(x = model, y = bic, fill = model)) +
#  geom_point(shape = 21, size = 10) +
 # #geom_errorbar(aes(ymin = bic - sem, ymax = bic + sem), width = 0.2) +
#  labs(title = paste(
 #   "Models Comparison p dep vs ind =",
  #  format(t_bic_act_dep_ind$p.value, digits = 2),
   # ", p dep vs no act = ",
  #  format(t_bic_act_dep_no_act$p.value, digits = 2)
  #)) +
  #scale_x_discrete(name = "Model") +
  #scale_y_continuous(name = "BIC") +
  #theme(legend.title = element_blank()) + stat_compare_means(comparisons = my_comparisons) +
  #  Add pairwise comparisons p-value
#  stat_compare_means(label.y = 85)

###################### playing with stats#####
#paired t tests 
t.test(act_indep_strat_indep_model$BIC_stress, no_act_strat_indep_model$BIC_stress, paired = TRUE, alternative = "two.sided")
t.test(act_indep_strat_indep_model$BIC_meal, no_act_strat_indep_model$BIC_meal, paired = TRUE, alternative = "two.sided")
t.test(act_indep_strat_indep_model$BIC_room, no_act_strat_indep_model$BIC_room, paired = TRUE, alternative = "two.sided")
t.test(act_indep_strat_indep_model$alpha_stress, act_indep_strat_indep_model$alpha_meal, paired = TRUE, alternative = "two.sided")
t.test(act_indep_strat_indep_model$alpha_stress, act_indep_strat_indep_model$alpha_room, paired = TRUE, alternative = "two.sided")
#simple linear regression,"YVAR ~ XVAR" where YVAR is the dependent, or predicted,XVAR is the independent, or predictor
lm1 <- lm(sAlpha_act_ind ~ CE, data = subj_data_merged_filtered)
summary(lm1)

glm_s <-
  glm(s ~ sAlpha_act_ind,
      data = subj_data_merged_filtered,
      family = gaussian(link = "identity"))
summary(glm_s)

ggplot(subj_data_merged_filtered, aes(y = s, x = sAlpha_act_dep)) +
  geom_point() +
  stat_smooth(
    method = "glm",
    se = F,
    method.args = list(family = gaussian())
  ) +
  ylab('stress mean value dist from chance') +
  xlab('stress learning rate')

learning_rates <-
  select(
    subj_data_merged_filtered,
    subject_nr,
    sAlpha_act_ind,
    sAlpha_act_dep,
    sAlpha_no_act
  )
learning_rates_longer <-
  pivot_longer(
    learning_rates,
    sAlpha_act_ind:sAlpha_no_act,
    names_to = "learning_rate",
    values_to = "value"
  )
learning_rates_mean <-
  learning_rates_longer %>% group_by(learning_rate) %>% summarise(mean = mean(value), sem =
                                                                    sd(value) / sqrt(12))

ggplot(data = learning_rates_mean,
       aes(x = learning_rate, y = mean, fill = learning_rate)) +
  geom_bar(position = position_dodge(), stat = "identity") +
  scale_x_discrete(name = "learning rate") +
  scale_y_continuous(name = "mean") +
  geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                width = .2)

# Compute the analysis of variance
res.aov <- aov(value ~ learning_rate, data = learning_rates_longer)
# Summary of the analysis
summary(res.aov)

################### for each subject, calculate frequencies of choices and plot them
for (subj_num_local in subjects) {
  subj_choices <-
    extracted_df %>% filter(subject_nr == subj_num_local)
  
  #calculate % of 1s (dist, sweet, wood choices)
  mean_chosen_individual <-
    subj_choices %>% group_by(subject_nr, actor) %>%
    summarise(
      mean_chosen = mean(chosen_numeric),
      mean_percent_chosen = round(mean_chosen * 100, digits = 2)
    )
  
  mean_chosen_individual <-
    mean_chosen_individual %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2"))
  #plot the complementary percentage for the strategy that is coded as '0'
  mean_chosen_individual <-
    mean_chosen_individual %>% mutate(
      mean_percent_chosen = if_else(
        color == "Condition 1",
        100 - mean_percent_chosen,
        mean_percent_chosen
      )
    )
  ggplot(
    data = subset(mean_chosen_individual, subject_nr == subj_num_local),
    aes(x = actor, y = mean_percent_chosen, fill = color)
  ) +
    geom_bar(position = position_dodge(), stat = "identity") +
    scale_x_discrete(name = "Strategies") +
    scale_y_continuous(name = "Frequency of chosing each strategy") +
    labs(title = paste("Subject ", subj_num_local)) +
    geom_text(
      aes(label =  paste(mean_percent_chosen, "%")),
      position = position_dodge(width = 1),
      vjust = -0.5,
      size = 2
    ) +
    scale_fill_discrete(name = "Preferred\nStrategy",
                        labels = c("Strategy 1", "Strategy 2"))
  
  ggsave(
    path = "output",
    filename = paste("subj", subj_num_local, "_strategy2_freq.png"),
    width = 5,
    height = 5
  )
}

#res <-t.test(needed_df[needed_df$actor=='s2',]$chosen_numeric, needed_df[needed_df$actor=='m2',]$chosen_numeric,var.equal = TRUE)
#res <-t.test(data[data$actor=='s2',]$chosen_numeric, data[data$actor=='m2',]$chosen_numeric,var.equal = TRUE)
#print(substring(csv,1,9), res)

###################### create plot for choices over trials per subject, 1 - pref of actor 2, 0 pref of actor 1############

par(mfrow = c(length(subjects) / 2 + 1, 2))
stress_plots <- list()
meal_plots <- list()
room_plots <- list()
i=1
for (subj_num in subjects) {
    df <- extracted_df %>% group_by(subject_nr, actor) %>% mutate(id = row_number()) %>% filter(subject_nr == subj_num) 
    plot_stress <- ggplot() +
      geom_line(data=subset(df, actor=="s1"), aes(x=id, y=chosen_numeric, col="s1") )+
      geom_line(data=subset(df, actor=="s2"), aes(x=id, y=chosen_numeric, col="s2")) + 
      scale_y_continuous(name="Actor 2 preferable choices", breaks = seq(0, 1, by = 0.5) )+
      scale_x_continuous(name="Trial", breaks = seq(0, 20, by = 1)) +
      labs(title = paste(subj_num, '- stress')) 
    plot_meal <- ggplot() +
      geom_line(data=subset(df, actor=="m1"), aes(x=id, y=chosen_numeric, col="m1") )+
      geom_line(data=subset(df, actor=="m2"), aes(x=id,y=chosen_numeric, col="m2")) + 
      scale_y_continuous(name="Actor 2 preferable choices", breaks = seq(0, 1, by = 0.5) )+
      scale_x_continuous(name="Trial", breaks = seq(0, 20, by = 1)) +
      labs(title = paste(subj_num, '- meals')) 
    plot_room <- ggplot() +
      geom_line(data=subset(df, actor=="r1"), aes(x=id, y =chosen_numeric, col="r1") )+
      geom_line(data=subset(df, actor=="r2"), aes(x=id, y =chosen_numeric, col="r2")) + 
      scale_y_continuous(name="Actor 2 preferable choices", breaks = seq(0, 1, by = 0.5) )+
      scale_x_continuous(name="Trial", breaks = seq(0, 20, by = 1)) +
      labs(title = paste(subj_num, '- rooms'))
    
    stress_plots[[i]] <- plot_stress
    meal_plots[[i]] <- plot_meal
    room_plots[[i]] <- plot_room
    i=i+1
}
do.call(grid.arrange, stress_plots)
do.call(grid.arrange, meal_plots)
do.call(grid.arrange, room_plots)

###################### plot of mean choices per each trial between subjects #########################
extracted_df_indexed <- extracted_df %>% group_by(subject_nr,actor) %>% mutate(id = row_number())
mean_steps <- extracted_df_indexed %>% group_by(id, actor) %>% summarise(mean = mean(chosen_numeric))
mean_steps_s1 <- mean_steps %>% filter(actor=="s1")
mean_steps_s2 <- mean_steps %>% filter(actor=="s2")
mean_steps_m1 <- mean_steps %>% filter(actor=="m1")
mean_steps_m2 <- mean_steps %>% filter(actor=="m2")
mean_steps_r1 <- mean_steps %>% filter(actor=="r1")
mean_steps_r2 <- mean_steps %>% filter(actor=="r2")

#add plots for stress actors with mean choice values, two in one graph
ggplot()+
  geom_smooth(data=mean_steps_s1,aes(x =id ,y = mean, col = "Actor 1"), size = 2) +
  geom_smooth(data=mean_steps_s2,aes(x = id, y = mean, col = "Actor 2"), size = 2) +
  scale_y_continuous(name="Actor 2 preferable choices", breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trial", breaks = seq(0, 20, 4)) +
  labs(title = "Mean choices for stress") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 17, vjust = -0.35),
    axis.title.y = element_text(size = 15, vjust = 0.5),
    axis.text.x = element_text(size = 12, vjust = 0.5),
    axis.text.y = element_text(size = 15, vjust = 0.5)
  )
ggsave("./output/mean steps stress.png", width = 5, height = 5)

#add plots for meal actors with mean choice values, two in one graph
ggplot()+
  geom_smooth(data=mean_steps_m1,aes(x =id ,y = mean, col = "Actor 1"), size = 2) +
  geom_smooth(data=mean_steps_m2,aes(x = id, y = mean, col = "Actor 2"), size = 2) +
  scale_y_continuous(name="Actor 2 preferable choices", breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trial", breaks = seq(0, 20, 4)) +
  labs(title = "Mean choices for meals") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 17, vjust = -0.35),
    axis.title.y = element_text(size = 15, vjust = 0.5),
    axis.text.x = element_text(size = 12, vjust = 0.5),
    axis.text.y = element_text(size = 15, vjust = 0.5)
  )
ggsave("./output/mean steps meals.png", width = 5, height = 5)

#add plots for stress actors with mean choice values, two in one graph
ggplot()+
  geom_smooth(data=mean_steps_r1,aes(x =id ,y = mean, col = "Actor 1"), size = 2) +
  geom_smooth(data=mean_steps_r2,aes(x = id, y = mean, col = "Actor 2"), size = 2) +
  scale_y_continuous(name="Actor 2 preferable choices", breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trial", breaks = seq(0, 20, 4)) +
  labs(title = "Mean choices for rooms") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 17, vjust = -0.35),
    axis.title.y = element_text(size = 15, vjust = 0.5),
    axis.text.x = element_text(size = 12, vjust = 0.5),
    axis.text.y = element_text(size = 15, vjust = 0.5)
  )
ggsave("./output/mean steps rooms.png", width = 5, height = 5)

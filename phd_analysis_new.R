#install.packages(c("ggplot2", "tidyverse", "gridExtra", "purrr","plotly"))
#ggthemes, "extrafont", "cowplot", "grid", ggforce, shiny, ggridges, ggrepel, reshape2
#devtools::install_github("thomasp85/patchwork")
library(ggplot2)
library(tidyverse)
library(gridExtra)
library(gtools)
library(plot3D)
library(readxl)
library(ggpval)
library(ggpubr)
library(BayesFactor)
library(rstatix)
library(pwr)

rm(list = ls())
dev.off() #clear all
#set working directory to the data files location
setwd('./data/behavioral')
# list the files in a directory, each file represents a subject
file_names <- list.files(pattern = ".csv$")
num_subjects <- length(file_names)
subj_data <- read_xlsx("responses_assignment.xlsx", sheet = "all traversed")
#subj_data <- read_xlsx("responses.xlsx", sheet = "all traversed")

###################### read all the files in the directory into one data frame and pre-process ###########
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
all_data <- merge(act_indep_strat_indep_model,subj_data,by = "subject_nr")
all_data <- all_data %>% mutate(empathy_level = as.integer(TE > median(all_data$TE)))
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

extracted_df <- extracted_df %>%  mutate(chosen_fixed = if_else(grepl('1', actor), 1 - chosen_numeric, chosen_numeric))
extracted_df_indexed <- extracted_df %>% group_by(subject_nr,actor) %>% mutate(id = row_number())


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
ggsave("./output/stress delta BIC (no generalization vs total generalization).png", width = 5, height = 5)

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
no_learning_model_mutated <- no_learning_model %>% mutate (model_name = 'no_learning_model')
no_act_strat_indep_model_mutated <-
  no_act_strat_indep_model %>% mutate (model_name = 'no_act_strat_indep_model')

all_models <-
  rbind(select(act_indep_strat_indep_model_mutated,BIC_stress,BIC_meal,BIC_room,model_name),
        select(no_act_strat_indep_model_mutated,BIC_stress,BIC_meal,BIC_room,model_name), select(no_learning_model_mutated,BIC_stress,BIC_meal,BIC_room,model_name))
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

###################### BAR plot mean bic for all models per condition/block #############
mean_bic_stress_plot <- ggplot(data = mean_bic_all_models, aes(x = model_name, y = m_stress, fill =
                                         model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_stress - sem_stress, ymax = m_stress + sem_stress),
                width = .2) +
  scale_y_continuous(limits = c(0, 70)) +
  scale_x_discrete(labels=c("model-based", "model-free", "open-loop"))+
  labs(title = "Interpersonal emotion regulation", x = "Models", y = "BIC value")+
  theme(legend.position = "none", legend.title=element_blank(), legend.text=element_text(size=12),
        axis.text=element_text(size=12),   axis.title=element_text(size=16),
        plot.title = element_text(size = 20, face = "bold",hjust=0.5) )+
  geom_signif(y_position = c(63,68), xmin = c(0.8,0.8), 
              xmax = c(2,3.2), annotation = c("**","**"))#p=0.00215 **","p=0.00192 **"))
  

ggsave("./output/mean bic stress for all models.png", width = 5, height = 5)

mean_bic_meal_plot <- ggplot(data = mean_bic_all_models, aes(x = model_name, y = m_meal, fill =
                                         model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_meal - sem_meal, ymax = m_meal + sem_meal),
                width = .2) +
  scale_y_continuous(limits = c(0, 70)) +
  scale_x_discrete(labels=c("model-based", "model-free", "open-loop"))+
  labs(title = "Interpersonal food preference", x = "Models", y = "BIC value")+
  theme(legend.position = "none", legend.title=element_blank(), legend.text=element_text(size=12),
        axis.text=element_text(size=12),   axis.title=element_text(size=16),
        plot.title = element_text(size = 20, face = "bold",hjust=0.5) )+
  geom_signif(y_position = c(63,68), xmin = c(0.8,0.8), xmax = c(2,3.2), annotation = c("*", "**")) #"p=0.04151 *","p=0.002634 **"))

ggsave("./output/mmean bic meals for all models.png", width = 5, height = 5)

mean_bic_room_plot <- ggplot(data = mean_bic_all_models, aes(x = model_name, y = m_room, fill =
                                                               model_name)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = m_room - sem_room, ymax = m_room + sem_room),
                width = .2) +
  scale_y_continuous(limits = c(0, 70)) +
  scale_x_discrete(labels=c("model-based", "model-free", "open-loop"))+
  labs(title = "Value-based", x = "Models", y = "BIC value")+
  theme(legend.position = "none", legend.title=element_blank(), legend.text=element_text(size=12),
        axis.text=element_text(size=12),   axis.title=element_text(size=16),
        plot.title = element_text(size = 20, face = "bold",hjust=0.5) )+
  geom_signif(y_position = c(63,68), xmin = c(0.8,0.8), 
              xmax = c(2,3.2), annotation = c("**","**"))#p=0.003333 **","p=0.003495 **"))

ggsave("./output/mean bic rooms for all models.png", width = 5, height = 5)
grid.arrange(mean_bic_stress_plot, mean_bic_meal_plot, mean_bic_room_plot, nrow=1)

###################### BAR plot mean beta per independent model per condition/block between models ########################

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

###################### BAR plot mean alpha per independent model per condition/block #######################

alphas <- select(act_indep_strat_indep_model,alpha_stress, alpha_meal, alpha_room, subject_nr)
alphas_longer <-  pivot_longer(alphas, alpha_stress: alpha_room, names_to = "alpha_type", values_to = "value")
alphas_grouped <- alphas_longer %>% group_by(alpha_type) %>% summarise(mean = mean(value), sem = sd(value)/sqrt(num_subj_filtered))
#bar plot means of alpha for specific model
ggplot(data = alphas_grouped, aes(x = alpha_type, y = mean, fill =alpha_type)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = .2) +
  labs(title = "mean alpha", x = "alpha type", y ="value")+
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0,0.4) )+
  geom_signif(y_position = c(0.35,0.25,0.38), xmin = c(0.8,1.9,0.8), 
              xmax = c(1.6,2.7,3), annotation = c("NS","NS","NS"))
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

###################### alpha correlations between domains within one model - lm ##############
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

###################### BAR plot mean beta per model per condition/block #######################
betas <- select(act_indep_strat_indep_model,beta_stress, beta_meal, beta_room, subject_nr)
betas_longer <-  pivot_longer(betas, beta_stress: beta_room, names_to = "beta_type", values_to = "value")
betas_grouped <- betas_longer %>% group_by(beta_type) %>% summarise(mean = mean(value), sem = sd(value)/sqrt(num_subj_filtered))
#bar plot means of beta for specific model
ggplot(data = betas_grouped, aes(x = beta_type, y = mean, fill =beta_type)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = .2) +
  labs(title = "mean beta", x = "beta type", y ="value")+
  scale_y_continuous(breaks = seq(0, 15, 1), limits = c(0,15) )+
  geom_signif(y_position = c(10,11.2,12), xmin = c(0.8,2.2,0.8), 
              xmax = c(1.6,2.9,3), annotation = c("NS","NS","NS"))

ggsave("./output/mean beta bar plot per condition.png", width = 5, height = 5)

ggplot(betas_longer, aes(x = beta_type, y = value, fill = beta_type)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(
    fun = mean,
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
ggsave("./output/mean beta box plot per condition.png", width = 5, height = 5)

###################### beta correlations between domains within one model - lm ###########
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

#extracted_df <- extracted_df_indexed %>% filter(id>=10)
#descriptive statistics
#summary(extracted_df)

# actor_trials <- subject_choices %>% filter(actor == subject_choices$actor[1])
# num_trials <- length(actor_trials$subject_nr)

subjects <- unique(extracted_df$subject_nr)
theme_set(theme_bw())
#mean_steps <- extracted_df_indexed %>% group_by(id, actor) %>% summarise(mean_percent = mean(chosen_numeric)*100,sem = sd(chosen_numeric * 100) / sqrt(num_subj_filtered)) %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2"))  %>% ungroup()

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
#chosen_all <-
 # extracted_df %>% mutate(color = if_else(grepl('1', actor), "Condition 1", "Condition 2")) %>%
#t.test(extracted_df$chosen_numeric[extracted_df$actor=='r1'], extracted_df$chosen_numeric[extracted_df$actor=='m1'], paired = TRUE, alternative = "two.sided")


#BAR plot the mean by actor between subjects
ggplot(data = mean_chosen_all,
       aes(x = actor, y = mean_percent, fill = color)) +
  geom_bar(position = position_dodge(), stat = "identity") +
  geom_errorbar(aes(ymin = mean_percent - sem, ymax = mean_percent + sem),
                width = .2) +
  scale_x_discrete(name = "Actors") +
  scale_y_continuous(name = "Choice frequency") +
  #labs(title = "Frequency of choosing preferred strategy - between subjects") +
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


###################### playing with stats#####
# BIC comparisons between models - paired t tests two sided
#distress
#M3 vs M2
t.test(act_indep_strat_indep_model$BIC_stress, no_act_strat_indep_model$BIC_stress, paired = TRUE, alternative = "two.sided")
#M3 vs M1
tt2 <- t.test(act_indep_strat_indep_model$BIC_stress, no_learning_model$BIC_stress, paired = TRUE, alternative = "two.sided")
#Bayes factor t test
ttbf2 <- ttestBF(act_indep_strat_indep_model$BIC_stress, no_act_strat_indep_model$BIC_stress, paired = TRUE, alternative = "two.sided")
#chains = posterior(ttbf2, iterations = 1000)
#summary(chains)
#plot(chains[,1:2])

#food
#M3 vs M2
t.test(act_indep_strat_indep_model$BIC_meal, no_act_strat_indep_model$BIC_meal, paired = TRUE, alternative = "two.sided")
#M3 vs M1
t.test(act_indep_strat_indep_model$BIC_meal, no_learning_model$BIC_meal, paired = TRUE, alternative = "two.sided")

#value-based
#M3 vs M2
t.test(act_indep_strat_indep_model$BIC_room, no_act_strat_indep_model$BIC_room, paired = TRUE, alternative = "two.sided")
#M3 vs M1
t.test(act_indep_strat_indep_model$BIC_room, no_learning_model$BIC_room, paired = TRUE, alternative = "two.sided")

#convert the columns into long format
all_data_l <- pivot_longer(all_data, alpha_meal: alpha_stress, names_to = "condition", values_to = "value")
all_data_l %>% group_by(condition) %>% get_summary_stats(value, type = "mean_sd")
#Create box plots of the learning rates per empathy level (high, low) colored by learning rate types
ggboxplot(all_data_l, x = "empathy_level", y = "value", color="condition", short.panel.labs = FALSE)
#find outliers
outliers <- all_data_l %>% group_by(condition) %>%  identify_outliers(value)#there are outliers in 4 subjects
#remove outliers
all_data_l_f <- all_data_l[!(all_data_l$subject_nr %in% outliers$subject_nr),]
#Create box plots of the learning rates per empathy level (high, low) colored by learning rate types after removing
ggboxplot(all_data_l_f, x = "empathy_level", y = "value", color="condition", short.panel.labs = FALSE)
ggboxplot(all_data_l_f, x = "condition", y = "value", color="condition", short.panel.labs = FALSE)

#three levels (social, food and value-based) repeated measures analyses of variance (ANOVAs) 
res.aov.n <- anova_test(
  data = all_data_l_f, dv = value, wid = subject_nr,
  within = c(condition)
)
get_anova_table(res.aov.n)
# follow up pairwise comparisons t-test alpha
pwc <- all_data_l_f %>%
  pairwise_t_test(value ~ condition, paired = TRUE, p.adjust.method = "bonferroni",alternative = "two.sided") %>%
  select(-df, -statistic) # Remove details

#apha comparisons within M3(act_indep_strat_indep_model) model - paired t tests two sided
#three levels (social, food and value-based) repeated measures analyses of variance (ANOVAs) 
act_indep_strat_indep_model_long <- act_indep_strat_indep_model %>%
  gather(key = "condition", value = "value", alpha_stress, alpha_meal, alpha_room) %>%
  convert_as_factor(subject_nr, condition)

b_act_indep_strat_indep_model_long <- act_indep_strat_indep_model %>%
  gather(key = "condition", value = "b_value", beta_stress, beta_meal, beta_room) %>%
  convert_as_factor(subject_nr, condition)


res.aov <- anova_test(
  data = act_indep_strat_indep_model_long, dv = value, wid = subject_nr,
  within = c(condition)
)
get_anova_table(res.aov)

res.aov_b <- anova_test(
  data = b_act_indep_strat_indep_model_long, dv = b_value, wid = subject_nr,
  within = c(condition)
)
get_anova_table(res.aov_b)

# follow up pairwise comparisons t-test alpha
pwc <- act_indep_strat_indep_model_long %>%
  pairwise_t_test(value ~ condition, paired = TRUE, p.adjust.method = "bonferroni",alternative = "two.sided") %>%
  select(-df, -statistic) # Remove details

# follow up pairwise comparisons t-test beta
pwc_b <- b_act_indep_strat_indep_model_long %>%
  pairwise_t_test(b_value ~ condition, paired = TRUE, p.adjust.method = "bonferroni",alternative = "two.sided") %>%
  select(-df, -statistic) # Remove details

#power analysis
pwr.t.test(n=15, power=0.9, d=NULL, sig.level=0.1868, type="paired", alternative="two.sided")
pwr.t.test(n=15, power=0.9, d=NULL, sig.level=0.5922, type="paired", alternative="two.sided")
pwr.t.test(n=15, power=0.9, d=NULL, sig.level=0.583, type="paired", alternative="two.sided")
pwr.anova.test(k=3, n=15, f=NULL, sig.level = 0.468, power=0.9)

#alpha distress vs alpha food
tt1 <- t.test(act_indep_strat_indep_model$alpha_stress, act_indep_strat_indep_model$alpha_meal, paired = TRUE, alternative = "two.sided")
#beta distress vs alpha food
tt1_b <- t.test(act_indep_strat_indep_model$beta_stress, act_indep_strat_indep_model$beta_meal, paired = TRUE, alternative = "two.sided")

#BF t test
ttbf_a_1 <- ttestBF(act_indep_strat_indep_model$alpha_stress, act_indep_strat_indep_model$alpha_meal, paired = TRUE, alternative = "two.sided")
chains1 = posterior(ttbf_a_1, iterations = 1000)
plot(chains1[,1:2])

#alpha distress vs alpha value based
tt2 <- t.test(act_indep_strat_indep_model$alpha_stress, act_indep_strat_indep_model$alpha_room, paired = TRUE, alternative = "two.sided")
#beta distress vs alpha value based
tt2_b <- t.test(act_indep_strat_indep_model$beta_stress, act_indep_strat_indep_model$beta_room, paired = TRUE, alternative = "two.sided")

#alpha food vs alpha value based
tt3 <- t.test(act_indep_strat_indep_model$alpha_meal, act_indep_strat_indep_model$alpha_room, paired = TRUE, alternative = "two.sided")
#beta food vs alpha value based
tt3_b <- t.test(act_indep_strat_indep_model$beta_meal, act_indep_strat_indep_model$beta_room, paired = TRUE, alternative = "two.sided")

library(pander)
library(broom)
library(purrr)
pander(tt1)

tab <- map_df(list(tt1, tt2, tt3), tidy)
tab[c("estimate", "statistic", "p.value", "conf.low", "conf.high")]

#power analysis
n1 = length(act_indep_strat_indep_model$alpha_meal)
n2 = length(act_indep_strat_indep_model$alpha_stress)
var1 = var(act_indep_strat_indep_model$alpha_meal)
var2 = var(act_indep_strat_indep_model$alpha_stress)
sdpool = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2)/(n1 + n2 -
                                                     + 2))
power <- tt1$estimate/sdpool

var1_b = var(act_indep_strat_indep_model$beta_stress)
var2_b = var(act_indep_strat_indep_model$beta_room)
sdpool_b = sqrt(((n1 - 1) * var1_b + (n2 - 1) * var2_b)/(n1 + n2 -
                                                     + 2))
power_b <- tt2_b$estimate/sdpool_b
ggpairs(act_indep_strat_indep_model)

#comparison of alpha distress/food/value-based for high and low empathy levels - indep t test
t.test(all_data$alpha_stress[all_data$empathy_level==1], all_data$alpha_stress[all_data$empathy_level==0], paired=FALSE, alternative = "two.sided")
t.test(all_data$alpha_meal[all_data$empathy_level==1], all_data$alpha_meal[all_data$empathy_level==0], paired=FALSE, alternative = "two.sided")
t.test(all_data$alpha_room[all_data$empathy_level==1], all_data$alpha_room[all_data$empathy_level==0], paired=FALSE, alternative = "two.sided")

#simple linear regression,"YVAR ~ XVAR" where YVAR is the dependent, or predicted,XVAR is the independent, or predictor
#check whether empathy or its components predict learning rate in distress condition
lm1 <- lm(alpha_stress ~ CE*CR, data = all_data)
summary(lm1)
lmBf1 <- lmBF(alpha_stress ~ TE, data = all_data)
chains1 = posterior(lmBf1, iterations = 1000)
plot(chains1[,2])
lm2 <- lm(alpha_stress ~ TE*ES, data = all_data)
summary(lm2)
lm3 <- lm(alpha_stress ~ AE, data = all_data)
summary(lm3)

glm_s <-
  glm(alpha_stress ~ AE*ES,
      data = all_data,
      family = gaussian(link = "identity"))
summary(glm_s)

ggplot(all_data, aes(y = alpha_stress, x = CE)) +
  geom_point() +
  stat_smooth(method = "glm", se = F, method.args = list(family = gaussian()) , color="red") +
  ylab('distress learning rate') +
  xlab('empathy level')

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

# Compute the analysis of variance: two-way anova - we want to know if learning rate in distress condition depends on empathy score and cognitive reappraisal preference
res.aov <- aov(alpha_stress ~ CE*CR, data = all_data)
# Summary of the analysis
summary(res.aov)


library(nlme)
model = lme(value ~ CE + CR + CE*CR, all_data_l, random = ~1|subject_nr,method="REML")
model.fixed <- gls(alpha_stress ~ CE + CR + CE*CR, data=all_data, method="REML")

anova(model)
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

###################### SMOOTH plot of mean choices per each trial between subjects #########################
mean_steps <- extracted_df_indexed %>% group_by(id, actor) %>% summarise(mean = mean(chosen_numeric))
mean_steps_s1 <- mean_steps %>% filter(actor=="s1")
mean_steps_s2 <- mean_steps %>% filter(actor=="s2")
mean_steps_m1 <- mean_steps %>% filter(actor=="m1")
mean_steps_m2 <- mean_steps %>% filter(actor=="m2")
mean_steps_r1 <- mean_steps %>% filter(actor=="r1")
mean_steps_r2 <- mean_steps %>% filter(actor=="r2")

#add plots for stress actors with mean choice values, two in one graph
mean_steps_stress <- ggplot()+
  geom_smooth(data=mean_steps_s1,aes(x =id ,y = mean, col = "Actor 1"), size = 2) +
  geom_smooth(data=mean_steps_s2,aes(x = id, y = mean, col = "Actor 2"), size = 2) +
  scale_y_continuous(name="Choice frequency", breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trial", breaks = seq(0, 20, 4)) +
  labs(title = "Interpersonal Emotion Regulation") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(size = 18, face = "bold"),
    axis.title.x = element_text(size = 17, vjust = -0.35),
    axis.title.y = element_text(size = 15, vjust = 0.5),
    axis.text.x = element_text(size = 12, vjust = 0.5),
    axis.text.y = element_text(size = 15, vjust = 0.5)
  )
ggsave("./output/mean steps distress.png", width = 5, height = 5)

#add plots for meal actors with mean choice values, two in one graph
mean_steps_meals <- ggplot()+
  geom_smooth(data=mean_steps_m1,aes(x =id ,y = mean, col = "Actor 1"), size = 2) +
  geom_smooth(data=mean_steps_m2,aes(x = id, y = mean, col = "Actor 2"), size = 2) +
  scale_y_continuous(name="Choice frequency", breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trial", breaks = seq(0, 20, 4)) +
  labs(title = "Interpersonal Food Preference") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(size = 18, face = "bold"),
    axis.title.x = element_text(size = 17, vjust = -0.35),
    axis.title.y = element_text(size = 15, vjust = 0.5),
    axis.text.x = element_text(size = 12, vjust = 0.5),
    axis.text.y = element_text(size = 15, vjust = 0.5)
  )
ggsave("./output/mean steps food.png", width = 5, height = 5)

#add plots for stress actors with mean choice values, two in one graph

mean_steps_rooms <- ggplot()+
  geom_smooth(data=mean_steps_r1,aes(x =id ,y = mean, col = "Actor 1"), size = 2) +
  geom_smooth(data=mean_steps_r2,aes(x = id, y = mean, col = "Actor 2"), size = 2) +
  scale_y_continuous(name="Choice frequency", breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trial", breaks = seq(0, 20, 4)) +
  labs(title = "Value-based") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(size = 18, face = "bold"),
    axis.title.x = element_text(size = 17, vjust = -0.35),
    axis.title.y = element_text(size = 15, vjust = 0.5),
    axis.text.x = element_text(size = 12, vjust = 0.5),
    axis.text.y = element_text(size = 15, vjust = 0.5)
  )
ggsave("./output/mean steps rooms.png", width = 5, height = 5)
grid.arrange(mean_steps_stress, mean_steps_meals, mean_steps_rooms, nrow=1)

###################### plots for proposal about predicted models ###########################

id = c(1:20)

const_1 <- rep(0.65,20)
const_2 <- rep(0.7,20)
df_c_1 <- data.frame(id,const_1)
df_c_2 <- data.frame(id,const_2)

no_l <- ggplot() +
  geom_line(data=df_c_1,aes(x =id ,y = const_1, col = "Actor 1"), size = 3) +
  geom_line(data=df_c_1,aes(x =id ,y = const_1, col = "Actor 2"), size = 1, linetype="dashed") +
  scale_y_continuous(name="Frequency of choosing strategy 1", limits = c(0, 1),breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trials")+
  labs(title = "No actor")+ 
  theme(
    axis.text=element_text(size=12),
    axis.text.x = element_blank(),
    axis.ticks = element_blank(), 
    axis.title=element_text(size=17),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    axis.line = element_line(size = 0.5, linetype = "solid", colour = "black"),
    plot.title = element_text(size = 20, face = "bold",hjust=0.5),
    legend.position = "bottom", legend.title=element_blank(), legend.text=element_text(size=12),
    plot.subtitle = element_text(color = "black", face = "italic", hjust = 0.5, size=12))+
  scale_color_manual(values = c('Actor 1' = 'orange','Actor 2' = 'darkblue'))+
  labs(title = "No learning", subtitle="Stable strategy preference to both actors")

mean_v_1 <- seq(0.5, 0.8, length.out = 8)
mean_v_1 <- append(mean_v_1, rep(0.8, 12), 8)
df_1 <- data.frame(id,mean_v_1)
mean_v_2 <- seq(0.5, 0.2, length.out = 8)
mean_v_2 <- append(mean_v_2, rep(0.2, 12), 8)
df_2 <- data.frame(id,mean_v_2)

c_l <-ggplot()+
  geom_smooth(data=df_1,aes(x =id ,y = mean_v_1, color="Actor 1"), size = 2,se=FALSE, ) +
  geom_smooth(data=df_2,aes(x =id ,y = mean_v_2, color="Actor 2"), size = 2,se=FALSE, linetype="dashed") +
  scale_y_continuous(name="Frequency of choosing strategy 1", limits = c(0, 1),breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trials")+
  theme(
    #axis.title.y=element_blank(),
    axis.text=element_text(size=12),
    axis.text.x = element_blank(),
    axis.ticks = element_blank(), 
    axis.title=element_text(size=17),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    axis.line = element_line(size = 0.5, linetype = "solid",colour = "black"),
    plot.title = element_text(size = 20, face = "bold",hjust=0.5),
    legend.position = "bottom", legend.title=element_blank(), legend.text=element_text(size=12),
    plot.subtitle = element_text(color = "black", face = "italic",hjust = 0.5, size=12))+
  scale_color_manual(values = c('Actor 1' = 'orange','Actor 2' = 'darkblue')) +
  labs(title = "Context learning", subtitle = "Fluctuations based on feedback due to ignoring actor identity")


mean_v_4 <- c(0.5, 0.6,0.65,0.7,0.75,0.7,0.6,0.55,0.5,0.66,0.7,0.75,0.7,0.65,0.5,0.45,0.5,0.61,0.7,0.5)
mean_v_3 <- mean_v_4+0.1
#mean_v_3 <- c(0.5, 0.55,0.6,0.62,0.65,0.75,0.7,0.65,0.6,0.56,0.6,0.7,0.75,0.72,0.6,0.55,0.4,0.51,0.65,0.6)
df_3 <- data.frame(id,mean_v_3)
df_4 <- data.frame(id,mean_v_4)

no_c_l <-ggplot()+
  geom_line(data=df_3,aes(x =id ,y = mean_v_3, col = "Actor 1"), size = 3) +
  geom_line(data=df_3,aes(x =id ,y = mean_v_3, col = "Actor 2"), size = 1, linetype="dashed") +
  scale_y_continuous(name="Frequency of choosing strategy 1", limits = c(0, 1),breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(name="Trials")+
  theme(
    #axis.title.y=element_blank(),
    axis.text=element_text(size=12),
    axis.text.x = element_blank(),
    axis.ticks = element_blank(), 
    axis.title=element_text(size=17),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    axis.line = element_line(size = 0.5, linetype = "solid", colour = "black"),
    plot.title = element_text(size = 20, face = "bold",hjust=0.5),
    legend.position = "bottom", legend.title=element_blank(), legend.text=element_text(size=12),
    plot.subtitle = element_text(color = "black", face = "italic", hjust = 0.5, size=12))+
  scale_color_manual(values = c('Actor 1' = 'orange','Actor 2' = 'darkblue'))+
  labs(title = "No context learning", subtitle = "Proper learning of each actor independently")
#grid.arrange(no_l,no_c_l,c_l, nrow=1)


###################### mean BIC per independent model per condition/block #######################
# bics <- select(act_indep_strat_indep_model,BIC_stress, BIC_meal, BIC_room, subject_nr)
# bics_longer <-  pivot_longer(bics, BIC_stress: BIC_room, names_to = "bic_type", values_to = "value")
# bics_grouped <- bics_longer %>% group_by(bic_type) %>% summarise(mean = mean(value), sem = sd(value)/sqrt(num_subj_filtered))
# #bar plot means of bic for specific model
# ggplot(data = bics_grouped, aes(x = bic_type, y = mean, fill =bic_type)) +
#   geom_bar(stat = "identity") +
#   geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = .2) +
#   labs(title = "mean bic", x = "bic type", y ="value")+
#   scale_y_continuous(breaks = seq(0, 70, 10), limits = c(0,70 ))
# ggsave("./output/mean bic bar plot per condition.png", width = 5, height = 5)
# 
# ggplot(bics_longer, aes(x = bic_type, y = value, fill = bic_type)) +
#   geom_boxplot(alpha = 0.7) +
#   stat_summary(
#     fun = mean,
#     geom = "point",
#     shape = 20,
#     size = 10,
#     color = "green"
#   ) +
#   theme(legend.position = "none") +
#   geom_jitter(color = "black",
#               size = 1.2,
#               alpha = 0.9) +
#   scale_fill_brewer(palette = "Set1")
# ggsave("./output/mean bic box plot per condition.png", width = 5, height = 5)
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
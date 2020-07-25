library(ggplot2)
library(tidyverse)
library(readxl)
library(gtools)
library(ggpubr)
library(rstatix)

rm(list = ls())
dev.off() #clear all

#read subject questionairre responses
subj_data <- read_xlsx("responses_assignment.xlsx", sheet = "all traversed")

#read RL model data based on the experiment responses
act_indep_strat_indep_model <-  read.csv("sub_act_indep_strat_indep.csv")

#take only data of subjects who fit the model
all_data <- merge(act_indep_strat_indep_model,subj_data,by = "subject_nr")
#convert cognitive empathy score to "high-1" and "low-0"
all_data <- all_data %>% mutate(empathy_level = as.integer(CE > median(all_data$CE)))
#all_data <- all_data %>% mutate(CR_level = as.integer(CR > median(all_data$CR)))

#number of subjects
num_subj <- nrow(all_data)

#prepare the data for ANOVA: convert the columns into long format
all_data_l <- pivot_longer(all_data, alpha_meal: alpha_stress, names_to = "condition", values_to = "value_a") %>%  
  convert_as_factor(subject_nr, condition,empathy_level)

#group fpr bar plot
data_grouped <- all_data_l %>% group_by(condition) %>% summarise(mean = mean(value_a), sem = sd(value_a)/sqrt(num_subj))

#bar plot means of learning rate per condition
ggplot(data = data_grouped, aes(x = condition, y = mean, fill =condition)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = .2) +
  labs(title = "mean learning rate per condition")+
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0,0.4) ,name="value")+
  scale_x_discrete(labels=c("food","non-social","empathy"), name="condition")+
  geom_signif(y_position = c(0.35,0.25,0.38), xmin = c(0.8,1.9,0.8), 
              xmax = c(1.6,2.7,3), annotation = c("NS","NS","NS"))+
  theme(legend.position = "none", legend.title=element_blank(), legend.text=element_text(size=12),
        axis.text=element_text(size=12),   axis.title=element_text(size=12),
        plot.title = element_text(size = 15, face = "bold",hjust=0.5) )

#same, but box plot with mean
ggplot(all_data_l, aes(x = condition, y = value_a, fill = condition)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "mean learning rate per condition")+
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 20,
    size = 10,
    color = "green"
  ) +
  geom_jitter(color = "black",
              size = 1.2,
              alpha = 0.9) +
  scale_fill_brewer(palette = "Set1")+
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0,1) ,name="value")+
  scale_x_discrete(labels=c("food","non-social","empathy"), name="condition")+
  theme(legend.position = "none", legend.title=element_blank(), legend.text=element_text(size=12),
        axis.text=element_text(size=12),   axis.title=element_text(size=16),
        plot.title = element_text(size = 12, face = "bold",hjust=0.5) )

#Group the data by condition and empathy level, and then compute some summary statistics of the value_a variable: mean and sd (standard deviation)
all_data_l %>% group_by(condition,empathy_level) %>% get_summary_stats(value_a, type = "mean_sd")

#Create box plots of the learning rates per learning rate types (conditions) colored by empathy level (high-1, low-0)
# bxp <-ggboxplot(all_data_l, x = "empathy_level", y = "value_a", color="condition", short.panel.labs = FALSE)+
#   scale_y_continuous(breaks = seq(0, 1, 0.1),name="value")+
#   scale_x_discrete(labels=c("food","non-social","empathy"), name="condition")
bxp <-ggboxplot(all_data_l, x = "condition", y = "value_a", color="empathy_level", short.panel.labs = FALSE)+
   scale_y_continuous(breaks = seq(0, 1, 0.1),name="value")+
   scale_x_discrete(labels=c("food","non-social","empathy"), name="condition")
bxp

#find outliers
outliers <- all_data_l %>% group_by(condition, empathy_level) %>%  identify_outliers(value_a)#there are outliers in 4 subjects
outliers
#remove outliers - OPTIONAL
all_data_l_f <- all_data_l[!(all_data_l$subject_nr %in% outliers$subject_nr),]
all_data_l_f %>% group_by(condition, empathy_level,CR_level) %>% get_summary_stats(value_a, type = "mean_sd")
#Create box plots of the learning rates per learning rate types (conditions) colored by empathy level (high-1, low-0) after removing outliers
#bxp_f <-ggboxplot(all_data_l_f, x = "empathy_level", y = "value_a", color="condition", short.panel.labs = FALSE)+
  # scale_y_continuous(breaks = seq(0, 1, 0.1),name="value")+
 #  scale_x_discrete( labels=c("low empathy","high empathy"),name="condition")
bxp_f <-ggboxplot(all_data_l_f, x = "condition", y = "value_a", color="empathy_level", short.panel.labs = FALSE)+
  scale_y_continuous(breaks = seq(0, 1, 0.1),name="value")+
  scale_x_discrete(labels=c("food","non-social","empathy"), name="condition")

bxp_f

#check the normality assumption by computing Shapiro-Wilk test for each combinations of factor levels
all_data_l_f %>% group_by(empathy_level,condition) %>% shapiro_test(value_a)
# value_a were  normally distributed (p > 0.05) for each cell, as assessed by Shapiro-Wilk’s test of normality.
#or QQ plot to draw the correlation between a given data and the normal distribution
ggqqplot(all_data_l_f, "value_a", ggtheme = theme_bw()) +
  facet_grid(condition ~ empathy_level,labeller = "label_both")

#The homogeneity of variance assumption of the between-subject factor (empathy_level) checked using the Levene’s test. 
#The test is performed at each level of condition variable
all_data_l_f %>%
  group_by(condition) %>%
  levene_test(value_a ~ empathy_level)#There was no homogeneity of variances, as assessed by Levene’s test of homogeneity of variance (p > .05). at conditon "meal"

#The homogeneity of covariances of the between-subject factor (empathy_level) evaluated using the Box’s M-test
box_m(all_data_l_f[, "value_a", drop = FALSE], all_data_l_f$empathy_level) #no homogeneity p<0.001

#two-way mixed ANOVA: within (three levels (social, food and value-based)) variable and between (two levels (low and high)) variable 
res.aov <- anova_test(
  data = all_data_l_f, dv = value_a, wid = subject_nr,
  within = c(condition), between=empathy_level)
#From the output above, it can be seen that, there is a statistically significant two-way interaction 
#between empathy_level and condition on learning rate value, F(2, 22) = 5.67, p < 0.05
get_anova_table(res.aov)

# post-hoc tests
#Simple main effect of the between-subject factor empathy_level on learning rate value at every condition
one.way <- all_data_l_f %>%
  group_by(condition) %>%
  anova_test(dv = value_a, wid = subject_nr, between = empathy_level) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
one.way

# there is simple main effect, but to need to continue to pairwise comparisons between empathy levels at each condition, due to two levels
#however this object helps to draw the final plot
pwc <- all_data_l_f %>% group_by(condition) %>%
   pairwise_t_test(value_a ~ empathy_level, p.adjust.method = "bonferroni")
pwc
#it can be seen that the simple main effect of empathy level was significant at condition "food" (p = 0.039), 
#but not at "non-social" or "empathy" conditions (p = 0.092 and p=0.768).

#OPTIONAL - Simple main effects of within-subject factor condition on learning rate value at each empathy level
one.way2 <- all_data_l_f %>%
  group_by(empathy_level) %>%
  anova_test(dv = value_a, wid = subject_nr, within = condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")

one.way2
# there is a simple main effect of condition, we can continue to pairwise comparisons between conditions at each empathy levels
#not a must, since there are two empathy levels
pwc2 <- all_data_l_f %>%
   group_by(empathy_level) %>%
   pairwise_t_test(value_a ~ condition, paired = TRUE, p.adjust.method = "bonferroni") 
pwc2

# Visualization: boxplots with p-values
pwc <- pwc %>% add_xy_position(x = "condition")
#pwc.filtered <- pwc %>% filter(time != "t1")
bxp_f + stat_pvalue_manual(pwc)+
  labs( subtitle = get_test_label(res.aov, detailed = TRUE),caption = get_pwc_label(pwc))

######################Report###########################################
# since my pilot data is extremely tiny and some of the assumptions are not met, i know that i should not have done the analysis above
#however for the sake of the exercise i did perform it. so the results are:
#There was a statistically significant interaction between cognitive empathy level and experiment condition block in explaining the learning rate value:
#F(2, 22) = 5.67, p < 0.05 (not sure why the plot shows 0.01 :-) ).
#it can be seen that the simple main effect of empathy level was significant at condition "food" (p = 0.039), 
#but not at "non-social" or "empathy" conditions (p = 0.092 and p=0.768).
#Pavel, thank you for the great course!
#Lena
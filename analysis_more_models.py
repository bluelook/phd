import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib as mpl  # plots
import matplotlib.pyplot as plt  # plots
from matplotlib.lines import Line2D
# import os  # OS operations
import scipy.optimize as opt
import scipy.stats as stats
import math
import glob

# agg backend is used to create plot as a .png file
mpl.use('agg')

# M0:baseline model, no learning, preference to one strategy only, choice: dist, sweet, wood_closet
def log_likelihood_no_learning(teta, data, choice):
    # d=distraction, r=reappraisal
    q_2 = teta[0]  # q_d, q_sw, q_w
    q_1 = 1 - q_2  # q_r, q_sa, q_b
    beta = teta[1]
    p_choice_list = []
    for index, row in data.iterrows():
        if row['selected_choice'] == choice:
            p_choice_list.append(np.exp(beta * q_2) / (np.exp(beta * q_2) + np.exp(beta * q_1)))
        else:  # reap
            p_choice_list.append(np.exp(beta * q_1) / (np.exp(beta * q_1) + np.exp(beta * q_2)))
    # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL

# Model M1AB1/2: doesn't learn from actors (doesn't see different people), only from the strategies, oneAB=true: the opposite of each other, oneAB=false: the strategies are the independent of each other
# teta = {alpha, beta} => {learning rate, 1/T (noise)}, choice: dist, sweet, wood_closet, oneAB: true/false
def log_likelihood_no_actor(teta, data, choice, oneAB):
    # d=distraction, r=reappraisal
    q_2 = 0.5  # q_d, q_sw, q_w
    q_1 = 1 - q_2  # q_r, q_sa, q_b
    alpha = teta[0]
    beta = teta[1]
    # 4 p choice options, calculate specific p and add to the list of p choices
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['selected_choice'] == choice:
            p_choice_list.append(np.exp(teta[1] * q_2) / (np.exp(beta * q_2) + np.exp(beta * q_1)))
            # value of the chosen is updated
            q_2 = q_2 + alpha * (reward - q_2)
            if oneAB:
                # the Q value of reap/salty/blue should be updated counterfactually
                q_1 = 1 - q_2
        else:  # reap
            p_choice_list.append(np.exp(teta[1] * q_1) / (np.exp(beta * q_1) + np.exp(beta * q_2)))
            q_1 = q_1 + alpha * (reward - q_1)
            if oneAB:
                # the Q value of dist/sweet/wood should be updated counterfactually
                q_2 = 1 - q_1
    # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL

# Model M2AB1/2: each actor is learned independently, no relation between them. oneAB=true:the strategies are the opposite of each other, oneAB=false: the strategies are the independent of each other.
# teta = {alpha,beta} => {learning rate, 1/T (noise)}, choice: dist, sweet, wood_closet, a_1: s1, m1, r1 (actor)
def log_likelihood_actor_ind_one_arm_b(teta, data, choice, a_1, oneAB):
    q_a1_d = 0.5
    q_a1_r = 0.5
    q_a2_d = 0.5
    q_a2_r = 0.5
    alpha = teta[0]
    beta = teta[1]
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == a_1:
            if row['selected_choice'] == choice:
                p_choice_list.append(np.exp(beta * q_a1_d) / (np.exp(beta * q_a1_d) + np.exp(beta * q_a1_r)))
                q_a1_d = q_a1_d + alpha * (reward - q_a1_d)
                if oneAB:
                    q_a1_r = 1 - q_a1_d
            else:
                p_choice_list.append(np.exp(beta * q_a1_r) / (np.exp(beta * q_a1_r) + np.exp(beta * q_a1_d)))
                q_a1_r = q_a1_r + alpha * (reward - q_a1_r)
                if oneAB:
                    q_a1_d = 1 - q_a1_r
        else:  # actor_2
            if row['selected_choice'] == 'dist':
                p_choice_list.append(np.exp(beta * q_a2_d) / (np.exp(beta * q_a2_d) + np.exp(beta * q_a2_r)))
                q_a2_d = q_a2_d + alpha * (reward - q_a2_d)
                if oneAB:
                    q_a2_r = 1 - q_a2_d
            else:
                p_choice_list.append(np.exp(beta * q_a2_r) / (np.exp(beta * q_a2_r) + np.exp(beta * q_a2_d)))
                q_a2_r = q_a2_r + alpha * (reward - q_a2_r)
                if oneAB:
                    q_a2_d = 1 - q_a2_r
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL

# Model M3AB1/2: the actors are the opposite of each other, oneAB=true:the strategies are the opposite of each other, oneAB=false: the strategies are the independent of each other
# teta = {alpha,beta} => {learning rate, 1/T (noise)},choice: dist, sweet, wood_closet, a_1: s1, m1, r1 (actor)
def log_likelihood_actor_dep(teta, data, choice, a_1, oneAB):
    q_a1_d = 0.5
    q_a1_r = 0.5
    q_a2_d = 0.5
    q_a2_r = 0.5
    alpha = teta[0]
    beta = teta[1]
    # 4 p choice options, calculate specific p and add to the list of p choices
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == a_1:
            if row['selected_choice'] == choice:
                p_choice_list.append(np.exp(beta * q_a1_d) / (np.exp(beta * q_a1_d) + np.exp(beta * q_a1_r)))
                q_a1_d = q_a1_d + alpha * (reward - q_a1_d)
                q_a2_r = q_a2_r + alpha * (
                        reward - q_a2_r)  # the opposite strategy of the opposite actor, generalization of learned on the other person
                if oneAB:  # opposite strategy update
                    q_a1_r = 1 - q_a1_d
                    q_a2_d = 1 - q_a2_r
            else:
                p_choice_list.append(np.exp(beta * q_a1_r) / (np.exp(beta * q_a1_r) + np.exp(beta * q_a1_d)))
                q_a1_r = q_a1_r + alpha * (reward - q_a1_r)
                q_a2_d = q_a2_d + alpha * (
                        reward - q_a2_d)  # the opposite strategy of the opposite actor, generalization of learned on the other person
                if oneAB:  # opposite strategy update
                    q_a1_d = 1 - q_a1_r
                    q_a2_r = 1 - q_a2_d
        else:  # a2
            if row['selected_choice'] == choice:
                p_choice_list.append(np.exp(beta * q_a2_d) / (np.exp(beta * q_a2_d) + np.exp(beta * q_a2_r)))
                q_a2_d = q_a2_d + alpha * (reward - q_a2_d)
                q_a1_r = q_a1_r + alpha * (
                        reward - q_a1_r)  # the opposite strategy of the opposite actor, generalization of learned on the other person
                if oneAB:  # opposite strategy update
                    q_a2_r = 1 - q_a2_d
                    q_a1_d = 1 - q_a1_r
            else:
                p_choice_list.append(np.exp(beta * q_a2_r) / (np.exp(beta * q_a2_r) + np.exp(beta * q_a2_d)))
                q_a2_r = q_a2_r + alpha * (reward - q_a2_r)
                q_a1_d = q_a1_d + alpha * (
                        reward - q_a1_d)  # the opposite strategy of the opposite actor, generalization of learned on the other person
                if oneAB:  # opposite strategy update
                    q_a2_d = 1 - q_a2_r
                    q_a1_r = 1 - q_a1_d
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL

# BIC function
def bic(ll, num_trials, num_param):
    return math.log(num_trials) * num_param + 2 * ll

# boxplotting
def box_plot_png(data, labels, param):
    plt.clf()
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot with patch_artist=True option to ax.boxplot() to get fill color
    bp = ax.boxplot(data, patch_artist=True)
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ## Custom x-axis labels
    ax.set_xticklabels(labels)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Save the figure
    fig.savefig(plotsFolderName + '/' + param + '_box_plot.png', bbox_inches='tight')

# identity line
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

# read all the csv files of participants into a list
folderName = './data/behavioral/'
plotsFolderName = './output'
# csvList = os.listdir(folderName)
csvList = glob.glob(folderName + '*.csv')
# iterate over all files in the folder(over all participants)
optMinDfStress = pd.DataFrame(columns=['subject_nr'])
optMinDfMeal = pd.DataFrame(columns=['subject_nr'])
optMinDfRoom = pd.DataFrame(columns=['subject_nr'])

bic_stress = pd.DataFrame(columns=['subject_nr']) 
bic_meal = pd.DataFrame(columns=['subject_nr'])
bic_room = pd.DataFrame(columns=['subject_nr'])

j = 0
for fileName in csvList:
    # participant = fileName.split('subject-')[1].split('.')[0]
    # read the files into df
    df = pd.read_csv(fileName)
    # pick the needed columns for focused data frame
    focused_df = df[0:120].copy()[['subject_nr', 'actor', 'correct', 'preferred']]
    # recode additional columns - Reward: 1 = reward, 0 = no reward
    focused_df['reward'] = np.where(focused_df.correct == 1, 1, 0)
    focused_df['prev_reward'] = 0

    # mark which strategy was selected by the participant
    selected_choice = []
    for i in range(0, focused_df.shape[0]):
        if focused_df.actor[i] == 's2' or focused_df.actor[i] == 's1':
            # if preferred strategy is distraction and the participant was correct or preferred is reappraisal and was wrong
            # --> selected choice is distraction
            if (focused_df.preferred[i] == 'dist' and focused_df.correct[i] == 1) or (
                    focused_df.preferred[i] == 'reap' and focused_df.correct[i] == 0):
                selected_choice.append('dist')
            # if preferred strategy is reapraisal or participant was correct or preferred is distraction and was wrong
            # --> selected choice is reapraisal
            else:
                selected_choice.append('reap')
        elif focused_df.actor[i] == 'm2' or focused_df.actor[i] == 'm1':
            if (focused_df.preferred[i] == 'sweet' and focused_df.correct[i] == 1) or (
                    focused_df.preferred[i] == 'salty' and focused_df.correct[i] == 0):
                selected_choice.append('sweet')
                # if preferred strategy is salty or participant was correct or preferred is sweet and was wrong
                # --> selected choice is salty
            else:
                selected_choice.append('salty')
        elif focused_df.actor[i] == 'r1' or focused_df.actor[i] == 'r2':
            if (focused_df.preferred[i] == 'wood_closet' and focused_df.correct[i] == 1) or (
                    focused_df.preferred[i] == 'blue_closet' and focused_df.correct[i] == 0):
                selected_choice.append('wood_closet')
            # if preferred strategy is blue_closet or participant was correct or preferred is wood_closet and was wrong
            # --> selected choice is blue_closet
            else:
                selected_choice.append('blue_closet')
    # create new column of choices
    focused_df['selected_choice'] = selected_choice
    xZero = [0.5, 1]  # alpha, beta
    # xZeroTwoAlpha = [0.5, 0.5, 1]
    bnds = ((0.01, 0.99), (0, 15))
    # bndsTwoAlpha = ((0.01, 0.99), (0.01, 0.99), (0, 15))
    # pass only the rows related to specific actors (blocks)
    actorsStress = ['s2', 's1']
    actorsMeal = ['m2', 'm1']
    actorsRoom = ['r2', 'r1']
    # stress block
    resultStressNoLearning = opt.minimize(fun=log_likelihood_no_learning, x0=xZero,
                                          args=(focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist'),
                                          bounds=bnds)
    resultStressNoActorOneArmB = opt.minimize(fun=log_likelihood_no_actor, x0=xZero,
                                              args=(
                                                  focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist', True),
                                              bounds=bnds)
    resultStressActorDepOneArmB = opt.minimize(fun=log_likelihood_actor_dep, x0=xZero,
                                               args=(
                                                   focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist', 's1',
                                                   True), bounds=bnds)
    resultStressActorIndOneArmB = opt.minimize(fun=log_likelihood_actor_ind_one_arm_b, x0=xZero,
                                               args=(
                                                   focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist', 's1',
                                                   True), bounds=bnds)
    resultStressNoActorTwoArmB = opt.minimize(fun=log_likelihood_no_actor, x0=xZero,
                                              args=(
                                                  focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist',
                                                  False),
                                              bounds=bnds)
    resultStressActorDepTwoArmB = opt.minimize(fun=log_likelihood_actor_dep, x0=xZero,
                                               args=(
                                                   focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist', 's1',
                                                   False), bounds=bnds)
    resultStressActorIndTwoArmB = opt.minimize(fun=log_likelihood_actor_ind_one_arm_b, x0=xZero,
                                               args=(
                                                   focused_df.loc[focused_df['actor'].isin(actorsStress)], 'dist', 's1',
                                                   False), bounds=bnds)
    # meals block
    resultMealNoLearning = opt.minimize(fun=log_likelihood_no_learning, x0=xZero,
                                        args=(focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet'),
                                        bounds=bnds)
    resultMealNoActorOneArmB = opt.minimize(fun=log_likelihood_no_actor, x0=xZero,
                                            args=(
                                                focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet', True),
                                            bounds=bnds)
    resultMealActorDepOneArmB = opt.minimize(fun=log_likelihood_actor_dep, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet', 'm1',
                                                 True), bounds=bnds)
    resultMealActorIndOneArmB = opt.minimize(fun=log_likelihood_actor_ind_one_arm_b, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet', 'm1',
                                                 True), bounds=bnds)
    resultMealNoActorTwoArmB = opt.minimize(fun=log_likelihood_no_actor, x0=xZero,
                                            args=(
                                                focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet',
                                                False),
                                            bounds=bnds)
    resultMealActorDepTwoArmB = opt.minimize(fun=log_likelihood_actor_dep, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet', 'm1',
                                                 False), bounds=bnds)
    resultMealActorIndTwoArmB = opt.minimize(fun=log_likelihood_actor_ind_one_arm_b, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsMeal)], 'sweet', 'm1',
                                                 False), bounds=bnds)

    # rooms block
    resultRoomNoLearning = opt.minimize(fun=log_likelihood_no_learning, x0=xZero,
                                        args=(focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet'),
                                        bounds=bnds)
    resultRoomNoActorOneArmB = opt.minimize(fun=log_likelihood_no_actor, x0=xZero,
                                            args=(
                                                focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet',
                                                True),
                                            bounds=bnds)
    resultRoomActorDepOneArmB = opt.minimize(fun=log_likelihood_actor_dep, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet',
                                                 'r1',
                                                 True), bounds=bnds)
    resultRoomActorIndOneArmB = opt.minimize(fun=log_likelihood_actor_ind_one_arm_b, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet',
                                                 'r1',
                                                 True), bounds=bnds)
    resultRoomNoActorTwoArmB = opt.minimize(fun=log_likelihood_no_actor, x0=xZero,
                                            args=(
                                                focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet',
                                                False),
                                            bounds=bnds)
    resultRoomActorDepTwoArmB = opt.minimize(fun=log_likelihood_actor_dep, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet',
                                                 'r1',
                                                 False), bounds=bnds)
    resultRoomActorIndTwoArmB = opt.minimize(fun=log_likelihood_actor_ind_one_arm_b, x0=xZero,
                                             args=(
                                                 focused_df.loc[focused_df['actor'].isin(actorsRoom)], 'wood_closet',
                                                 'r1',
                                                 False), bounds=bnds)

    num_of_trials_stress = focused_df.loc[focused_df['actor'].isin(actorsStress)].shape[0]
    optMinDfStress = optMinDfStress.append(
        {'subject_nr': focused_df.subject_nr[1],
         'q_d_no_learn': resultStressNoLearning.x[0], 'q_r_no_learn': 1 - resultStressNoLearning.x[0],
         'beta_no_learn': resultStressNoLearning.x[1], 'll_no_learn': resultStressNoLearning.fun,
         'sAlpha_act_dep': resultStressActorDepOneArmB.x[0], 'sBeta_act_dep': resultStressActorDepOneArmB.x[1],
         'sLL_act_dep': resultStressActorDepOneArmB.fun,
         'sP_act_dep_model_acc': np.exp(-resultStressActorDepOneArmB.fun / num_of_trials_stress),
         'sAlpha_no_act': resultStressNoActorOneArmB.x[0], 'sBeta_no_act': resultStressNoActorOneArmB.x[1],
         'sLL_no_act': resultStressNoActorOneArmB.fun,
         'sP_no_act_model_acc': np.exp(-resultStressNoActorOneArmB.fun / num_of_trials_stress),
         'sAlpha_act_ind': resultStressActorIndOneArmB.x[0], 'sBeta_act_ind': resultStressActorIndOneArmB.x[1],
         'sLL_act_ind': resultStressActorIndOneArmB.fun,
         'sP_act_ind_model_acc': np.exp(-resultStressActorIndOneArmB.fun / num_of_trials_stress),
         'sAlpha_act_dep_2A': resultStressActorDepTwoArmB.x[0], 'sBeta_act_dep_2A': resultStressActorDepTwoArmB.x[1],
         'sLL_act_dep_2A': resultStressActorDepTwoArmB.fun,
         'sP_act_dep_model_acc_2A': np.exp(-resultStressActorDepTwoArmB.fun / num_of_trials_stress),
         'sAlpha_no_act_2A': resultStressNoActorTwoArmB.x[0], 'sBeta_no_act_2A': resultStressNoActorTwoArmB.x[1],
         'sLL_no_act_2A': resultStressNoActorTwoArmB.fun,
         'sP_no_act_model_acc_2A': np.exp(-resultStressNoActorTwoArmB.fun / num_of_trials_stress),
         'sAlpha_act_ind_2A': resultStressActorIndTwoArmB.x[0], 'sBeta_act_ind_2A': resultStressActorIndTwoArmB.x[1],
         'sLL_act_ind_2A': resultStressActorIndTwoArmB.fun,
         'sP_act_ind_model_acc_2A': np.exp(-resultStressActorIndTwoArmB.fun / num_of_trials_stress)},
        ignore_index=True, sort=False)

    # meals
    num_of_trials_meal = focused_df.loc[focused_df['actor'].isin(actorsMeal)].shape[0]
    optMinDfMeal = optMinDfMeal.append(
        {'subject_nr': focused_df.subject_nr[1],
         'q_d_no_learn': resultMealNoLearning.x[0], 'q_r_no_learn': 1 - resultMealNoLearning.x[0],
         'beta_no_learn': resultMealNoLearning.x[1], 'll_no_learn': resultMealNoLearning.fun,
         'mAlpha_act_dep': resultMealActorDepOneArmB.x[0], 'mBeta_act_dep': resultMealActorDepOneArmB.x[1],
         'mLL_act_dep': resultMealActorDepOneArmB.fun,
         'mP_act_dep_model_acc': np.exp(-resultMealActorDepOneArmB.fun / num_of_trials_meal),
         'mAlpha_no_act': resultMealNoActorOneArmB.x[0], 'mBeta_no_act': resultMealNoActorOneArmB.x[1],
         'mLL_no_act': resultMealNoActorOneArmB.fun,
         'mP_no_act_model_acc': np.exp(-resultMealNoActorOneArmB.fun / num_of_trials_meal),
         'mAlpha_act_ind': resultMealActorIndOneArmB.x[0], 'mBeta_act_ind': resultMealActorIndOneArmB.x[1],
         'mLL_act_ind': resultMealActorIndOneArmB.fun,
         'mP_act_ind_model_acc': np.exp(-resultMealActorIndOneArmB.fun / num_of_trials_meal),
         'mAlpha_act_dep_2A': resultMealActorDepTwoArmB.x[0], 'mBeta_act_dep_2A': resultMealActorDepTwoArmB.x[1],
         'mLL_act_dep_2A': resultMealActorDepTwoArmB.fun,
         'mP_act_dep_model_acc_2A': np.exp(-resultStressActorDepTwoArmB.fun / num_of_trials_meal),
         'mAlpha_no_act_2A': resultMealNoActorTwoArmB.x[0], 'mBeta_no_act_2A': resultMealNoActorTwoArmB.x[1],
         'mLL_no_act_2A': resultMealNoActorTwoArmB.fun,
         'mP_no_act_model_acc_2A': np.exp(-resultMealNoActorTwoArmB.fun / num_of_trials_meal),
         'mAlpha_act_ind_2A': resultMealActorIndTwoArmB.x[0], 'mBeta_act_ind_2A': resultMealActorIndTwoArmB.x[1],
         'mLL_act_ind_2A': resultMealActorIndTwoArmB.fun,
         'mP_act_ind_model_acc_2A': np.exp(-resultMealActorIndTwoArmB.fun / num_of_trials_meal)},
        ignore_index=True, sort=False)

    # rooms
    num_of_trials_room = focused_df.loc[focused_df['actor'].isin(actorsRoom)].shape[0]
    optMinDfRoom = optMinDfRoom.append(
        {'subject_nr': focused_df.subject_nr[1],
         'q_d_no_learn': resultMealNoLearning.x[0], 'q_r_no_learn': 1 - resultMealNoLearning.x[0],
         'beta_no_learn': resultMealNoLearning.x[1], 'll_no_learn': resultMealNoLearning.fun,
         'rAlpha_act_dep': resultMealActorDepOneArmB.x[0], 'rBeta_act_dep': resultMealActorDepOneArmB.x[1],
         'rLL_act_dep': resultMealActorDepOneArmB.fun,
         'rP_act_dep_model_acc': np.exp(
             -resultMealActorDepOneArmB.fun / num_of_trials_room),
         'rAlpha_no_act': resultMealNoActorOneArmB.x[0], 'rBeta_no_act': resultMealNoActorOneArmB.x[1],
         'rLL_no_act': resultMealNoActorOneArmB.fun,
         'rP_no_act_model_acc': np.exp(
             -resultMealNoActorOneArmB.fun / num_of_trials_room),
         'rAlpha_act_ind': resultMealActorIndOneArmB.x[0], 'rBeta_act_ind': resultMealActorIndOneArmB.x[1],
         'rLL_act_ind': resultMealActorIndOneArmB.fun,
         'rP_act_ind_model_acc': np.exp(
             -resultMealActorIndOneArmB.fun / num_of_trials_room),
         'rAlpha_act_dep_2A': resultMealActorDepTwoArmB.x[0], 'rBeta_act_dep_2A': resultMealActorDepTwoArmB.x[1],
         'rLL_act_dep_2A': resultMealActorDepTwoArmB.fun,
         'rP_act_dep_model_acc_2A': np.exp(
             -resultStressActorDepTwoArmB.fun / num_of_trials_room),
         'rAlpha_no_act_2A': resultMealNoActorTwoArmB.x[0], 'rBeta_no_act_2A': resultMealNoActorTwoArmB.x[1],
         'rLL_no_act_2A': resultMealNoActorTwoArmB.fun,
         'rP_no_act_model_acc_2A': np.exp(
             -resultMealNoActorTwoArmB.fun / num_of_trials_room),
         'rAlpha_act_ind_2A': resultMealActorIndTwoArmB.x[0], 'rBeta_act_ind_2A': resultMealActorIndTwoArmB.x[1],
         'rLL_act_ind_2A': resultMealActorIndTwoArmB.fun,
         'rP_act_ind_model_acc_2A': np.exp(
             -resultMealActorIndTwoArmB.fun / num_of_trials_room)},
        ignore_index=True, sort=False)

    # aggregate BIC scores
    # BIC - log likelihood value,num of trials, num of params. bigger BIC, worser model
    bic_stress = bic_stress.append({'subject_nr': focused_df.subject_nr[1],
                                    'BIC_no_learn': bic(resultStressNoLearning.fun, 40, 2),
                                    'BIC_no_act': bic(resultStressNoActorOneArmB.fun, 40, 2),
                                    'BIC_act_dep': bic(resultStressActorDepOneArmB.fun, 40, 2),
                                    'BIC_act_ind': bic(resultStressActorIndOneArmB.fun, 40, 2),
                                    'BIC_no_act_2A': bic(resultStressNoActorTwoArmB.fun, 40, 2),
                                    'BIC_act_dep_2A': bic(resultStressActorDepTwoArmB.fun, 40, 2),
                                    'BIC_act_ind_2A': bic(resultStressActorIndTwoArmB.fun, 40, 2),
                                    'deltaBICInd2AB_Dep2AB': bic(resultStressActorIndTwoArmB.fun, 40, 2) - bic(
                                        resultStressActorDepTwoArmB.fun, 40, 2),
                                    'deltaBICInd2AB_Ind1AB': bic(resultStressActorIndTwoArmB.fun, 40, 2) - bic(
                                        resultStressActorIndOneArmB.fun, 40, 2),
                                    'deltaBICInd2AB_Dep1AB': bic(resultStressActorIndTwoArmB.fun, 40, 2) - bic(
                                        resultStressActorDepOneArmB.fun, 40, 2)}, ignore_index=True, sort=False)

    bic_meal = bic_meal.append({'subject_nr': focused_df.subject_nr[1],
                                    'BIC_no_learn': bic(resultMealNoLearning.fun, 40, 2),
                                    'BIC_no_act': bic(resultMealNoActorOneArmB.fun, 40, 2),
                                    'BIC_act_dep': bic(resultMealActorDepOneArmB.fun, 40, 2),
                                    'BIC_act_ind': bic(resultMealActorIndOneArmB.fun, 40, 2),
                                    'BIC_no_act_2A': bic(resultMealNoActorTwoArmB.fun, 40, 2),
                                    'BIC_act_dep_2A': bic(resultMealActorDepTwoArmB.fun, 40, 2),
                                    'BIC_act_ind_2A': bic(resultMealActorIndTwoArmB.fun, 40, 2),
                                    'deltaBICInd2AB_Dep2AB': bic(resultMealActorIndTwoArmB.fun, 40, 2) - bic(
                                        resultMealActorDepTwoArmB.fun, 40, 2),
                                    'deltaBICInd2AB_Ind1AB': bic(resultMealActorIndTwoArmB.fun, 40, 2) - bic(
                                        resultMealActorIndOneArmB.fun, 40, 2),
                                    'deltaBICInd2AB_Dep1AB': bic(resultMealActorIndTwoArmB.fun, 40, 2) - bic(
                                        resultMealActorDepOneArmB.fun, 40, 2)}, ignore_index=True, sort=False)

    bic_room = bic_room.append({'subject_nr': focused_df.subject_nr[1],
                                'BIC_no_learn': bic(resultRoomNoLearning.fun, 40, 2),
                                'BIC_no_act': bic(resultRoomNoActorOneArmB.fun, 40, 2),
                                'BIC_act_dep': bic(resultRoomActorDepOneArmB.fun, 40, 2),
                                'BIC_act_ind': bic(resultRoomActorIndOneArmB.fun, 40, 2),
                                'BIC_no_act_2A': bic(resultRoomNoActorTwoArmB.fun, 40, 2),
                                'BIC_act_dep_2A': bic(resultRoomActorDepTwoArmB.fun, 40, 2),
                                'BIC_act_ind_2A': bic(resultRoomActorIndTwoArmB.fun, 40, 2),
                                'deltaBICInd2AB_Dep2AB': bic(resultRoomActorIndTwoArmB.fun, 40, 2) - bic(
                                    resultRoomActorDepTwoArmB.fun, 40, 2),
                                'deltaBICInd2AB_Ind1AB': bic(resultRoomActorIndTwoArmB.fun, 40, 2) - bic(
                                    resultRoomActorIndOneArmB.fun, 40, 2),
                                'deltaBICInd2AB_Dep1AB': bic(resultRoomActorIndTwoArmB.fun, 40, 2) - bic(
                                    resultRoomActorDepOneArmB.fun, 40, 2)}, ignore_index=True, sort=False)
    j = j + 1
subset_bic_stress = bic_stress[(bic_stress['BIC_act_ind'] < 60) & (bic_stress['BIC_act_dep'] < 60)]
subset_bic_meal = bic_meal[(bic_meal['BIC_act_ind'] < 60) & (bic_meal['BIC_act_dep'] < 60)]
subset_bic_room = bic_room[(bic_room['BIC_act_ind'] < 60) & (bic_room['BIC_act_dep'] < 60)]

# t test to compare BICs, see if there is a significant difference between the means of two groups: act independent and actor ind
s_t_test_result = stats.ttest_ind(subset_bic_stress['BIC_act_ind'], subset_bic_stress['BIC_act_dep'])
print("stress act ind vs act dep {} :".format(s_t_test_result))
m_t_test_result = stats.ttest_ind(subset_bic_meal['BIC_act_ind'], subset_bic_meal['BIC_act_dep'])
print("meal act ind vs act dep {} :".format(m_t_test_result))
r_t_test_result = stats.ttest_ind(subset_bic_room['BIC_act_ind'], subset_bic_room['BIC_act_dep'])
print("room act ind vs act dep {} :".format(m_t_test_result))
s_t_test_result = stats.ttest_ind(subset_bic_stress['BIC_act_ind_2A'], subset_bic_stress['BIC_act_dep_2A'])
print("stress act ind vs act dep 2armB{} :".format(s_t_test_result))
m_t_test_result = stats.ttest_ind(subset_bic_meal['BIC_act_ind_2A'], subset_bic_meal['BIC_act_dep_2A'])
print("meal act ind vs act dep 2armB{} :".format(m_t_test_result))
r_t_test_result = stats.ttest_ind(subset_bic_room['BIC_act_ind_2A'], subset_bic_room['BIC_act_dep_2A'])
print("room act ind vs act dep 2armB{} :".format(r_t_test_result))
print(subset_bic_stress)
print(subset_bic_meal)
print(subset_bic_room)

#save params and scores to csv
optMinDfStress.to_csv(plotsFolderName + '/optResultStress.csv', index=False)
optMinDfMeal.to_csv(plotsFolderName + '/optResultMeal.csv', index=False)
optMinDfRoom.to_csv(plotsFolderName + '/optResultRoom.csv', index=False)

bic_stress.to_csv(plotsFolderName + '/BICStress.csv', index=False)
bic_meal.to_csv(plotsFolderName + '/BICMeal.csv', index=False)
bic_room.to_csv(plotsFolderName + '/BICRoom.csv', index=False)

alpha_to_plot = [optMinDfStress.sLL_act_dep, optMinDfStress.sLL_act_ind, optMinDfStress.sLL_no_act]
beta_to_plot = [optMinDfStress.sBeta_act_dep, optMinDfStress.sBeta_act_ind, optMinDfStress.sBeta_no_act]
ll_to_plot = [optMinDfStress.sLL_act_dep, optMinDfStress.sBeta_act_ind, optMinDfStress.sLL_no_act]

box_plot_png(alpha_to_plot, ['\u03B1 stress act dep', '\u03B1 stress act ind', '\u03B1 stress no act'], '\u03B1')
box_plot_png(beta_to_plot, ['\u03B2 stress act dep', '\u03B2 meal act ind', '\u03B2 stress no act'], '\u03B2')
box_plot_png(ll_to_plot, ['LL stress act dep', 'LL stress act ind', 'LL stress no act'], 'LL')

# plot alphas relation
# people
# ax1 = optMinDf.plot(kind='scatter', x='sAlphaD', y='sAlphaR', color='red')
# add_identity(ax1)
# ax1.set_xlabel('\u03B1 distraction')
# ax1.set_ylabel('\u03B1 reappraisal')
# optMinDf[['sAlphaD','sAlphaR','participantID']].apply(lambda row: ax1.text(*row),axis=1);
# plt.savefig(plotsFolderName + '/stress alphas.png', dpi=300)
# plt.clf()

# for i,name in enumerate(stat_df['name']):
#   x = stat_df.loc[i,'lr']
#  y = stat_df.loc[i,'PT']
# plt.scatter(x, y, marker='o', color='red')
# if type(name) is str:
#   plt.text(x-0.03, y+0.5, name, fontsize=20, color='blue')


# meal
# ax2 = optMinDf.plot(kind='scatter', x='mAlphaSW', y='mAlphaSA', color='red')
# add_identity(ax2)
# ax2.set_xlabel('\u03B1 sweet meal')
# ax2.set_ylabel('\u03B1 salty meal')
# optMinDf[['mAlphaSW','mAlphaSA','participantID']].apply(lambda row: ax2.text(*row),axis=1);
# plt.savefig(plotsFolderName + '/meals alphas.png',dpi=300)
# plt.clf()

# rooms
# ax3 = optMinDf.plot(kind='scatter', x='rAlphaW', y='rAlphaB', color='red')
# add_identity(ax3)
# ax3.set_xlabel('\u03B1 wood cabinet')
# ax3.set_ylabel('\u03B1 blue cabinet')
# optMinDf[['rAlphaW','rAlphaB','participantID']].apply(lambda row: ax3.text(*row),axis=1);
# plt.savefig(plotsFolderName + '/room alphas.png',dpi=300)
# plt.clf()

# ind diff comparison
ax_bic = subset_bic_stress.plot.scatter(x='BIC_act_dep', y='BIC_no_act', colormap='winter')
# add_identity(ax_bic)
line = Line2D(xdata=[0, 70], ydata=[0, 70], color="red")
ax_bic.add_line(line)
ax_bic.set_xlabel('BIC sLL_act_dep')
ax_bic.set_ylabel('BIC sLL_no_act')
plt.savefig(plotsFolderName + '/ind diff comparison BIC dep vs no act.png', dpi=300)
plt.clf()

ax_bic = subset_bic_stress.plot.scatter(x='BIC_act_ind', y='BIC_no_act', colormap='winter')
# add_identity(ax_bic)
line = Line2D(xdata=[0, 70], ydata=[0, 70], color="red")
ax_bic.add_line(line)
ax_bic.set_xlabel('BIC sLL_act_ind')
ax_bic.set_ylabel('BIC sLL_no_act')
plt.savefig(plotsFolderName + '/ind diff comparison BIC ind vs no act.png', dpi=300)

ax_bic = subset_bic_stress.plot.scatter(x='BIC_act_ind', y='BIC_act_dep', colormap='winter')
# add_identity(ax_bic)
line = Line2D(xdata=[0, 70], ydata=[0, 70], color="red")
ax_bic.add_line(line)
ax_bic.set_xlabel('BIC sLL_act_ind')
ax_bic.set_ylabel('BIC sLL_act_dep')
plt.savefig(plotsFolderName + '/ind diff comparison BIC ind vs act dep.png', dpi=300)

ax_alpha = optMinDfStress.plot.scatter(x='sAlpha_act_dep',
                                       y='sAlpha_act_ind')  # , c='isActIndVsActDepBetter', colormap='winter')
# add_identity(ax_bic)
line = Line2D(xdata=[0, 1], ydata=[0, 1], color="red")
ax_alpha.add_line(line)
ax_alpha.set_xlabel('alpha sLL_act_dep')
ax_alpha.set_ylabel('alpha sLL_act_ind')
plt.savefig(plotsFolderName + '/ind diff comparison alpha ind vs act dep.png', dpi=300)

ax_alpha = optMinDfStress.plot.scatter(x='sAlpha_act_dep_2A',
                                       y='sAlpha_act_ind_2A')  # , c='isActIndVsActDepBetter', colormap='winter')
# add_identity(ax_bic)
line = Line2D(xdata=[0, 1], ydata=[0, 1], color="red")
ax_alpha.add_line(line)
ax_alpha.set_xlabel('alpha sLL_act_dep_2A')
ax_alpha.set_ylabel('alpha sLL_act_ind_2A')
plt.savefig(plotsFolderName + '/ind diff comparison alpha ind vs act dep 2armB.png', dpi=300)

print("Successfully processed %s output files" % len(csvList))

###########
# teta = {alpha0, alpha1, beta} => {learning rate dist, learning rate reap, 1/T (noise)}
# def logLikelihoodStressTwoAlpha(teta, data):
#     # s1 = stress actor 1, s2 = stress actor 2, d=distraction, r=reappraisal
#     q_s1_d = 0
#     q_s1_r = 0
#     q_s2_d = 0
#     q_s2_r = 0
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 's1':
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(teta[2] * q_s1_d) / (np.exp(teta[2] * q_s1_d) + np.exp(teta[2] * q_s1_r)))
#                 q_s1_d = q_s1_d + teta[0] * (reward - q_s1_d)
#             else:  # reap
#                 p_choice_list.append(np.exp(teta[2] * q_s1_r) / (np.exp(teta[2] * q_s1_r) + np.exp(teta[2] * q_s1_d)))
#                 q_s1_r = q_s1_r + teta[1] * (reward - q_s1_r)
#         else:  # stress actor 2
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(teta[2] * q_s2_d) / (np.exp(teta[2] * q_s2_d) + np.exp(teta[2] * q_s2_r)))
#                 q_s2_d = q_s2_d + teta[0] * (reward - q_s2_d)
#             else:  # reap
#                 p_choice_list.append(np.exp(teta[2] * q_s2_r) / (np.exp(teta[2] * q_s2_r) + np.exp(teta[2] * q_s2_d)))
#                 q_s2_r = q_s2_r + teta[1] * (reward - q_s2_r)
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
#
#
# # teta = {alpha0, alpha1, beta} => {learning rate sweet, learning rate salty, 1/T (noise)}
# def logLikelihoodMealTwoAlpha(teta, data):
#     # m2 = meal actor 2, m1 = meal actor 1, sw=sweet, sa=salty
#     q_m2_sw = 0
#     q_m2_sa = 0
#     q_m1_sw = 0
#     q_m1_sa = 0
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         # teta[2] = beta. teta[0] = alpha sweet, teta[1] = alpha salty
#         if row['actor'] == 'm2':
#             if row['selected_choice'] == 'sweet':
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_m2_sw) / (np.exp(teta[2] * q_m2_sw) + np.exp(teta[2] * q_m2_sa)))
#                 q_m2_sw = q_m2_sw + teta[0] * (reward - q_m2_sw)
#             else:
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_m2_sa) / (np.exp(teta[2] * q_m2_sa) + np.exp(teta[2] * q_m2_sw)))
#                 q_m2_sa = q_m2_sa + teta[1] * (reward - q_m2_sa)
#         else:  # meal actor 1
#             if row['selected_choice'] == 'sweet':
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_m1_sw) / (np.exp(teta[2] * q_m1_sw) + np.exp(teta[2] * q_m1_sa)))
#                 q_m1_sw = q_m1_sw + teta[0] * (reward - q_m1_sw)
#             else:
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_m1_sa) / (np.exp(teta[2] * q_m1_sa) + np.exp(teta[2] * q_m1_sw)))
#                 q_m1_sa = q_m1_sa + teta[1] * (reward - q_m1_sa)
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
#
#
# # teta = {alpha0, alpha1, beta} => {learning rate wood closet, learning rate blue closet, 1/T (noise)}
# def logLikelihoodRoomTwoAlpha(teta, data):
#     # r1 = room 1 (white), r2 = room2 (green), wo=wood_closet, bl=blue_closet
#     q_r1_wo = 0
#     q_r1_bl = 0
#     q_r2_wo = 0
#     q_r2_bl = 0
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 'r1':
#             if row['selected_choice'] == 'wood_closet':
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_r1_wo) / (np.exp(teta[2] * q_r1_wo) + np.exp(teta[2] * q_r1_bl)))
#                 q_r1_wo = q_r1_wo + teta[0] * (reward - q_r1_wo)
#             else:
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_r1_bl) / (np.exp(teta[2] * q_r1_bl) + np.exp(teta[2] * q_r1_wo)))
#                 q_r1_bl = q_r1_bl + teta[1] * (reward - q_r1_bl)
#         else:  # r2 green room
#             if row['selected_choice'] == 'wood_closet':
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_r2_wo) / (np.exp(teta[2] * q_r2_wo) + np.exp(teta[2] * q_r2_bl)))
#                 q_r2_wo = q_r2_wo + teta[0] * (reward - q_r2_wo)
#             else:  # blue_closet
#                 p_choice_list.append(
#                     np.exp(teta[2] * q_r2_bl) / (np.exp(teta[2] * q_r2_bl) + np.exp(teta[2] * q_r2_wo)))
#                 q_r2_bl = q_r2_bl + teta[1] * (reward - q_r2_bl)
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL

# teta = {alpha, beta} => {Q, 1/T (noise)}
# def logLikelihoodStressNoLearning(teta, data):
#     # d=distraction, r=reappraisal
#     q_d = teta[0]
#     q_r = 1 - q_d
#     beta = teta[1]
#     p_choice_list = []
#     for index, row in data.iterrows():
#         if row['selected_choice'] == 'dist':
#             p_choice_list.append(np.exp(beta * q_d) / (np.exp(beta * q_d) + np.exp(beta * q_r)))
#         else:  # reap
#             p_choice_list.append(np.exp(beta * q_r) / (np.exp(beta * q_r) + np.exp(beta * q_d)))
#     # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL

# # Model M1AB1: doesn't learn from actors (doesn't see different people), only from the strategies, which are the opposite of each other
# #baseline model
# # teta = {alpha, beta} => {learning rate, 1/T (noise)}
# def logLikelihoodStressNoActorOneArmB(teta, data):
#     # d=distraction, r=reappraisal
#     q_d = 0.5
#     q_r = 1 - q_d
#     alpha = teta[0]
#     beta = teta[1]
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         #prev_reward = row['prev_reward']
#         if row['selected_choice'] == 'dist':
#             p_choice_list.append(np.exp(teta[1] * q_d) / (np.exp(beta * q_d) + np.exp(beta * q_r)))
#             #value of the chosen is updated
#             q_d = q_d + alpha * (reward - q_d)
#             # the Q value of reap should be updated counterfactually
#             q_r = 1 - q_d
#         else:  # reap
#             p_choice_list.append(np.exp(teta[1] * q_r) / (np.exp(beta * q_r) + np.exp(beta * q_d)))
#             q_r = q_r + alpha * (reward - q_r)
#             # the Q value of reap should be updated counterfactually
#             q_d = 1 - q_r
#     # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL

# # Model M2AB1: each actor is learned independently, no relation between them. the strategies are the opposite of each other.
# # teta = {alpha,beta} => {learning rate, 1/T (noise)}
# def logLikelihoodStressActorIndOneArmB(teta, data):
#     # s1 = stress actor 1, s2 = stress actor 2, d=distraction, r=reappraisal
#     q_s1_d = 0.5
#     q_s1_r = 0.5
#     q_s2_d = 0.5
#     q_s2_r = 0.5
#     alpha = teta[0]
#     beta = teta[1]
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 's1':
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s1_d) / (np.exp(beta * q_s1_d) + np.exp(beta * q_s1_r)))
#                 q_s1_d = q_s1_d + alpha * (reward - q_s1_d)
#                 q_s1_r = 1 - q_s1_d
#             else:
#                 p_choice_list.append(np.exp(beta * q_s1_r) / (np.exp(beta * q_s1_r) + np.exp(beta * q_s1_d)))
#                 q_s1_r = q_s1_r + alpha * (reward - q_s1_r)
#                 q_s1_d = 1 - q_s1_r
#         else:  # s2
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s2_d) / (np.exp(beta * q_s2_d) + np.exp(beta * q_s2_r)))
#                 q_s2_d = q_s2_d + alpha * (reward - q_s2_d)
#                 q_s2_r = 1 - q_s2_d
#             else:
#                 p_choice_list.append(np.exp(beta * q_s2_r) / (np.exp(beta * q_s2_r) + np.exp(beta * q_s2_d)))
#                 q_s2_r = q_s2_r + alpha * (reward - q_s2_r)
#                 q_s2_d = 1 - q_s2_r
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
# Model M3AB1: the actors are the opposite of each other, the strategies are the opposite of each other
# teta = {alpha,beta} => {learning rate, 1/T (noise)}
# def logLikelihoodStressActorDepOneArmB(teta, data):
#     # s1 = stress actor 1, s2 = stress actor 2, d=distraction, r=reappraisal
#     q_s1_d = 0.5
#     q_s1_r = 0.5
#     q_s2_d = 0.5
#     q_s2_r = 0.5
#     alpha = teta[0]
#     beta = teta[1]
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 's1':
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s1_d) / (np.exp(beta * q_s1_d) + np.exp(beta * q_s1_r)))
#                 q_s1_d = q_s1_d + alpha * (reward - q_s1_d)
#                 q_s1_r = 1 - q_s1_d #opposite strategy update
#                 q_s2_r = q_s2_r + alpha * (reward - q_s2_r) #the opposite strategy of the opposite actor, generalization of learned on the other person
#                 q_s2_d = 1 - q_s2_r #
#             else:
#                 p_choice_list.append(np.exp(beta * q_s1_r) / (np.exp(beta * q_s1_r) + np.exp(beta * q_s1_d)))
#                 q_s1_r = q_s1_r + alpha * (reward - q_s1_r)
#                 q_s1_d = 1 - q_s1_r
#                 q_s2_d = q_s2_d + alpha * (reward - q_s2_d) #the opposite strategy of the opposite actor, generalization of learned on the other person
#                 q_s2_r = 1 - q_s2_d
#         else:  # s2
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s2_d) / (np.exp(beta * q_s2_d) + np.exp(beta * q_s2_r)))
#                 q_s2_d = q_s2_d + alpha * (reward - q_s2_d)
#                 q_s2_r = 1 - q_s2_d
#                 q_s1_r = q_s1_r + alpha * (reward - q_s1_r) #the opposite strategy of the opposite actor, generalization of learned on the other person
#                 q_s1_d = 1 - q_s1_r
#             else:
#                 p_choice_list.append(np.exp(beta * q_s2_r) / (np.exp(beta * q_s2_r) + np.exp(beta * q_s2_d)))
#                 q_s2_r = q_s2_r + alpha * (reward - q_s2_r)
#                 q_s2_d = 1 - q_s2_r
#                 q_s1_d = q_s1_d + alpha * (reward - q_s1_d) #the opposite strategy of the opposite actor, generalization of learned on the other person
#                 q_s1_r = 1 - q_s1_d
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
# Model M1AB2: doesn't learn from actors (doesn't see different people), only from the strategies, which are the independent of each other
# teta = {alpha, beta} => {learning rate, 1/T (noise)}
# def logLikelihoodStressNoActorTwoArmB(teta, data):
#     # d=distraction, r=reappraisal
#     q_d = 0.5
#     q_r = 1 - q_d
#     alpha = teta[0]
#     beta = teta[1]
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         #prev_reward = row['prev_reward']
#         if row['selected_choice'] == 'dist':
#             p_choice_list.append(np.exp(teta[1] * q_d) / (np.exp(beta * q_d) + np.exp(beta * q_r)))
#             #value of the chosen is updated
#             q_d = q_d + alpha * (reward - q_d)
#         else:  # reap
#             p_choice_list.append(np.exp(teta[1] * q_r) / (np.exp(beta * q_r) + np.exp(beta * q_d)))
#             q_r = q_r + alpha * (reward - q_r)
#     # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
# Model M3AB2: the actors are the opposite of each other, the strategies are the independent of each other - is this correct?? TODO
# teta = {alpha,beta} => {learning rate, 1/T (noise)}
# def logLikelihoodStressActorDepTwoArmB(teta, data):
#     # s1 = stress actor 1, s2 = stress actor 2, d=distraction, r=reappraisal
#     q_s1_d = 0.5
#     q_s1_r = 0.5
#     q_s2_d = 0.5
#     q_s2_r = 0.5
#     alpha = teta[0]
#     beta = teta[1]
#     # 4 p choice options, calculate specific p and add to the list of p choices
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 's1':
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s1_d) / (np.exp(beta * q_s1_d) + np.exp(beta * q_s1_r)))
#                 q_s1_d = q_s1_d + alpha * (reward - q_s1_d)
#                 q_s2_r = q_s2_r + alpha * (reward - q_s2_r) #the opposite strategy of the opposite actor, generalization of learned on the other person
#             else:
#                 p_choice_list.append(np.exp(beta * q_s1_r) / (np.exp(beta * q_s1_r) + np.exp(beta * q_s1_d)))
#                 q_s1_r = q_s1_r + alpha * (reward - q_s1_r)
#                 q_s2_d = q_s2_d + alpha * (reward - q_s2_d)  # the opposite strategy of the opposite actor, generalization of learned on the other person
#
#         else:  # s2
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s2_d) / (np.exp(beta * q_s2_d) + np.exp(beta * q_s2_r)))
#                 q_s2_d = q_s2_d + alpha * (reward - q_s2_d)
#                 q_s1_r = q_s1_r + alpha * (reward - q_s1_r) #the opposite strategy of the opposite actor, generalization of learned on the other person
#             else:
#                 p_choice_list.append(np.exp(beta * q_s2_r) / (np.exp(beta * q_s2_r) + np.exp(beta * q_s2_d)))
#                 q_s2_r = q_s2_r + alpha * (reward - q_s2_r)
#                 q_s1_d = q_s1_d + alpha * (reward - q_s1_d) #the opposite strategy of the opposite actor, generalization of learned on the other person
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
# Model M2AB2 actors are independent from each other, strategies are independent from each other
# teta = {alpha,beta} => {learning rate, 1/T (noise)}
# def logLikelihoodStressActorIndTwoArmB(teta, data):
#     # s1 = stress actor 1, s2 = stress actor 2, d=distraction, r=reappraisal
#     q_s1_d = 0.5
#     q_s1_r = 0.5
#     q_s2_d = 0.5
#     q_s2_r = 0.5
#     alpha = teta[0]
#     beta = teta[1]
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 's1':
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s1_d) / (np.exp(beta * q_s1_d) + np.exp(beta * q_s1_r)))
#                 q_s1_d = q_s1_d + alpha * (reward - q_s1_d)
#             else:
#                 p_choice_list.append(np.exp(beta * q_s1_r) / (np.exp(beta * q_s1_r) + np.exp(beta * q_s1_d)))
#                 q_s1_r = q_s1_r + alpha * (reward - q_s1_r)
#         else:  # s2
#             if row['selected_choice'] == 'dist':
#                 p_choice_list.append(np.exp(beta * q_s2_d) / (np.exp(beta * q_s2_d) + np.exp(beta * q_s2_r)))
#                 q_s2_d = q_s2_d + alpha * (reward - q_s2_d)
#             else:
#                 p_choice_list.append(np.exp(beta * q_s2_r) / (np.exp(beta * q_s2_r) + np.exp(beta * q_s2_d)))
#                 q_s2_r = q_s2_r + alpha * (reward - q_s2_r)
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL

# teta = {alpha,beta} => {learning rate, 1/T (noise)}
# def logLikelihoodMeal(teta, data):
#     # m2 = meal actor 2 (shay), m1 = meal actor 1 (tal), sw=sweet, sa=salty
#     q_m2_sw = 0
#     q_m2_sa = 0
#     q_m1_sw = 0
#     q_m1_sa = 0
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 'm2':
#             if row['selected_choice'] == 'sweet':
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_m2_sw) / (np.exp(teta[1] * q_m2_sw) + np.exp(teta[1] * q_m2_sa)))
#                 q_m2_sw = q_m2_sw + teta[0] * (reward - q_m2_sw)
#             else:
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_m2_sa) / (np.exp(teta[1] * q_m2_sa) + np.exp(teta[1] * q_m2_sw)))
#                 q_m2_sa = q_m2_sa + teta[0] * (reward - q_m2_sa)
#         else:  # m1
#             if row['selected_choice'] == 'sweet':
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_m1_sw) / (np.exp(teta[1] * q_m1_sw) + np.exp(teta[1] * q_m1_sa)))
#                 q_m1_sw = q_m1_sw + teta[0] * (reward - q_m1_sw)
#             else:
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_m1_sa) / (np.exp(teta[1] * q_m1_sa) + np.exp(teta[1] * q_m1_sw)))
#                 q_m1_sa = q_m1_sa + teta[0] * (reward - q_m1_sa)
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
#
#
# def logLikelihoodRoom(teta, data):
#     # r1 = room 1 (white), r2 = room 2 (green), wo=wood_closet, bl=blue_closet
#     q_r1_wo = 0
#     q_r1_bl = 0
#     q_r2_wo = 0
#     q_r2_bl = 0
#     # 4 p choice options, calculate specific p and add to the list of p choices, only for people
#     # update Qs
#     p_choice_list = []
#     for index, row in data.iterrows():
#         reward = row['reward']
#         if row['actor'] == 'r1':
#             if row['selected_choice'] == 'wood_closet':
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_r1_wo) / (np.exp(teta[1] * q_r1_wo) + np.exp(teta[1] * q_r1_bl)))
#                 q_r1_wo = q_r1_wo + teta[0] * (reward - q_r1_wo)
#             else:
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_r1_bl) / (np.exp(teta[1] * q_r1_bl) + np.exp(teta[1] * q_r1_wo)))
#                 q_r1_bl = q_r1_bl + teta[0] * (reward - q_r1_bl)
#         else:  # r2 green room
#             if row['selected_choice'] == 'wood_closet':
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_r2_wo) / (np.exp(teta[1] * q_r2_wo) + np.exp(teta[1] * q_r2_bl)))
#                 q_r2_wo = q_r2_wo + teta[0] * (reward - q_r2_wo)
#             else:  # blue_closet
#                 p_choice_list.append(
#                     np.exp(teta[1] * q_r2_bl) / (np.exp(teta[1] * q_r2_bl) + np.exp(teta[1] * q_r2_wo)))
#                 q_r2_bl = q_r2_bl + teta[0] * (reward - q_r2_bl)
#         # sum over the log of the list of probabilities
#     minus_ll = -np.log(p_choice_list).sum()
#     return minus_ll  # return -LL
# sort the dataframe
# focused_df.sort_values(by=['actor'])
# # set the index to be this and don't drop
# focused_df.set_index(keys=['actor'], drop=False, inplace=True)
# # get a list of names
# names = focused_df['actor'].unique().tolist()
# # now we can perform a lookup on a 'view' of the dataframe
# s1_df = focused_df.loc[focused_df.actor == 's1']
# s2_df = focused_df.loc[focused_df.actor == 's2']
# m1_df = focused_df.loc[focused_df.actor == 'm1']
# m2_df = focused_df.loc[focused_df.actor == 'm2']
# r1_df = focused_df.loc[focused_df.actor == 'r1']
# r2_df = focused_df.loc[focused_df.actor == 'r2']
#
# # fill prev reward based on actor for all
# s1_df.prev_reward = s1_df.reward.shift(1)
# s2_df.prev_reward = s2_df.reward.shift(1)
# m1_df.prev_reward = m1_df.reward.shift(1)
# m2_df.prev_reward = m2_df.reward.shift(1)
# r1_df.prev_reward = r1_df.reward.shift(1)
# r2_df.prev_reward = r2_df.reward.shift(1)
# # focused_df = pd.concat([s1_df,s2_df,m1_df,m2_df,r1_df,r2_df])
# # focused_df.fillna(0)
# resultStressTwoAlpha = opt.minimize(fun=logLikelihoodStressTwoAlpha, x0=xZeroTwoAlpha,
    #      args=focused_df.loc[focused_df['actor'].isin(actorsStress)],
    #      bounds=bndsTwoAlpha)
    # resultMeal = opt.minimize(fun=logLikelihoodMeal, x0=xZero,
    # args=focused_df.loc[focused_df['actor'].isin(actorsMeal)], bounds=bnds)
    # resultMealTwoAlpha = opt.minimize(fun=logLikelihoodMealTwoAlpha, x0=xZeroTwoAlpha,args=focused_df.loc[focused_df['actor'].isin(actorsMeal)], bounds=bndsTwoAlpha)
    # resultRoom = opt.minimize(fun=logLikelihoodRoom, x0=xZero,
    #     args=focused_df.loc[focused_df['actor'].isin(actorsRoom)], bounds=bnds)
    # resultRoomTwoAlpha = opt.minimize(fun=logLikelihoodRoomTwoAlpha, x0=xZeroTwoAlpha, args=focused_df.loc[focused_df['actor'].isin(actorsRoom)], bounds=bndsTwoAlpha)
# likelihood ratio - are they really nested models? we didn't add any params.. TODO
# sLR_act_dep_no_act = 2*(optMinDf['sLL_act_dep']- optMinDf['sLL_no_act'])
# chi-squared distribution with df=1
# print(sLR_act_dep_no_act)
# print("sLR significance dep vs no act is {} :".format(stats.chi2.sf(sLR_act_dep_no_act,1)))

# sLR_act_ind_no_act = 2*(optMinDf['sLL_act_ind']- optMinDf['sLL_no_act'])
# chi-squared distribution with df=1
# print(sLR_act_ind_no_act)
# print("sLR significance ind vs no act is {} :".format(stats.chi2.sf(sLR_act_ind_no_act,1)))

# sLR_act_ind_dep = 2*(optMinDf['sLL_act_ind']- optMinDf['sLL_act_dep']) #ll specific - ll general
# chi-squared distribution with df=1
# print(sLR_act_ind_dep)
# print("sLR significance ind vs dep is {} :".format(stats.chi2.sf(sLR_act_ind_dep,1)))

# likelihood ratio stress
# fLR = 2*(optMinDf['mLL'].sum()- optMinDf['mLL2A'].sum())
# chi-squared distribution with df=1
# print(fLR)
# print("mLR significance is {} :".format(stats.chi2.sf(fLR,1)))

# likelihood ratio stress
# rLR = 2 * (optMinDf['rLL'].sum() - optMinDf['rLL2A'].sum())
# chi-squared distribution with df=1
# print(rLR)
# print("rLR significance is {} :".format(stats.chi2.sf(rLR, 1)))
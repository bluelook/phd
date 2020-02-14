import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib as mpl  # plots
import matplotlib.pyplot as plt  # plots
import os  # OS operations
# import scipy.stats.stats as stat
import scipy.optimize as opt
import scipy.stats as stats

# agg backend is used to create plot as a .png file
mpl.use('agg')

# teta = {alpha0, alpha1, beta} => {learning rate dist, learning rate reap, 1/T (noise)}
def logLikelihoodStressTwoAlpha(teta, data):
    # s1 = stress actor 1, s2 = stress actor 2, d=distraction, r=reappraisal
    q_s1_d = 0
    q_s1_r = 0
    q_s2_d = 0
    q_s2_r = 0
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == 's1':
            if row['selected_choice'] == 'dist':
                p_choice_list.append(np.exp(teta[2] * q_s1_d) / (np.exp(teta[2] * q_s1_d) + np.exp(teta[2] * q_s1_r)))
                q_s1_d = q_s1_d + teta[0] * (reward - q_s1_d)
            else:  # reap
                p_choice_list.append(np.exp(teta[2] * q_s1_r) / (np.exp(teta[2] * q_s1_r) + np.exp(teta[2] * q_s1_d)))
                q_s1_r = q_s1_r + teta[1] * (reward - q_s1_r)
        else:  # stress actor 2
            if row['selected_choice'] == 'dist':
                p_choice_list.append(np.exp(teta[2] * q_s2_d) / (np.exp(teta[2] * q_s2_d) + np.exp(teta[2] * q_s2_r)))
                q_s2_d = q_s2_d + teta[0] * (reward - q_s2_d)
            else:  # reap
                p_choice_list.append(np.exp(teta[2] * q_s2_r) / (np.exp(teta[2] * q_s2_r) + np.exp(teta[2] * q_s2_d)))
                q_s2_r = q_s2_r + teta[1] * (reward - q_s2_r)
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL


# teta = {alpha0, alpha1, beta} => {learning rate sweet, learning rate salty, 1/T (noise)}
def logLikelihoodMealTwoAlpha(teta, data):
    # m2 = meal actor 2, m1 = meal actor 1, sw=sweet, sa=salty
    q_m2_sw = 0
    q_m2_sa = 0
    q_m1_sw = 0
    q_m1_sa = 0
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == 'm2':
            if row['selected_choice'] == 'sweet':
                p_choice_list.append(np.exp(teta[2] * q_m2_sw) / (np.exp(teta[2] * q_m2_sw) + np.exp(teta[2] * q_m2_sa)))
                q_m2_sw = q_m2_sw + teta[0] * (reward - q_m2_sw)
            else:
                p_choice_list.append(np.exp(teta[2] * q_m2_sa) / (np.exp(teta[2] * q_m2_sa) + np.exp(teta[2] * q_m2_sw)))
                q_m2_sa = q_m2_sa + teta[1] * (reward - q_m2_sa)
        else:  # meal actor 1
            if row['selected_choice'] == 'sweet':
                p_choice_list.append(np.exp(teta[2] * q_m1_sw) / (np.exp(teta[2] * q_m1_sw) + np.exp(teta[2] * q_m1_sa)))
                q_m1_sw = q_m1_sw + teta[0] * (reward - q_m1_sw)
            else:
                p_choice_list.append(np.exp(teta[2] * q_m1_sa) / (np.exp(teta[2] * q_m1_sa) + np.exp(teta[2] * q_m1_sw)))
                q_m1_sa = q_m1_sa + teta[1] * (reward - q_m1_sa)
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL


# teta = {alpha0, alpha1, beta} => {learning rate wood closet, learning rate blue closet, 1/T (noise)}
def logLikelihoodRoomTwoAlpha(teta, data):
    # r1 = room 1 (white), r2 = room2 (green), wo=wood_closet, bl=blue_closet
    q_r1_wo = 0
    q_r1_bl = 0
    q_r2_wo = 0
    q_r2_bl = 0
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == 'r1':
            if row['selected_choice'] == 'wood_closet':
                p_choice_list.append(np.exp(teta[2] * q_r1_wo) / (np.exp(teta[2] * q_r1_wo) + np.exp(teta[2] * q_r1_bl)))
                q_r1_wo = q_r1_wo + teta[0] * (reward - q_r1_wo)
            else:
                p_choice_list.append(np.exp(teta[2] * q_r1_bl) / (np.exp(teta[2] * q_r1_bl) + np.exp(teta[2] * q_r1_wo)))
                q_r1_bl = q_r1_bl + teta[1] * (reward - q_r1_bl)
        else:  # r2 green room
            if row['selected_choice'] == 'wood_closet':
                p_choice_list.append(np.exp(teta[2] * q_r2_wo) / (np.exp(teta[2] * q_r2_wo) + np.exp(teta[2] * q_r2_bl)))
                q_r2_wo = q_r2_wo + teta[0] * (reward - q_r2_wo)
            else:  # blue_closet
                p_choice_list.append(np.exp(teta[2] * q_r2_bl) / (np.exp(teta[2] * q_r2_bl) + np.exp(teta[2] * q_r2_wo)))
                q_r2_bl = q_r2_bl + teta[1] * (reward - q_r2_bl)
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL


# teta = {alpha,beta} => {learning rate, 1/T (noise)}
def logLikelihoodStress(teta, data):
    # s1 = stress actor 1(ben), s2 = stress actor 2(gal), d=distraction, r=reappraisal
    q_s1_d = 0
    q_s1_r = 0
    q_s2_d = 0
    q_s2_r = 0
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == 's1':
            if row['selected_choice'] == 'dist':
                p_choice_list.append(np.exp(teta[1] * q_s1_d) / (np.exp(teta[1] * q_s1_d) + np.exp(teta[1] * q_s1_r)))
                q_s1_d = q_s1_d + teta[0] * (reward - q_s1_d)
            else:
                p_choice_list.append(np.exp(teta[1] * q_s1_r) / (np.exp(teta[1] * q_s1_r) + np.exp(teta[1] * q_s1_d)))
                q_s1_r = q_s1_r + teta[0] * (reward - q_s1_r)
        else:  # s2
            if row['selected_choice'] == 'dist':
                p_choice_list.append(np.exp(teta[1] * q_s2_d) / (np.exp(teta[1] * q_s2_d) + np.exp(teta[1] * q_s2_r)))
                q_s2_d = q_s2_d + teta[0] * (reward - q_s2_d)
            else:
                p_choice_list.append(np.exp(teta[1] * q_s2_r) / (np.exp(teta[1] * q_s2_r) + np.exp(teta[1] * q_s2_d)))
                q_s2_r = q_s2_r + teta[0] * (reward - q_s2_r)
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL


# teta = {alpha,beta} => {learning rate, 1/T (noise)}
def logLikelihoodMeal(teta, data):
    # m2 = meal actor 2 (shay), m1 = meal actor 1 (tal), sw=sweet, sa=salty
    q_m2_sw = 0
    q_m2_sa = 0
    q_m1_sw = 0
    q_m1_sa = 0
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == 'm2':
            if row['selected_choice'] == 'sweet':
                p_choice_list.append(np.exp(teta[1] * q_m2_sw) / (np.exp(teta[1] * q_m2_sw) + np.exp(teta[1] * q_m2_sa)))
                q_m2_sw = q_m2_sw + teta[0] * (reward - q_m2_sw)
            else:
                p_choice_list.append(np.exp(teta[1] * q_m2_sa) / (np.exp(teta[1] * q_m2_sa) + np.exp(teta[1] * q_m2_sw)))
                q_m2_sa = q_m2_sa + teta[0] * (reward - q_m2_sa)
        else:  # m1
            if row['selected_choice'] == 'sweet':
                p_choice_list.append(np.exp(teta[1] * q_m1_sw) / (np.exp(teta[1] * q_m1_sw) + np.exp(teta[1] * q_m1_sa)))
                q_m1_sw = q_m1_sw + teta[0] * (reward - q_m1_sw)
            else:
                p_choice_list.append(np.exp(teta[1] * q_m1_sa) / (np.exp(teta[1] * q_m1_sa) + np.exp(teta[1] * q_m1_sw)))
                q_m1_sa = q_m1_sa + teta[0] * (reward - q_m1_sa)
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL


def logLikelihoodRoom(teta, data):
    # r1 = room 1 (white), r2 = room 2 (green), wo=wood_closet, bl=blue_closet
    q_r1_wo = 0
    q_r1_bl = 0
    q_r2_wo = 0
    q_r2_bl = 0
    # 4 p choice options, calculate specific p and add to the list of p choices, only for people
    # update Qs
    p_choice_list = []
    for index, row in data.iterrows():
        reward = row['reward']
        if row['actor'] == 'r1':
            if row['selected_choice'] == 'wood_closet':
                p_choice_list.append(np.exp(teta[1] * q_r1_wo) / (np.exp(teta[1] * q_r1_wo) + np.exp(teta[1] * q_r1_bl)))
                q_r1_wo = q_r1_wo + teta[0] * (reward - q_r1_wo)
            else:
                p_choice_list.append(np.exp(teta[1] * q_r1_bl) / (np.exp(teta[1] * q_r1_bl) + np.exp(teta[1] * q_r1_wo)))
                q_r1_bl = q_r1_bl + teta[0] * (reward - q_r1_bl)
        else:  # r2 green room
            if row['selected_choice'] == 'wood_closet':
                p_choice_list.append(np.exp(teta[1] * q_r2_wo) / (np.exp(teta[1] * q_r2_wo) + np.exp(teta[1] * q_r2_bl)))
                q_r2_wo = q_r2_wo + teta[0] * (reward - q_r2_wo)
            else:  # blue_closet
                p_choice_list.append(np.exp(teta[1] * q_r2_bl) / (np.exp(teta[1] * q_r2_bl) + np.exp(teta[1] * q_r2_wo)))
                q_r2_bl = q_r2_bl + teta[0] * (reward - q_r2_bl)
        # sum over the log of the list of probabilities
    minus_ll = -np.log(p_choice_list).sum()
    return minus_ll  # return -LL


# read all the csv files of participants into a list
folderName = './outputs'
plotsFolderName = './plots'
csvList = os.listdir(folderName)
# iterate over all files in the folder(over all participants)
optMinDf = pd.DataFrame(
    columns=['participantID', 'sAlpha', 'sBeta', 'sLL', 'sP', 'mAlpha', 'mBeta', 'mLL', 'mP', 'rAlpha', 'rBeta', 'rLL',
             'rP', 'sAlphaD', 'sAlphaR', 'sBeta2A', 'sLL2A', 'sP2A', 'mAlphaSW', 'mAlphaSA', 'mBeta2A', 'mLL2A', 'mP2A',
             'rAlphaW', 'rAlphaB', 'rBeta2A', 'rLL2A', 'rP2A'])
for fileName in csvList:
    #participant = fileName.split('subject-')[1].split('.')[0]
    # read the files into df
    df = pd.read_csv(folderName + "/" + fileName)
    # pick the needed columns for focused data frame
    focused_df = df[0:120].copy()[['subject_nr','actor', 'correct', 'preferred']]
    # recode additional columns - Reward: 1 = reward, -1 = no reward
    focused_df['reward'] = np.where(focused_df.correct == 1, 1, -1)
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
    xZero = [0.5, 1]
    xZeroTwoAlpha = [0.5, 0.5, 1]
    bnds = ((0.01, 0.99), (1, None))
    bndsTwoAlpha = ((0.01, 0.99), (0.01, 0.99), (1, None))
    # pass only the rows related to specific actors (blocks)
    actorsStress = ['s2', 's1']
    actorsMeal = ['m2', 'm1']
    actorsRoom = ['r2', 'r1']
    resultStress = opt.minimize(fun=logLikelihoodStress, x0=xZero,
                                args=focused_df.loc[focused_df['actor'].isin(actorsStress)], bounds=bnds)
    resultStressTwoAlpha = opt.minimize(fun=logLikelihoodStressTwoAlpha, x0=xZeroTwoAlpha,
                                        args=focused_df.loc[focused_df['actor'].isin(actorsStress)],
                                        bounds=bndsTwoAlpha)
    resultMeal = opt.minimize(fun=logLikelihoodMeal, x0=xZero,
                              args=focused_df.loc[focused_df['actor'].isin(actorsMeal)], bounds=bnds)
    resultMealTwoAlpha = opt.minimize(fun=logLikelihoodMealTwoAlpha, x0=xZeroTwoAlpha,
                                      args=focused_df.loc[focused_df['actor'].isin(actorsMeal)], bounds=bndsTwoAlpha)
    resultRoom = opt.minimize(fun=logLikelihoodRoom, x0=xZero,
                              args=focused_df.loc[focused_df['actor'].isin(actorsRoom)], bounds=bnds)
    resultRoomTwoAlpha = opt.minimize(fun=logLikelihoodRoomTwoAlpha, x0=xZeroTwoAlpha,
                                      args=focused_df.loc[focused_df['actor'].isin(actorsRoom)], bounds=bndsTwoAlpha)

    optMinDf = optMinDf.append(
        {'participantID': focused_df.subject_nr[1], 'sAlpha': resultStress.x[0], 'sBeta': resultStress.x[1], 'sLL': resultStress.fun,
         'sP': np.exp(-resultStress.fun / focused_df.loc[focused_df['actor'].isin(actorsStress)].shape[0]),
         'mAlpha': resultMeal.x[0], 'mBeta': resultMeal.x[1], 'mLL': resultMeal.fun,
         'mP': np.exp(-resultMeal.fun / focused_df.loc[focused_df['actor'].isin(actorsMeal)].shape[0]),
         'rAlpha': resultRoom.x[0], 'rBeta': resultRoom.x[1], 'rLL': resultRoom.fun,
         'rP': np.exp(-resultRoom.fun / focused_df.loc[focused_df['actor'].isin(actorsRoom)].shape[0]),
         'sAlphaD': resultStressTwoAlpha.x[0], 'sAlphaR': resultStressTwoAlpha.x[1],
         'sBeta2A': resultStressTwoAlpha.x[2], 'sLL2A': resultStressTwoAlpha.fun,
         'sP2A': np.exp(-resultStressTwoAlpha.fun / focused_df.loc[focused_df['actor'].isin(actorsStress)].shape[0]),
         'mAlphaSW': resultMealTwoAlpha.x[0], 'mAlphaSA': resultMealTwoAlpha.x[1], 'mBeta2A': resultMealTwoAlpha.x[2],
         'mLL2A': resultMealTwoAlpha.fun,
         'mP2A': np.exp(-resultMealTwoAlpha.fun / focused_df.loc[focused_df['actor'].isin(actorsMeal)].shape[0]),
         'rAlphaW': resultRoomTwoAlpha.x[0], 'rAlphaB': resultRoomTwoAlpha.x[1], 'rBeta2A': resultRoomTwoAlpha.x[2],
         'rLL2A': resultRoomTwoAlpha.fun,
         'rP2A': np.exp(-resultRoomTwoAlpha.fun / focused_df.loc[focused_df['actor'].isin(actorsMeal)].shape[0])},
        ignore_index=True, sort=False)

optMinDf.to_csv(plotsFolderName + '/optimizationResultAll.csv', index=False)
alpha_to_plot = [optMinDf.sAlpha, optMinDf.mAlpha, optMinDf.rAlpha]
beta_to_plot = [optMinDf.sBeta, optMinDf.mBeta, optMinDf.rBeta]
ll_to_plot = [optMinDf.sLL, optMinDf.mLL, optMinDf.rLL]

#likelihood ratio stress
sLR = 2*(optMinDf['sLL'].sum()- optMinDf['sLL2A'].sum())
#chi-squared distribution with df=1
print(sLR)
print("sLR significance is {} :".format(stats.chi2.sf(sLR,1)))

#likelihood ratio stress
fLR = 2*(optMinDf['mLL'].sum()- optMinDf['mLL2A'].sum())
#chi-squared distribution with df=1
print(fLR)
print("mLR significance is {} :".format(stats.chi2.sf(fLR,1)))

# likelihood ratio stress
rLR = 2 * (optMinDf['rLL'].sum() - optMinDf['rLL2A'].sum())
# chi-squared distribution with df=1
print(rLR)
print("rLR significance is {} :".format(stats.chi2.sf(rLR, 1)))

#boxplotting
def boxPlotPng(data, labels, param):
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
    fig.savefig(plotsFolderName +'/'+ param + '_box_plot.png', bbox_inches='tight')


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


boxPlotPng(alpha_to_plot, ['\u03B1 stress', '\u03B1 meal', '\u03B1 room'], '\u03B1')
boxPlotPng(beta_to_plot, ['\u03B2 stress', '\u03B2 meal', '\u03B2 room'], '\u03B2')
boxPlotPng(ll_to_plot, ['LL stress', 'LL meal', 'LL rooms'], 'll')

#plot alphas relation
#people
ax1 = optMinDf.plot(kind='scatter', x='sAlphaD', y='sAlphaR', color='red')
add_identity(ax1)
ax1.set_xlabel('\u03B1 distraction')
ax1.set_ylabel('\u03B1 reappraisal')
optMinDf[['sAlphaD','sAlphaR','participantID']].apply(lambda row: ax1.text(*row),axis=1);
plt.savefig(plotsFolderName + '/stress alphas.png', dpi=300)
plt.clf()

#for i,name in enumerate(stat_df['name']):
 #   x = stat_df.loc[i,'lr']
  #  y = stat_df.loc[i,'PT']
   # plt.scatter(x, y, marker='o', color='red')
    #if type(name) is str:
     #   plt.text(x-0.03, y+0.5, name, fontsize=20, color='blue')


#meal
ax2 = optMinDf.plot(kind='scatter', x='mAlphaSW', y='mAlphaSA', color='red')
add_identity(ax2)
ax2.set_xlabel('\u03B1 sweet meal')
ax2.set_ylabel('\u03B1 salty meal')
optMinDf[['mAlphaSW','mAlphaSA','participantID']].apply(lambda row: ax2.text(*row),axis=1);
plt.savefig(plotsFolderName + '/meals alphas.png',dpi=300)
plt.clf()

#rooms
ax3 = optMinDf.plot(kind='scatter', x='rAlphaW', y='rAlphaB', color='red')
add_identity(ax3)
ax3.set_xlabel('\u03B1 wood cabinet')
ax3.set_ylabel('\u03B1 blue cabinet')
optMinDf[['rAlphaW','rAlphaB','participantID']].apply(lambda row: ax3.text(*row),axis=1);
plt.savefig(plotsFolderName + '/room alphas.png',dpi=300)
print("Successfully processed %s output files" % len(csvList))

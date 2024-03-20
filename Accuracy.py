#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:30:02 2022

@author: yuhan
"""


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 1, 1, 0)
Geodata["lu11_label"] = np.where( Geodata['LUCode11re'] == 1, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 1, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)

auc0 = roc_auc_score(Geodata["lu20_label"], Geodata["lu11_label"])
print('AUC: %.3f' % auc0)

auc01 = roc_auc_score(Geodata["lu11_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc01)



Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 2, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 2, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 3, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 3, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 4, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 4, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 5, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 5, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 6, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 6, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


ns_probs = [0 for _ in range(len( Geodata["lu20_label"] ))]

ns_auc = roc_auc_score(Geodata["lu20_label"], ns_probs)
lr_auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"] )

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))


Geodata['Percentile Rank'] = (Geodata.LUCpr0 - Geodata.LUCpr0.min()) / ( Geodata.LUCpr0.max() - Geodata.LUCpr0.min())

#Geodata.LUCpr0.rank(pct = True)
#ls_probs = 
ns_fpr, ns_tpr, _ = roc_curve(Geodata["lu20_label"], ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Geodata["lu20_label"], Geodata["lu_predict"])
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Null model')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Model results')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


import pandas
import numpy as np

df = pandas.read_pickle('data2012_bu.pkl')
sf = pandas.read_pickle('sig2012_bu.pkl')
sf = sf.drop("B_BKGCAT", axis=1)

# concat the two frames into one with a label
df["sig"] = 0
sf["sig"] = 1
fr = pandas.concat([df,sf], ignore_index=True)

# add some variables
for part in ["B","Kst","Kstb","Kp","Km","pip","pim"]:
  fr[part+"_ETA"] = np.arccosh( fr[part+"_P"] / fr[part+"_PT"] )
fr["max_trk_PT"] = fr[ ["Kp_PT","Km_PT","pip_PT","pim_PT"] ].max(axis=1)
fr["min_trk_PT"] = fr[ ["Kp_PT","Km_PT","pip_PT","pim_PT"] ].min(axis=1)

# define training vars
train_vars = ["B_PT","Kst_PT","Kstb_PT","max_trk_PT","min_trk_PT"]

# make training sets
# X is an array with the number of columns corresponding to the input variables
# y is an array with one column which labels whether "signal" or "background"
X, y = fr[ train_vars ].to_numpy(), fr["sig"].to_numpy()

# do mva stuff
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=21)

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_curve

classifiers = [ AdaBoostClassifier(n_estimators=50) ,
                GradientBoostingClassifier(n_estimators=100),
                #MLPClassifier(),
                #SVC( C=1.0, kernel='rbf', tol=0.001, gamma=0.005, probability=True )
              ]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2,figsize=(12,8))

for i, mva in enumerate(classifiers):

  # this trains the algorithm
  mva.fit( X_train, y_train )

  # this gives you the average score
  score = mva.score( X_test, y_test )

  # this gives you the decision (i.e. the BDT output evaluation)
  # note not all classifiers give a decision function
  # some will only give a predicted probability or some other score
  out_sig_train = mva.decision_function( X_train[y_train==1] ).ravel()
  out_sig_test  = mva.decision_function( X_test[y_test==1] ).ravel()
  out_bkg_train = mva.decision_function( X_train[y_train==0] ).ravel()
  out_bkg_test  = mva.decision_function( X_test[y_test==0] ).ravel()

  #out_sig_train = mva.predict_proba( X_train[y_train==1] )
  #out_sig_test  = mva.predict_proba( X_test[y_test==1] )
  #out_bkg_train = mva.predict_proba( X_train[y_train==0] )
  #out_bkg_test  = mva.predict_proba( X_test[y_test==0] )

  # now make histograms of the output
  rng = (min( min( out_sig_train ), min(out_sig_test), min(out_bkg_train), min(out_bkg_test) ) , max( max( out_sig_train), max(out_sig_test), max(out_bkg_train), max(out_bkg_test) ) )
  bins = 50

  hs_tr = np.histogram( out_sig_train, bins=bins, range=rng )
  hb_tr = np.histogram( out_bkg_train, bins=bins, range=rng )
  hs_ts = np.histogram( out_sig_test , bins=bins, range=rng )
  hb_ts = np.histogram( out_bkg_test , bins=bins, range=rng )

  bin_edges = hs_tr[1]
  bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
  bin_widths  = (bin_edges[1:] - bin_edges[:-1])

  ax[i,0].hist( out_sig_train, bins=bins, range=rng, facecolor='red' , lw=0, alpha=0.5, label='S (Train)' )
  ax[i,0].hist( out_bkg_train, bins=bins, range=rng, facecolor='blue', lw=0, alpha=0.5, label='B (Train)' )
  ax[i,0].errorbar( bin_centers, hs_ts[0], yerr=hs_ts[0]**0.5, xerr=None, ecolor='red', c='red', fmt='o', label='S (Test)')
  ax[i,0].errorbar( bin_centers, hb_ts[0], yerr=hb_ts[0]**0.5, xerr=None, ecolor='blue' , c='blue' , fmt='o', label='B (Test)')
  ax[i,0].set_xlabel('MVA output score')
  ax[i,0].legend()

  # and now make a ROC curve
  s_test_prob  = mva.predict_proba( X_test )[:,1]
  fpr, tpr, thresholds = roc_curve( y_test, s_test_prob )
  ax[i,1].plot( tpr, 1-fpr )
  ax[i,1].set_xlabel('Signal efficiency')
  ax[i,1].set_ylabel('Background rejection')


fig.tight_layout()
fig.savefig('dummy_train.pdf')
plt.show()

import os, sys
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-l','--load', default=False,action="store_true",help='Load fit result')
parser.add_argument('-M','--Run1', default=False,action="store_true",help='Old Run 1 dataset')
parser.add_argument('-L','--Luke', default=False,action="store_true",help='Luke\'s files')
parser.add_argument('-R','--Ryan', default=False,action="store_true",help='Ryan\'s files')
parser.add_argument('-y','--year', default='2012' , choices=['2012','2018'], help='Year to run on')
parser.add_argument('-c','--classifier', default='BDTD' , choices=['BDTD','BDTG', 'MLP', 'MLPBFGS'], help='Which combinatorial background classifier to cut on')
parser.add_argument('-o','--optimise', default='Bs' , choices=['Bs','Bd', 'Bs+Bd'], help='Which peak to optimise comb cut')
opts = parser.parse_args()

if opts.Run1 and opts.year!='2012':
  sys.exit('Only 2012 available for Run1')

import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import expon
from models import dcb

from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

read_root=False

#dm = 87.42
mrange = (5000,5800)

def read(files,cut=None,vars=["B_M"], flat=False):

  df = uproot.concatenate( files, vars, cut, library="pd" )
  #print(len(df))
  if flat: return df[vars[0]].to_numpy()

  return df

if read_root or not os.path.exists('data.npz'):
  #data = read( ["DTT_2012_Down_KstKst_RealData.root","DTT_2012_Up_KstKst_RealData.root"], flat=True )
  data = read( ["AnalysisOutTrimmed.root"], vars=["B_s0_DTF_B_s0_M"], flat=True )
  np.savez('data', data=data)

### LOAD MC FIT PARS ###
def load_pars(fname):
  f = open(fname)
  lines = f.readlines()
  res = {}
  for l in lines:
    res[l.split()[0]] = float(l.split()[1])
  return res

bskstkst_pars = load_pars('bskstkst.log')
bdphikst_pars = load_pars('bdphikst.log')
bdrhokst_pars = load_pars('rhokst.log')
lbkppp_pars = load_pars('lb2kppp.log')
lbpppp_pars = load_pars('lb2pppp.log')

### load appropriate files
data = None
if opts.Luke:
  if opts.year == '2012':
    df = pd.read_pickle('lukes/2012_KstKst_RealData_scored.pkl')
  if opts.year == '2018':
    df = pd.read_pickle('lukes/2018_KstKst_RealData_scored.pkl')

  # place cuts
  data = df[ (df[b'comb_score']>0.6) & (df[b'sig_score']>2) ][b'B_M'].to_numpy()

if opts.Ryan:
  if opts.year == '2012':
    if opts.classifier == 'BDTD':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_BDTD_Score>0.0013) & (PID_Combined_Var_BDTG_Score>0.85)', vars=['B_M'],  flat=True) #0.0013
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_BDTD_Score>0.085) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.0784
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_BDTD_Score>0.0134) & (PID_Combined_Var_BDTG_Score>0.4)', vars=['B_M'],  flat=True) #0.0134

    if opts.classifier == 'BDTG':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_BDTG_Score>0.081) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.081
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_BDTG_Score>0.82) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.82
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_BDTG_Score>0.396) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.396

    if opts.classifier == 'MLP':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_MLP_Score>0.64) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.64
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_MLP_Score>0.93755) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.93755
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_MLP_Score>0.706) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.706

    if opts.classifier == 'MLPBFGS':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_MLPBFGS_Score>0.55) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.55
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_MLPBFGS_Score>0.95) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.90835
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2012.root'], cut='(Comb_Bkg_MLPBFGS_Score>0.7418) & (PID_Combined_Var_BDTG_Score>0.5)', vars=['B_M'],  flat=True) #0.7418

  if opts.year == '2018':
    if opts.classifier == 'BDTD':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_BDTD_Score>0.0077) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.0077
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_BDTD_Score>0.0444) & (PID_Combined_Var_BDTG_Score>0.7)', vars=['B_M'],  flat=True) #0.0444
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_BDTD_Score>0.0204) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.0204

    if opts.classifier == 'BDTG':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_BDTG_Score>0.25) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.25
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_BDTG_Score>0.8524) & (PID_Combined_Var_BDTG_Score>0.8)', vars=['B_M'],  flat=True) #0.8524
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_BDTG_Score>0.4245) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.4245

    if opts.classifier == 'MLP':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_MLP_Score>0.66) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.66
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_MLP_Score>0.9382) & (PID_Combined_Var_BDTG_Score>0.8)', vars=['B_M'],  flat=True) #0.9382
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_MLP_Score>0.7978) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.7978

    if opts.classifier == 'MLPBFGS':
      if opts.optimise == 'Bs':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_MLPBFGS_Score>0.64) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.64
      if opts.optimise == 'Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_MLPBFGS_Score>0.9083) & (PID_Combined_Var_BDTG_Score>0.85)', vars=['B_M'],  flat=True) #0.9083
      if opts.optimise == 'Bs+Bd':
        data = read( files=['ryans/KstKst_RealData_2018.root'], cut='(Comb_Bkg_MLPBFGS_Score>0.7723) & (PID_Combined_Var_BDTG_Score>0.9)', vars=['B_M'],  flat=True) #0.7723

## check
if data is None: sys.exit('Nothing to do')
print('Dataset loaded with', len(data), 'events')

def pdf(x, nbs, nbd, nbkg, nbdphi, nrho, nlbk, nlbp, mu, sgl, sgr, dm, ss, lb, comps=None):

  if comps is None:
    comps = ['bs2kstkst','bd2kstkst','bd2phikst','bd2rhokst','lb2kppp','lb2pppp','combinatorial']

  tot = 0

  # signal bs2kstkst
  p = dict(bskstkst_pars)
  dmu = p['mur']-p['mul']
  p['mul'] = mu
  p['mur'] = mu + dmu
  p['sgl'] = sgl
  p['sgr'] = sgr
  p['rng'] = mrange

  bskstkst_pdf = lambda x: dcb(x,**p)
  if 'bs2kstkst' in comps:
    tot += nbs*bskstkst_pdf(x)

  # signal bd2kstkst
  pd = dict(bskstkst_pars)
  dmu = pd['mur']-p['mul']
  pd['mul'] = mu - dm
  pd['mur'] = mu - dm + dmu
  pd['sgl'] = sgl*ss
  pd['sgr'] = sgr*ss
  pd['rng'] = mrange

  bdkstkst_pdf = lambda x: dcb(x,**pd)
  if 'bd2kstkst' in comps:
    tot += nbd*bdkstkst_pdf(x)

  # background bd2phikst
  pb = dict(bdphikst_pars)
  pb['rng'] = mrange
  bdphikst_pdf = lambda x: dcb(x,**pb)
  if 'bd2phikst' in comps:
    tot += nbdphi*bdphikst_pdf(x)

  # background bd2rhokst
  pr = dict(bdrhokst_pars)
  pr['rng'] = mrange
  bdrhokst_pdf = lambda x: dcb(x,**pr)
  if 'bd2rhokst' in comps:
    tot += nrho*bdrhokst_pdf(x)

  # background lb2kppp
  plk = dict(lbkppp_pars)
  plk['rng'] = mrange
  lbkppp_pdf = lambda x: dcb(x,**plk)
  if 'lb2kppp' in comps:
    tot += nlbk*lbkppp_pdf(x)

  # background lbpkppp
  plp = dict(lbpppp_pars)
  mu = plp['mu']
  plp.pop('mu')
  plp['mul'] = mu
  plp['mur'] = mu
  plp['rng'] = mrange
  lbpppp_pdf = lambda x: dcb(x,**plp)
  if 'lb2pppp' in comps:
    tot += nlbp*lbpppp_pdf(x)

  # combinatorial
  exp_pdf = lambda x: expon(mrange[0],lb).pdf(x)

  if 'combinatorial' in comps:
    tot += nbkg*exp_pdf(x)

  return tot

def epdf(x, nbs, nbd, nbkg, nbdphi, nrho, nlbk, nlbp, mu, sgl, sgr, dm, ss, lb):
  return (nbs+nbd+nbkg+nbdphi+nrho+nlbk+nlbp, pdf(x, nbs, nbd, nbkg, nbdphi, nrho, nlbk, nlbp, mu, sgl, sgr, dm, ss, lb))

mi = Minuit( ExtendedUnbinnedNLL( data, epdf ),
             nbs = 0.2*len(data),
             nbd = 0.1*len(data),
             nbkg = 0.7*len(data),
             nbdphi = 0.05*len(data),
             nrho = 0.05*len(data),
             nlbk = 0.05*len(data),
             nlbp = 0.05*len(data),
             mu = 5350,
             sgl = 30,
             sgr = 30,
             dm = 87.32,
             ss = 1,
             lb = 200
            )

mi.limits['nbs'] = (0,len(data))
mi.limits['nbd'] = (0,len(data))
mi.limits['nbkg'] = (0,len(data))
mi.limits['nbdphi'] = (0,len(data))
mi.limits['nrho'] = (0,len(data))
mi.limits['nlbk'] = (0,len(data))
mi.limits['nlbp'] = (0,len(data))
mi.limits['mu'] = (5300,5400)
mi.limits['sgl'] = (0,50)
mi.limits['sgr'] = (0,50)
mi.limits['dm'] = (80,95)
mi.limits['ss'] = (0.5,1.5)
mi.limits['lb'] = (50,500)

if opts.load:
  print('Loading fit result from datavals.npy')
  vals = np.load('datavals.npy')
  mi.values = vals
else:
  print('Running minimisation')
  mi.migrad()
  mi.hesse()
  np.save('datavals.npy',mi.values)

print(mi.params)
print('Making plot')

# make the plot of the fit
fig, ax = plt.subplots(2,1,figsize=(8,8),sharex=True,gridspec_kw={'height_ratios': [3, 1]})

# histogram the data and plot it
nh, xe = np.histogram(data, range=mrange, bins=100)
cx = 0.5*(xe[1:]+xe[:-1])
ax[0].errorbar( cx, nh, nh**0.5, fmt='ko', capsize=2, label='Data' )

# each component drawing options
comps = { 'combinatorial' : { 'color': 'tab:red'   , 'alpha': 0.5, 'lw': 0, 'label': 'Combinatorial' },
          'lb2kppp'       : { 'color': 'tab:purple', 'alpha': 0.5, 'lw': 0, 'label': r'$\Lambda_{b}^0 \to p K^-\pi^+\pi^-$'},
          'lb2pppp'       : { 'color': 'tab:gray'  , 'alpha': 0.5, 'lw': 0, 'label': r'$\Lambda_{b}^0 \to p \pi^-\pi^+\pi^-$'},
          'bd2phikst'     : { 'color': 'tab:cyan'  , 'alpha': 0.5, 'lw': 0, 'label': r'$B^0 \to \phi K^{*0}$'},
          'bd2rhokst'     : { 'color': 'tab:orange', 'alpha': 0.5, 'lw': 0, 'label': r'$B^0 \to \rho K^{*0}$'},
          'bd2kstkst'     : { 'color': 'tab:green' , 'alpha': 0.5, 'lw': 0, 'label': r'$B^0$ signal'},
          'bs2kstkst'     : { 'color': 'tab:pink'  , 'alpha': 0.5, 'lw': 0, 'label': r'$B_s^0$ signal'}
        }

# now evaluate the pdfs of each component and plots them
N = (mrange[1]-mrange[0])/100
x = np.linspace(*mrange,200)

pdfs = {}
for comp in comps.keys():
  pdfs[comp] = N*pdf(x,*mi.values,comps=[comp])
tot     = N*pdf(x,*mi.values)

# plotting happens here with the "fill_between" option
lower = 0
upper = 0
for comp, options in comps.items():
  upper += pdfs[comp]
  ax[0].fill_between( x, lower, upper, **options )
  lower += pdfs[comp]

# plot the total fit model
ax[0].plot(x, tot, 'b-', label='Total PDF' )

# compute the pull between the data and the fit model
pull = (nh-N*pdf(cx,*mi.values))/(nh**0.5)

# plot box and line at zero
ax[1].fill_between(x, -1, 1, alpha=0.5, color='0.5')
ax[1].plot(x, np.zeros_like(x), 'b-')

# plot pull
ax[1].errorbar( cx, pull, np.ones_like(cx), capsize=2, fmt='ko' )
ylim = ax[1].get_ylim()

# symmetrise pull y-axis
y = max(abs(ylim[0]),abs(ylim[1]))
ax[1].set_ylim(-y,y)

# labels
ax[1].set_xlabel('$m(B_s^0)$ [MeV]')
ax[0].set_ylabel('Events / {0} MeV'.format( (mrange[1]-mrange[0]/100) ) )
ax[1].set_ylabel('Pull')
ax[0].legend()

fig.tight_layout()
if opts.year == '2012':
  if opts.classifier == 'BDTD':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2012_BDTD_Bs.pdf')
      fig.savefig('plots/datafit_2012_BDTD_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2012_BDTD_Bd.pdf')
      fig.savefig('plots/datafit_2012_BDTD_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2012_BDTD_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2012_BDTD_Bs+Bd.png')

  if opts.classifier == 'BDTG':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2012_BDTG_Bs.pdf')
      fig.savefig('plots/datafit_2012_BDTG_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2012_BDTG_Bd.pdf')
      fig.savefig('plots/datafit_2012_BDTG_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2012_BDTG_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2012_BDTG_Bs+Bd.png')

  if opts.classifier == 'MLP':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2012_MLP_Bs.pdf')
      fig.savefig('plots/datafit_2012_MLP_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2012_MLP_Bd.pdf')
      fig.savefig('plots/datafit_2012_MLP_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2012_MLP_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2012_MLP_Bs+Bd.png')

  if opts.classifier == 'MLPBFGS':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2012_MLPBFGS_Bs.pdf')
      fig.savefig('plots/datafit_2012_MLPBFGS_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2012_MLPBFGS_Bd.pdf')
      fig.savefig('plots/datafit_2012_MLPBFGS_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2012_MLPBFGS_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2012_MLPBFGS_Bs+Bd.png')

if opts.year == '2018':
  if opts.classifier == 'BDTD':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2018_BDTD_Bs.pdf')
      fig.savefig('plots/datafit_2018_BDTD_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2018_BDTD_Bd.pdf')
      fig.savefig('plots/datafit_2018_BDTD_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2018_BDTD_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2018_BDTD_Bs+Bd.png')

  if opts.classifier == 'BDTG':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2018_BDTG_Bs.pdf')
      fig.savefig('plots/datafit_2018_BDTG_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2018_BDTG_Bd.pdf')
      fig.savefig('plots/datafit_2018_BDTG_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2018_BDTG_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2018_BDTG_Bs+Bd.png')

  if opts.classifier == 'MLP':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2018_MLP_Bs.pdf')
      fig.savefig('plots/datafit_2018_MLP_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2018_MLP_Bd.pdf')
      fig.savefig('plots/datafit_2018_MLP_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2018_MLP_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2018_MLP_Bs+Bd.png')

  if opts.classifier == 'MLPBFGS':
    if opts.optimise == 'Bs':
      fig.savefig('plots/datafit_2018_MLPBFGS_Bs.pdf')
      fig.savefig('plots/datafit_2018_MLPBFGS_Bs.png')
    if opts.optimise == 'Bd':
      fig.savefig('plots/datafit_2018_MLPBFGS_Bd.pdf')
      fig.savefig('plots/datafit_2018_MLPBFGS_Bd.png')
    if opts.optimise == 'Bd+Bs':
      fig.savefig('plots/datafit_2018_MLPBFGS_Bs+Bd.pdf')
      fig.savefig('plots/datafit_2018_MLPBFGS_Bs+Bd.png')

plt.show()

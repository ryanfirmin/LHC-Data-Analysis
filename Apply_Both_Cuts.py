import ROOT as r

c = r.TCanvas()

data2018  = r.TChain("DecayTree")
data2018.AddFile("/home/ryan/anaconda3/envs/my_root_env/Project/Datasets/KstKst_RealData_2018.root")

def plot(save=None, bins=None, range=None):

  c.Clear()
  c.cd()

  leg = r.TLegend(0.15,0.65,0.4,0.85)
  leg.SetFillColor(0)
  leg.SetLineColor(0)

  hcfg = ''
  if bins is not None and range is not None:
    hcfg += '({0},{1},{2})'.format(bins,range[0],range[1])

  data2018.Draw('B_M'+'>>sighist1'+hcfg, '(Comb_Bkg_MLPBFGS_Score>0.9083) & (PID_Combined_Var_BDTG_Score>0.85)')
  dathist1 = r.gPad.GetPrimitive('sighist1')
  data2018.Draw('B_M'+'>>sighist2'+hcfg)
  dathist2 = r.gPad.GetPrimitive('sighist2')

  dathist1.SetLineColor(r.kRed)
  dathist2.SetLineColor(r.kBlue)

  dathist1.SetLineWidth(2)
  dathist2.SetLineWidth(2)

  dathist1.GetYaxis().SetRangeUser(0, 1.1*max(dathist1.GetMaximum(),dathist2.GetMaximum()))
  dathist2.GetYaxis().SetRangeUser(0, 1.1*max(dathist1.GetMaximum(),dathist2.GetMaximum()))

  dathist1.Draw("HIST")
  dathist2.Draw("HISTsame")

  leg.AddEntry( dathist1, "B_M after Cuts Applied", "L" )
  leg.AddEntry( dathist2, "B_M before Cuts Applied", "L" )
  leg.Draw()

  c.Update()
  c.Modified()

  if save is not None:
    c.Print(save)

plot(save='B_M_MLPBFGS_Cuts_Applied_2018.png', bins=100, range=(5000,5800) )
from __future__ import division
from pdb import set_trace


class counter():
  def __init__(self, before, after, indx):
    self.indx = indx
    self.actual = before
    self.predicted = after
    self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
    for a, b in zip(self.actual, self.predicted):
      if a == indx and b == indx:
        self.TP += 1
      elif a == b and a != indx:
        self.TN += 1
      elif a != indx and b == indx:
        self.FP += 1
      elif a == indx and b != indx:
        self.FN += 1
      elif a != indx and b != indx:
        pass

  def stats(self):
    pd, pf, prec, F, G, acc = 0, 0, 0, 0, 0,0
    if self.TP + self.FN:
      pd = self.TP / (self.TP + self.FN)
    if self.FP+self.TN:
      pf = self.FP/(self.FP+self.TN)
    if self.TP+self.FP:
      prec = self.TP/(self.TP+self.FP)
    if self.TP+self.FP+self.TN+self.TP:
      acc = (self.TP +self.TN)/(self.TP+self.TN+self.FP+self.FN)
    if pd+prec:
      F = 2*pd*prec/(pd+prec)
    if pd+(1-pf):
      G = 2*pd*(1-pf)/(pd+1-pf)
    return pd, pf,prec,acc,F,G

      # try:
      ## this implementation will get wrong results if the testing data only includes 1 label not two!
      ## in that case, all measures were zero. actually, for the actual label it is 1, the absent label is 0
      #   Sen = self.TP / (self.TP + self.FN)
      #   Spec = self.TN / (self.TN + self.FP)
      #   Prec = self.TP / (self.TP + self.FP)
      #   Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
      #   F = 2 * (Prec*Sen) / (Prec+Sen)
      #   F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
      #   G = 2 * Sen * Spec / (Sen + Spec)
      #   G1 = Sen * Spec / (Sen + Spec)
      #   return Sen, 1-Spec, Prec, Acc, F, G
      # except ZeroDivisionError:
      #   return 0, 0, 0, 0, 0, 0


class ABCD():
  "Statistics Stuff, confusion matrix, all that jazz..."

  def __init__(self, before, after):
    self.actual = before
    self.predicted = after

  def __call__(self):
    uniques = set(self.actual)
    for u in list(uniques):
      yield counter(self.actual, self.predicted, indx=u)

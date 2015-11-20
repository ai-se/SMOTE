# __author__ = 'WeiFu'
from __future__ import division
import sys
from sklearn import metrics
import pdb
from settings import *

def sk_abcd(pred_lst, actual_lst):
  # f1_Def = metrics.f1_score(actual,predicted_txt,pos_label = "Defective")*100
  # f1_NonDef = metrics.f1_score(actual,predicted_txt, pos_label = "Non-Defective")*100
  # pd_Def = metrics.recall_score(actual,predicted_txt,pos_label = "Defective")*100
  # pd_NonDef = metrics.recall_score(actual,predicted_txt,pos_label = "Non-Defective")*100
  # precision_Def = metrics.precision_score(actual,predicted_txt,pos_label = "Defective")*100
  # precision_NonDef = metrics.precision_score(actual,predicted_txt,pos_label = "Non-Defective")*100
  # score = [[pd_NonDef,0,precision_NonDef,f1_NonDef,0],[pd_Def,0,precision_Def,f1_Def,0]]
  # return score
  n = lambda x: int(x)
  p = lambda x: int(x * 100)
  def getLabel():
    label = []
    for i in actual_lst:
      if i not in label:
        label.append(i)
    return label

  def getABCD(label):
    for actual, predict in zip(actual_lst, pred_lst):
      for i in label:
        if actual == i:
          if actual == predict:
            D[i] = D.get(i, 0) + 1
          else:
            B[i] = B.get(i, 0) + 1
        else:
          if predict == i:
            C[i] = C.get(i, 0) + 1
          else:
            A[i] = A.get(i, 0) + 1
    return A, B, C,D

  def score(label, show = False):
    out = []
    for i in label:
      pd = pf = prec = f = g = w = acc = 0
      a = A.get(i,0);b = B.get(i,0);c = C.get(i,0);d = D.get(i,0)
      if b + d: pd = d / (b + d)
      if a + c: pf = c / (a + c)
      if c + d: prec = d / (c + d)
      if prec + pd: f = 2 * pd * prec / (pd + prec)
      if pd + 1 - pf: g = 2 * pd * (1 - pf) / (1 - pf + pd)
      if pd + 1-pf+ prec :w = 3* pd*(1-pf)*prec /(1-pf+pd+prec)
      if a + b + c + d: acc = (a + d) / (a + b + c + d)
      if show:
        print "#", (
        '{0:20s}{1:10s} {2:4d} {3:4d} {4:4d} ' +
        '{5:4d} {6:4d} {7:4d} {8:3d} {9:3d} ' +
        '{10:3d} {11:3d} {12:3d} {''13:10s}').format(
        "hello","test", n(b + d), n(a), n(b), n(c),
        n(d), p(acc), p(pd), p(pf), p(prec), p(f), p(g), i)
      out+=[[p(pd), p(pf), p(prec), p(f), p(g),p(w)]]
    return out

  A = {}; B ={}; C ={}; D = {}
  labels = getLabel()
  A,B,C,D = getABCD(labels)
  return score(labels)

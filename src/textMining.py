from __future__ import print_function, division
# __author__ = 'WeiFu'
import pickle
import random
import pdb
import sys
import mpi4py
import os.path
import numpy as np
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import KFold
from mpi4py import MPI
import pandas as pd
from learner import *


class Settings(object):
  def __init__(self, src="", method='tfidf'):
    self.threshold = 20
    self.data_src = src
    self.processors = 4
    self.method = method
    # self.data = self.get_data()
    self.corpus = self.load_data()
    # self.matrix, self.label = self.make_feature(num_features,method)

  def get_data(self):
    folders = [os.path.join(self.data_src, f) for f in listdir(self.data_src)
               if os.path.isfile(
        os.path.join(self.data_src, f)) and ".DS" not in f]
    print(folders)
    return folders

  def load_data(self):
    """
    loading data from files into self.corpus
    """

    def process(txt):
      """
      Preprocessing: stemming + stopwords removing
      """
      stemmer = PorterStemmer()
      cached_stopwords = stopwords.words("english")
      return ' '.join([stemmer.stem(word) for word in txt.lower().split() if
                       word not in cached_stopwords and len(word) > 1])

    if not self.data_src:
      raise ValueError('data src required!')
    all_label, corpus, used_label = [], [], []
    with open(self.data_src, 'r') as f:
      content = f.read().splitlines()
      for line in content:
        label = line.lower().split(' >>> ')[1].split()[0]
        # ' >>> ' the first term is selected as label.
        all_label.append(label)
        corpus.append([label] + process(line.split(' >>> ')[0]).split())
    label_dist = Counter(all_label)
    for key, val in label_dist.iteritems():
      if val > self.threshold:
        used_label.append(key)
    used_label.append('others')
    for doc in corpus:
      if doc[0] not in used_label:
        doc[0] = 'others'
    return corpus

  def make_feature(self, num_features=1000):
    """
    making feature matrix
    :param num: # of features selected
    :param method: "tfidf" and "tf"
    """

    def calculate_tf_idf():
      features, files, tfidf, matrix = {}, {}, {}, []
      for row in self.corpus:
        row_no_label = Counter(row[1:])
        matrix.append(row_no_label)  # keep each row into a matrix
        for key, val in row_no_label.iteritems():
          features[key] = features.get(key, 0) + val
          files[key] = files.get(key, 0) + 1

      for each_feature in files.keys():
        tfidf[each_feature] = features[each_feature] / sum(
          features.values()) * np.log(len(self.corpus) / files[each_feature])
      return tfidf, matrix

    def norm(mat):
      """
      l2 normalization. I haven't checked it out.
      """
      mat = mat.astype(float)
      for i, row in enumerate(mat):
        nor = np.linalg.norm(row, 2)
        if not nor == 0:
          mat[i] = row / nor
      return mat

    def hash(mat, num_features):
      """
      hashing trick, why need this hash function ????
      """
      hasher = FeatureHasher(num_features)
      X = hasher.transform(mat)
      X = X.toarray()
      return X

    matrix_selected = []
    label = list(zip(*self.corpus)[0])
    # pdb.set_trace()
    if self.method == 'tfidf':
      tfidf, matrix = calculate_tf_idf()
      features_selected = [pair[0] for pair in
                           sorted(tfidf.items(), key=lambda x: x[1])[
                           -num_features:]]
      for row in matrix:
        matrix_selected.append([row[each] for each in features_selected])
      matrix_selected = np.array(matrix_selected)
      matrix_selected = norm(matrix_selected)
    else:  # tf
      mat = [Counter(row[1:]) for row in self.corpus]
      matrix_selected = hash(mat, num_features)
      matrix_selected = norm(matrix_selected)
    data = pd.DataFrame(
      [list(a) + [b] for a, b in zip(matrix_selected, label)])
    return data


def cross_val(pd_data, learner, fold=5):
  """
  do 5-fold cross_validation
  """
  F = {}
  for i in xrange(5): # repeat 5 times here
    kf = KFold(len(pd_data),fold)
    for train_index, test_index in kf:
      train_X = pd_data.ix[train_index,:999].values
      train_Y = pd_data.ix[train_index,1000].values
      test_X = pd_data.ix[test_index,999].values
      test_Y = pd_data.ix[test_index,1000].values
      F = learner(train_X,train_Y,test_X,test_Y,F)
  return F


def run(data_src='../data/StackExchange/anime.txt', process=4):
  # model = Settings(data_src)
  # data = model.make_feature()
  # pdb.set_trace()
  # print(model.label)
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  features_num = [(rank + i * size) * 100 for i in xrange(12 + 1) if
                  rank + i * size <= 12]
  # different processes run different feature experiments
  model_hash = Settings(data_src, method='hash')
  model_tfidf = Settings(data_src, method='tfidf')
  learners = [naive_bayes]
  F_method = {}
  for learner in learners:
    random.seed(1)
    F_feature = {}
    for f_num in features_num:
      for method in [model_tfidf, model_hash]:
        pd_data = method.make_feature(f_num)
        F_feature[f_num] = cross_val(pd_data, learner)
    F_method[learner.func_name] = F_feature



if __name__ == "__main__":
  # settings().get_data()
  run()

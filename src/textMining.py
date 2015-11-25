from __future__ import print_function, division
# __author__ = 'WeiFu'
import pickle
import random
import time
import mpi4py
import os.path
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import KFold
from mpi4py import MPI
import pandas as pd
from learner import *
from sk import *
from smote import smote


class Settings(object):
  def __init__(self, src, method):
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
      # content = f.read().splitlines()
      for line in f.readlines():
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

  def make_feature(self, num_features):
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
        if np.log(len(self.corpus) / files[each_feature]) < 0:
          set_trace()
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
      hasher = FeatureHasher(n_features=num_features)
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


def cross_val(pd_data1, learner, issmote=False, fold=5):
  """
  do 5-fold cross_validation
  """
  F = {}
  for i in xrange(5):  # repeat 5 times here
    if issmote:
      pd_data1 = smote(pd_data1).run()
      # pdb.set_trace()
    pd_data1 = pd_data1.reindex(np.random.permutation(pd_data1.index))
    pd_data = pd.DataFrame(pd_data1.values)
    kf = KFold(len(pd_data), fold)
    for train_index, test_index in kf:
      train_X = pd_data.ix[train_index, pd_data.columns[:-1]].values
      train_Y = pd_data.ix[train_index, pd_data.columns[-1]].values
      test_X = pd_data.ix[test_index, pd_data.columns[:-1]].values
      test_Y = pd_data.ix[test_index, pd_data.columns[-1]].values
      F = learner(train_X, train_Y, test_X, test_Y, F)
  return F


def scott(features_num, learners, score):
  """
   pass results to scott knott
  """
  out = []
  for num in features_num:
    for learner in learners:
      out.append(
        [learner.func_name + "_" + str(num)] + score[num][learner.func_name][
          'mean'])
      out.append([learner.func_name + "_" + str(num) + "_weighted"] +
                 score[num][learner.func_name]['mean_weighted'])
  rdivDemo(out)


def run(data_src='../data/StackExchange/anime.txt', process=4):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  print("process", str(rank), "started:", time.strftime("%b %d %Y %H:%M:%S "))
  # different processes run different feature experiments
  features_num = [1000 * i for i in xrange(1, 11)]
  features_num_process = [features_num[i] for i in
                          xrange(rank, len(features_num), size)]
  # model_hash = Settings(data_src, method='hash')
  model_tfidf = Settings(data_src, method='tfidf')
  methods_lst = [model_tfidf]
  learners = [naive_bayes]
  F_feature = {}
  for f_num in features_num_process:
    F_method = {}
    for learner in learners:
      random.seed(1)
      for isSmote in [True, False]:
        for method in methods_lst:
          pd_data = method.make_feature(f_num)
          if isSmote:
            F_method[learner.func_name+"_smote"] = cross_val(pd_data, learner, isSmote)
          elif not isSmote:
            F_method[learner.func_name] = cross_val(pd_data, learner, isSmote)
    F_feature[f_num] = F_method
  if rank == 0:
    for i in xrange(1, size):
      temp = comm.recv(source=i)
      F_feature.update(temp)  # receive data from other process
    scott(features_num, learners, F_feature)
    file_name = data_src[data_src.rindex('/') + 1:data_src.rindex('.')]
    with open('../pickles/' + file_name + '.pickle', 'wb') as mypickle:
      pickle.dump(F_feature, mypickle)
  else:
    comm.send(F_feature, dest=0)
  print("process", str(rank), "end:", time.strftime("%b %d %Y %H:%M:%S "))


def atom(x):
  try:
    return int(x)
  except ValueError:
    try:
      return float(x)
    except ValueError:
      return x


def cmd(com="Nothing"):
  "Convert command line to a function call."
  if len(sys.argv) < 2: return com
  def strp(x): return isinstance(x, basestring)
  def wrap(x): return "'%s'" % x if strp(x) else str(x)
  words = map(wrap, map(atom, sys.argv[2:]))
  return sys.argv[1] + '(' + ','.join(words) + ')'


if __name__ == "__main__":
  run()
  # settings().get_data()
  # eval(cmd())

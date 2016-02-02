from __future__ import print_function, division
# __author__ = 'WeiFu'
import pickle
import time
from os import listdir
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import StratifiedKFold
from mpi4py import MPI
from learner import *
from sk import *
from smote import smote
from tuner import *
import pdb


class Settings(object):
  def __init__(self, src, method, isYes_label, target_class):
    self.total_class = 20
    self.data_src = src
    self.processors = 4
    self.method = method
    self.isYes_label = isYes_label
    self.target_class = target_class
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
        label = line.lower().split('>>>')[1].split()[0]
        # ' >>> ' the first term is selected as label.
        all_label.append(label)
        corpus.append([label] + process(line.split('>>>')[0]).split())
    label_dist = Counter(all_label)
    if self.isYes_label:
      used_label = ["yes", "no"]
    else:
      used_label = [iterm[0] for iterm in label_dist.most_common(self.total_class)]
    for doc in corpus:
      if doc[0] not in used_label:
        doc[0] = 'others'
    return corpus

  def make_binary(self, corpus):
    pass

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
      hashing trick
      """
      hasher = FeatureHasher(n_features=num_features, non_negative=True)
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
    else:  # hash
      mat = [Counter(row[1:]) for row in self.corpus]
      matrix_selected = hash(mat, num_features)
      matrix_selected = norm(matrix_selected)
    data = pd.DataFrame(
      [list(a) + [b] for a, b in zip(matrix_selected, label)])
    return data


def cross_val(pd_data, learner, target_class, goal, isWhat="", fold=2,
              repeats=1):
  """
  do 5-fold cross_validation
  """

  def tune_learner(train_X):
    train_len = len(train_X)
    new_train_index = np.random.choice(range(train_len), train_len * 0.7)
    new_tune_index = list(set(range(train_len)) - set(new_train_index))
    new_train_X = train_X[new_train_index]
    new_train_Y = train_Y[new_train_index]
    new_tune_X = train_X[new_tune_index]
    new_tune_Y = train_Y[new_tune_index]
    clf = learner(new_train_X, new_train_Y, new_tune_X, new_tune_Y, goal)
    tuner = DE_Tune_ML(clf, clf.get_param(), target_class, goal)
    return tuner.Tune()

  def tune_SMOTE(train_pd):

    train_len = len(train_pd)
    new_train_index = random.sample(train_pd.index, int(train_len * 0.7))
    new_train = train_pd.ix[new_train_index]
    if "_TunedSmote" in isWhat:
      new_train_X = new_train.ix[:, new_train.columns[:-1]].values
      new_train_Y = new_train.ix[:, new_train.columns[-1]].values
      new_tune = train_pd.drop(new_train_index)
      new_tune_X = new_tune.ix[:, new_tune.columns[:-1]].values
      new_tune_Y = new_tune.ix[:, new_tune.columns[-1]].values
      # clf = learner(new_train_X, new_train_Y, new_tune_X, new_tune_Y)
      A_smote = smote(new_train)
      num_range = [[int(A_smote.get_majority_num() * 0.5),
                    int(A_smote.get_majority_num() * 1.5)]] * (
                    A_smote.label_num - 1)
      params_to_tune = {"k": [2, 20], "up_to_num": num_range}
      # pdb.set_trace()
      tuner = DE_Tune_SMOTE(learner, smote, params_to_tune, new_train,
                            new_tune, target_class, goal)
      params = tuner.Tune()
      return params, new_train

  F = {}
  for i in xrange(repeats):  # repeat 5 times here
    # if isWhat == "_Smote":
    #   pd_data1 = smote(pd_data1).run()
    #   # pdb.set_trace()
    # pd_data1 = pd_data1.reindex(np.random.permutation(pd_data1.index))
    # pd_data = pd.DataFrame(pd_data1.values)
    kf = StratifiedKFold(pd_data.ix[:, pd_data.columns[-1]].values, fold, shuffle=True)
    for train_index, test_index in kf:
      train_pd = pd_data.ix[train_index]
      test_pd = pd_data.ix[test_index]
      if "Smote" in isWhat:
        k = 5
        up_to_num = []
        if "_TunedSmote" in isWhat:
          params, train_pd = tune_SMOTE(train_pd)
          # use new training data not original, because some are used as tuning
          k = params["k"]
          up_to_num = params["up_to_num"]
        train_pd = smote(train_pd, k, up_to_num).run()

      train_X = train_pd.ix[:, train_pd.columns[:-1]].values
      train_Y = train_pd.ix[:, train_pd.columns[-1]].values
      test_X = test_pd.ix[:, test_pd.columns[:-1]].values
      test_Y = test_pd.ix[:, test_pd.columns[-1]].values
      params = tune_learner(train_X) if "_TunedLearner" in isWhat else {}
      F = learner(train_X, train_Y, test_X, test_Y, goal).learn(F, **params)
  # pdb.set_trace()
  return F


def scott(scores):
  """
   pass results to scott knott
  """
  out = []
  for key, val in scores.iteritems():
    try:
      out.append([key] + val)
    except:
      print("Score has errors")
  rdivDemo(out)


def run(data_src, process, isYes_label=False, target_class="Macro_F",
        goal="Macro_F"):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F", 7: "Micro_F"}[6]
  print("process", str(rank), "started:", time.strftime("%b %d %Y %H:%M:%S "))
  # different processes run different feature experiments
  features_num = [100 * i for i in xrange(1, 11, 3)]
  model_tfidf = Settings(data_src, 'tfidf', isYes_label, target_class)
  # model_hash = Settings(data_src, 'hash', isBinary, isYes_label,target_class)
  methods_lst = [model_tfidf]
  modification = ["_Naive"]
  # modification = ["_TunedLearner"]  # [
  learners = [Linear_SVM]
  F_feature = {}

  ## Way2: distribute all jobs to different processors.
  jobs = [(num, trick, learner, m) for learner in learners for m in methods_lst for trick in modification for num in
          features_num]
  jobs_processor = [jobs[i] for i in xrange(rank, len(jobs), size)]
  for f_num, isWhat, learner, method in jobs_processor:
    random.seed(1)
    pd_data = method.make_feature(f_num)
    target_class = method.target_class  # here's an issue: We need to predict what????? yes or no, or mean_weighted?
    name = learner.name + isWhat
    key_name = name + "_" + str(f_num) + "_" + target_class
    F_feature[key_name] = cross_val(pd_data, learner, target_class, goal, isWhat)[target_class]

  if rank == 0:
    for i in xrange(1, size):
      temp = comm.recv(source=i)
      F_feature.update(temp)  # receive data from other process
    scott(F_feature)
    file_name = data_src[data_src.rindex('/') + 1:data_src.rindex('.')]
    with open('../pickles/linearSVC/' + file_name + '.pickle', 'wb') as mypickle:
      pickle.dump(F_feature, mypickle)
  else:
    comm.send(F_feature, dest=0)
  print("process", str(rank), "last job is", jobs_processor[-1][1], "end:", time.strftime("%b %d %Y %H:%M:%S "))


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
  if len(sys.argv) == 1:
    run('../data/StackExchange/android.txt', 1)
  else:
    eval(cmd())
    # run('../data/StackExchange/anime.txt')


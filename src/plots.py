from __future__ import  division, print_function
import sys
import pickle
import pdb
from os.path import join, isfile
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def read_pickles(src = "../pickles_HPC_0127/"):
  pickle_folders = [ i for i in listdir(src) if not isfile(join(src, i))]
  all_scores_dist = {}
  for folder in pickle_folders: # folder will be hash, tfidf, or Lienar SVM
    full_path = join(src,folder)
    temp_files = [ f for f in listdir(full_path) if isfile(join(full_path,f)) and ".pickle" in f]
    temp_data_dict = {} # a dict to hold all results of this method.
    for file in temp_files: # file is the name, like SE_codereview.pickle
      path = join(full_path,file)
      with open(path, "rb") as handle:
        temp_data_dict[file]=pickle.load(handle)
    all_scores_dist[folder] = temp_data_dict
  return all_scores_dist


def median_ipr(folder, one_dataset_result):
  method = ["_Naive", "_Smote", "_TunedLearner", "_TunedSmote"]
  feature_num = [100,400,700,1000]
  learner_name = one_dataset_result.keys()[0][:one_dataset_result.keys()[0].index("_")]
  out = {}
  for m in method:
    median= []
    iqr=[]
    for num in feature_num:
      name =learner_name +m +"_" +str(num)+"_yes"
      # pdb.set_trace()
      if name in one_dataset_result.keys():
        median.append(np.median(one_dataset_result[name]))
        q75, q25 = np.percentile(one_dataset_result[name],[75, 25])
        # pdb.set_trace()
        iqr.append(q75-q25)
    if len(iqr) >=1:
      out[folder+"_"+learner_name+m] = [median,iqr] # for each method as key, the median and iqr are the elements of list
  return out




def plot(data):
  tot_subplot = max(data.values()[0].keys())
  x_aix = [0,1,2,3]
  color = {"_Naive":"r","_Smote":"g","_TunedLearner":"b","_TunedSmote":"k"}
  method_mark = ["-","^-","s-"]
  iqr_mark = ["--","^","s"]
  data_set = data.values()[0].keys()
  # data_set = ['SE_codereview.pickle', 'SE_webmasters.pickle', 'SE_unix.pickle', 'SE_opendata.pickle']
  # pdb.set_trace()
  for i, each in enumerate(data_set):
    plt.figure(i+1)
    to_plot = [] # all the plots for this one dataset
    for key, val in data.iteritems():
      if each not in val.keys():
        continue
      method_data = val[each]
      to_plot.append(median_ipr(key,method_data))
    for n_index, method in enumerate(to_plot):
      count = 0
      for name,median_iqr in method.iteritems():
        plt.plot(x_aix,median_iqr[0], method_mark[n_index],color = color[name[name.rindex("_"):]], label =name+"_median")
        plt.plot(x_aix,median_iqr[1],iqr_mark[n_index], color = color[name[name.rindex("_"):]],label =name+"_iqr")
        count+=1
    plt.xticks([0,1,2,3],["100","400","700","1000"])
    plt.xlabel("Number of features")
    plt.ylabel("F score median_iqr")
    plt.legend(fontsize ="small", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Datasets:" +each[:each.index(".")])
    plt.savefig(each[:each.index(".")]+".png", bbox_inches='tight')
  plt.show()
  plt.close()





if __name__ == "__main__":
  plot(read_pickles())




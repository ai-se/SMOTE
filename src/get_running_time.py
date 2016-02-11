from __future__ import division, print_function
import sys, pdb
from os.path import join, isfile
from os import listdir


def read(src="out/0210_result"):
  files = [join(src, i) for i in listdir(src) if isfile(join(src, i)) and ".DS" not in i]
  file_names = [ i for i in listdir(src) if isfile(join(src, i)) and ".DS" not in i]
  learner_name = ["Naive","Smote","TunedLearner"]
  results={}
  for each, n in zip(files,file_names):
    with open(each,"rb") as f:
      lsts = f.read().splitlines()
      this_file_result = {}
      for ii,line in enumerate(lsts):
        if "sec" in line:
          item = line.split(" ")
          for i,name in enumerate(learner_name):
            if name in item[0]:
              if "evaluation" in lsts[ii-1]:
                eval_line =lsts[ii-1]
                eval_item = eval_line.split(":")[1]
                this_file_result[name] = this_file_result.get(name, [])+[str(round(float(item[3]),1))+"/"+eval_item]
              else:
                this_file_result[name] = this_file_result.get(name, [])+[str(round(float(item[3]),1))]
        # if "evaluation" in line:
        #   item = line.split(":")
        #   this_file_result["Tuning_eval"] = this_file_result.get("tuning_eval",[]) + [item[1]]

    if this_file_result is not None:
      results[n] = this_file_result
  output = ""
  for key,val in results.iteritems():
    output += "\n"*3+key[:key.index(".")] +"\n"*2
    output += ",".join(["name","100","400","700","1000","1300","1600","1900"]) +"\n"
    for learner,time in val.iteritems():
      output+=learner+","+",".join(time) +"\n"

  f = open("runingtime.csv","wb")
  f.write(output)

read()
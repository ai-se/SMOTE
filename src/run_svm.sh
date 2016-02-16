#! /bin/tcsh

rm out_svm/*
rm err_svm/*


#foreach VAR (drupal academia apple gamedev rpg english electronics physics tex scifi SE0 SE1 SE2 SE3 SE4 SE5 SE6 SE7 SE8 SE9 SE10 SE11 SE12 SE13 SE14)
#  bsub -W 1000 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 4 /share3/wfu/miniconda/bin/python2.7 textMining.py run /share3/wfu/Datasets/StackExchange/$VAR.txt
#end



###### this is only for testing on HPC
#foreach VAR (androidd rupal academia apple gamedev rpg english electronics physics tex scifi)
#  bsub -W 3600 -n 16 -o ./out_svm/$VAR.out.%J -e ./err_svm/$VAR.err.%J mpiexec -n 16 /share3/wfu/miniconda/bin/python2.7 textMining_LinearSVC.py run /share3/wfu/Datasets/StackExchange/$VAR.txt 16
#end


##### this is only for testing on HPC, SE tag level
foreach VAR (ansible atom d3 Ghost graphite-web logstash moment scikit-learn)
  bsub -W 3600 -n 16 -o ./out{_svm/$VAR.out.%J -e ./err_svm/$VAR.err.%J mpiexec -n 16 /share3/wfu/miniconda/bin/python2.7 textMining.py run /share3/wfu/Datasets/SE/$VAR.txt 16
end
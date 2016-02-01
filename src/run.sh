#! /bin/tcsh

rm out/*
rm err/*


#foreach VAR (drupal academia apple gamedev rpg english electronics physics tex scifi SE0 SE1 SE2 SE3 SE4 SE5 SE6 SE7 SE8 SE9 SE10 SE11 SE12 SE13 SE14)
#  bsub -W 1000 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 4 /share3/wfu/miniconda/bin/python2.7 textMining.py run /share3/wfu/Datasets/StackExchange/$VAR.txt
#end

#rm out_test/*
#rm err_test/*


##### this is only for testing on HPC, site level
#foreach VAR (SE_codereview SE_datascience SE_emacs SE_opendata SE_programmers SE_softwarerecs SE_unix SE_webapps SE_webmasters SE_windowsphone SE_wordpress)
#  bsub -W 3600 -n 16 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 16 /share3/wfu/miniconda/bin/python2.7 textMining.py run /share3/wfu/Datasets/StackExchange/$VAR.txt 16
#end


##### this is only for testing on HPC, tag level
foreach VAR (android)
  bsub -W 3600 -n 16 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 16 /share3/wfu/miniconda/bin/python2.7 textMining.py run /share3/wfu/Datasets/StackExchange/$VAR.txt 16
end
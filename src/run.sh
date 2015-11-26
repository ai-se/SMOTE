#! /bin/tcsh

rm out/*
rm err/*


foreach VAR (drupal academia apple gamedev rpg english electronics physics tex scifi SE0 SE1 SE2 SE3 SE4 SE5 SE6 SE7 SE8 SE9 SE10 SE11 SE12 SE13 SE14)
  bsub -W 300 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 4 /share3/wfu/miniconda/bin/python2.7 textMining.py run /share3/wfu/Datasets/StackExchange/$VAR.txt
end


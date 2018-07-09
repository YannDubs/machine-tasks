#!/usr/local/bin/bash

python ../make_lookup_tables.py -d . --is-target-attention

# jump : awk -F '\t' 'split($1,a,"[ \t]+") != 5' train.tsv > trainJump.tsv

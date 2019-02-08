#!/usr/bin/env bash

python3 ../AttentionLocalization/make_attn_loc.py -d . -n 10000 --max-step 3 --max-len-src 15 --max-len-tgt 20 --not-add-rep

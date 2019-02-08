#!/usr/bin/env bash

python3 ../AttentionLocalization/make_attn_loc.py -d . -n 5000 --not-add-rep --max-step 1 --max-len-src 15 --max-len-tgt 20

#!/bin/bash

rm -i *.db hpopt.log
python HP-optimization.py >> hpopt.log &
tail -f hpopt.log

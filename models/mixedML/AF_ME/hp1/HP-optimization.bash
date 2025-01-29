#!/bin/bash

rm -ir *.db hpopt.log results/
python HP-optimization.py >> hpopt.log &
tail -f hpopt.log

#!/usr/bin/env Rscript

library("formatR")

args = commandArgs(trailingOnly=TRUE)

if (length(args)!=1) {
  stop("One argument must be supplied (input file).n", call.=FALSE)
}

tidy_eval(args[1])

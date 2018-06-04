#!/bin/zsh

BIN=sparse_adjacency_matrices
INPUT_PATH="../Data"
OUTPUT_PATH="../Results"

for DIR in ${INPUT_PATH}/*; do
  NAME=`basename $DIR`
  mkdir $OUTPUT_PATH/$NAME
  $BIN -o $OUTPUT_PATH/$NAME "$DIR/${NAME}_A.txt"
done

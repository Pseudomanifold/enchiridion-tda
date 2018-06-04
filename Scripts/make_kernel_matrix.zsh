#!/bin/zsh

BIN=topological_distance
INPUT_PATH="../Results"

for DIR in ${INPUT_PATH}/*; do
  $BIN -w -p 1 -k $DIR/*_d*.txt > "$DIR/Kernel.txt"
done

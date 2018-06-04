#!/bin/zsh

BIN=topological_distance
INPUT_PATH="../Results"

for DIR in ${INPUT_PATH}/*; do
  find $DIR -name "*_*.txt" -print > "$DIR/Files.txt"
  $BIN -n -i -p 1 -k -l "$DIR/Files.txt" > "$DIR/Kernel.txt"
done

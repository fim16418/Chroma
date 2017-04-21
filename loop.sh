#!/bin/bash

PROGRAM=prop

OUTPUT_FILE=loopTest.txt

MIN_SIZE=2
MAX_SIZE=16
INCREMENT_SIZE=2

NUM_LOOPS=1000

rm -f $OUTPUT_FILE

for (( size=$MIN_SIZE; size<=$MAX_SIZE; size=size+$INCREMENT_SIZE ))
do
  echo "$size"
  ./$PROGRAM -ioblock 1 --outFile $OUTPUT_FILE \
             --nLoops $NUM_LOOPS --lattice $size $size $size $size
done

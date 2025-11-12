#!/bin/bash

OUTPUT_FILE="3.2.2_blocks_results.csv"
THREADS_PER_BLOCK=1024
GRID_DIMS=$(seq 1 32)

MEM_SIZES=()
SIZE=1024
while [ $SIZE -le 1073741824 ]; do
    MEM_SIZES+=($SIZE)
    SIZE=$((SIZE * 4))
done

echo "Size(Bytes),Blocks,ThreadsPerBlock,Bandwidth(GB/s)" > $OUTPUT_FILE

for SIZE in "${MEM_SIZES[@]}"; do
    if [ $SIZE -le 16384 ]; then
        ITER=10000
    elif [ $SIZE -le 1048576 ]; then
        ITER=500
    elif [ $SIZE -le 67108864 ]; then
        ITER=100
	else [ $SIZE -le 268435456 ]; then
		ITER=10
	else
		ITER=5
    fi

    for G in $GRID_DIMS; do
        echo "Running test: size=$SIZE, blocks=$G, iterations=$ITER"

        RESULT=$(srun -p exercise-gpu --gres=gpu:1 \
            ./bin/3.2.1 --global-coalesced -s $SIZE -t $THREADS_PER_BLOCK -g $G -i $ITER -y 2>&1)

        BW=$(echo "$RESULT" | grep "bw=" | awk -F'bw=' '{print $2}' | awk '{print $1}')
        [ -z "$BW" ] && BW="NaN"

        echo "$SIZE,$G,$THREADS_PER_BLOCK,$BW" >> $OUTPUT_FILE
    done
done

echo "All tests done. Results saved in $OUTPUT_FILE"

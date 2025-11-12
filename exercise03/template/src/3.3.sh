#!/bin/bash

OUTPUT_FILE="3.3_stride_results.csv"

MEM_SIZES=()
SIZE=1024
while [ $SIZE -le 1073741824 ]; do
    MEM_SIZES+=($SIZE)
    SIZE=$((SIZE * 4))
done

THREADS_PER_BLOCK=1024
BLOCKS=(1 2 4 8 16 32)
STRIDES=(1 2 4 8 16 32)

echo "Size(Bytes),ThreadsPerBlock,Blocks,Stride,Bandwidth(GB/s)" > $OUTPUT_FILE

for SIZE in "${MEM_SIZES[@]}"; do
    for B in "${BLOCKS[@]}"; do
        for STRIDE in "${STRIDES[@]}"; do
            echo "Testing Size=$SIZE, TPB=$THREADS_PER_BLOCK, Blocks=$B, Stride=$STRIDE"

            if [ $SIZE -le 16384 ]; then
                ITER=10000
            elif [ $SIZE -le 1048576 ]; then
                ITER=500
            elif [ $SIZE -le 67108864 ]; then
                ITER=100
            else
                ITER=10
            fi

            RESULT=$(srun -p exercise-gpu --gres=gpu:1 ./bin/3.2.1 \
                --global-stride -s $SIZE -t $THREADS_PER_BLOCK -g $B --stride $STRIDE -i $ITER -y 2>&1)

            BW=$(echo "$RESULT" | grep "bw=" | awk -F'bw=' '{print $2}' | awk '{print $1}')
            [ -z "$BW" ] && BW=0

            echo "$SIZE,$THREADS_PER_BLOCK,$B,$STRIDE,$BW" >> $OUTPUT_FILE
        done
    done
done

echo "Stride test done. Results saved in $OUTPUT_FILE"

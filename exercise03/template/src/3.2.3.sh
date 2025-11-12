#!/bin/bash

OUTPUT_FILE="3.2.3_all_results.csv"

MEM_SIZES=(1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 134217728 268435456 1073741824)
THREADS_PER_BLOCK=(32 64 128 256 512 1024)
BLOCKS=(1 2 4 8 16 32)

echo "Size(Bytes),ThreadsPerBlock,Blocks,Bandwidth(GB/s)" > $OUTPUT_FILE

for SIZE in "${MEM_SIZES[@]}"; do
    for TPB in "${THREADS_PER_BLOCK[@]}"; do
        for B in "${BLOCKS[@]}"; do
            echo "Testing size=$SIZE, TPB=$TPB, Blocks=$B"

            if [ $SIZE -le 16384 ]; then
                ITER=10000
            elif [ $SIZE -le 1048576 ]; then
                ITER=500
            elif [ $SIZE -le 67108864 ]; then
                ITER=100
            elif [ $SIZE -le 268435456 ]; then
                ITER=10
            else
                ITER=5
            fi

            RESULT=$(srun -p exercise-gpu --gres=gpu:1 ./bin/3.2.1 \
                --global-coalesced -s $SIZE -t $TPB -g $B -i $ITER -y 2>&1)

            BW=$(echo "$RESULT" | grep "bw=" | awk -F'bw=' '{print $2}' | awk '{print $1}')
            [ -z "$BW" ] && BW="NaN"

            echo "$SIZE,$TPB,$B,$BW" >> $OUTPUT_FILE
        done
    done
done

echo "All tests done. Results saved in $OUTPUT_FILE"

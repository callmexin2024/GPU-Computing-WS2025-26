#!/bin/bash

OUTPUT_FILE="3.2.1_coalesced_results.csv"
THREADS_PER_BLOCK=(1 2 4 8 16 32 64 128 256 512 1024)

MEM_SIZES=()
SIZE=1024
while [ $SIZE -le 1073741824 ]; do
    MEM_SIZES+=($SIZE)
    SIZE=$((SIZE * 4))
done

echo "Size(Bytes),ThreadsPerBlock,Bandwidth(GB/s)" > $OUTPUT_FILE

for SIZE in "${MEM_SIZES[@]}"; do
    if [ $SIZE -le 16384 ]; then
        ITER=1000
    elif [ $SIZE -le 1048576 ]; then
        ITER=500
    elif [ $SIZE -le 67108864 ]; then
        ITER=100
	else
		ITER=10
    fi

    for TPB in "${THREADS_PER_BLOCK[@]}"; do
        echo "Running test: size=$SIZE, threads_per_block=$TPB, iterations=$ITER"

        RESULT=$(srun -p exercise-gpu --gres=gpu:1 \
            ./bin/3.2.1 --global-coalesced -s $SIZE -t $TPB -i $ITER -y 2>&1)

        BW=$(echo "$RESULT" | grep "Coalesced copy" | awk -F'bw=' '{print $2}' | awk '{print $1}')
        [ -z "$BW" ] && BW="NaN"

        echo "$SIZE,$TPB,$BW" >> $OUTPUT_FILE
    done
done

echo "All tests done. Results saved in $OUTPUT_FILE"


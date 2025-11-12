#!/bin/bash

OUTPUT_FILE="3.4_offset_results.csv"

MEM_SIZES=()
SIZE=1024
while [ $SIZE -le 1073741824 ]; do
    MEM_SIZES+=($SIZE)
    SIZE=$((SIZE * 4))
done

THREADS_PER_BLOCK=128
OFFSETS=(1 2 4 8 16 32 64 128 256)

echo "Size(Bytes),ThreadsPerBlock,Blocks,Offset,Bandwidth(GB/s)" > $OUTPUT_FILE

for SIZE in "${MEM_SIZES[@]}"; do
    TOTAL_THREADS=$((SIZE / 4))
    BLOCKS_NEEDED=$(( (TOTAL_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK ))
    BLOCKS_NEEDED=$(( BLOCKS_NEEDED < 1 ? 1 : BLOCKS_NEEDED ))

    for OFFSET in "${OFFSETS[@]}"; do
        echo "Testing Size=$SIZE, Blocks=$BLOCKS_NEEDED, Offset=$OFFSET"

        if [ $SIZE -le 16384 ]; then
            ITER=100000
        elif [ $SIZE -le 1048576 ]; then
            ITER=10000
        elif [ $SIZE -le 67108864 ]; then
            ITER=1000
        else
            ITER=100
        fi

        RESULT=$(srun -p exercise-gpu --gres=gpu:1 ./bin/3.2.1 \
            --global-offset -s $SIZE -t $THREADS_PER_BLOCK -g $BLOCKS_NEEDED --offset $OFFSET -i $ITER -y 2>&1)

        BW=$(echo "$RESULT" | awk -F'bw=' '{print $2}' | awk '{print $1}' | tr -d '\r\n')
		[ -z "$BW" ] && BW=0



        echo "$SIZE,$THREADS_PER_BLOCK,$BLOCKS_NEEDED,$OFFSET,$BW" >> $OUTPUT_FILE
    done
done

echo "Offset test done. Results saved in $OUTPUT_FILE"

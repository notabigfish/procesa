for SEED in 101 167 400 668 877
do
    for MODELIND in 49 53 24
    do
        for XXX in mixed_split_esm2 # human_cell_esm2 human_esm2  #
        do
            mkdir -p ${XXX}/seed-${SEED}/model${MODELIND} && cp ../../protein4all/results/${XXX}/seed-${SEED}/model${MODELIND}/results.txt ${XXX}/seed-${SEED}/model${MODELIND}
        done
    done
done

export ROOTDIR=/
export PYTHONPATH=$PWD:$PYTHONPATH
# export DATANAME=S_0
export MODELNAME=esm1b
DATALIST=(S_0, S_1, S_2, S_3, S_4)
NAMELIST=(hotprotein_20, hotprotein_21, hotprotein_22, hotprotein_23, hotprotein_24)

for i in "${!DATALIST[@]}"
do
	cd $ROOTDIR/procesa/FLIP/baselines
	export PYTHONPATH=$PWD
	DATANAME=${DATALIST[i]}
	KEYNAME=${NAMELIST[i]}

	python embeddings/embeddings.py \
		${KEYNAME} \
		$MODELNAME \
		--local_esm_model ${ROOTDIR}/dwnl_ckpts/esm1b_t33_650M_UR50S.pt \
		--datadir ${ROOTDIR}/datasets/procesa_data/ \
		--bulk_compute \
		--outdir /datasets/procesa_data/hotprotein/$DATANAME/$MODELNAME \
		--make_fasta \
		--truncate 1 \
		--trunc_len 800 \
		--toks_per_batch 1024 \
		--include 'per_tok contacts'
		# --concat_tensors \

	cd /procesa
	export PYTHONPATH=$PWD
	python build_dgl_graph.py \
		--dataroot /datasets/procesa_data/hotprotein \
		--dataset ${DATANAME} \
		--model ${MODELNAME}

	rm -r /datasets/procesa_data/hotprotein/$DATANAME/$MODELNAME/train
	rm -r /datasets/procesa_data/hotprotein/$DATANAME/$MODELNAME/test
	rm -r /datasets/procesa_data/hotprotein/$DATANAME/$MODELNAME/*.fasta
done

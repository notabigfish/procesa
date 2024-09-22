export ROOTDIR=/
export PYTHONPATH=$PWD:$PYTHONPATH
export DATANAME=human
export MODELNAME=esm1v
python embeddings/embeddings.py \
	meltome_2 \
	$MODELNAME \
	--local_esm_model ${ROOTDIR}/dwnl_ckpts/esm1v_t33_650M_UR90S_1.pt \
	--datadir ${ROOTDIR}/datasets/procesa_data/ \
	--bulk_compute \
	--outdir ${ROOTDIR}/datasets/procesa_data/meltome/$DATANAME/$MODELNAME \
	--make_fasta \
	--truncate 1 \
	--trunc_len 800 \
	--toks_per_batch 1024 \
	--include 'per_tok contacts'
	# --concat_tensors \

cd /procesa
export PYTHONPATH=$PWD
python build_dgl_graph.py \
	--dataroot ${ROOTDIR}/datasets/procesa_data/meltome \
	--dataset ${DATANAME} \
	--model ${MODELNAME}

rm -r ${ROOTDIR}/datasets/procesa_data/meltome/$DATANAME/$MODELNAME/train
rm -r ${ROOTDIR}/datasets/procesa_data/meltome/$DATANAME/$MODELNAME/test
rm -r ${ROOTDIR}/datasets/procesa_data/meltome/$DATANAME/$MODELNAME/*.fasta


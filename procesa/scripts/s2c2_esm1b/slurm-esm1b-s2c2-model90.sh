
export ROOTDIR=/procesa
cd $ROOTDIR
export PYTHONPATH=$PWD:$PYTHONPATH
export DATANAME='s2c2_0'
export MODELNAME='esm1b'
export EXPNAME='model90'
export DATAROOT=/datasets/procesa_data/hotprotein

for SEED in 101
do
	export RESULT_PATH=$ROOTDIR/results/${DATANAME}_${MODELNAME}/seed-${SEED}/${EXPNAME}
	python train.py \
		configs/${DATANAME:0:4}_${MODELNAME}/${EXPNAME}.py \
		--result_path $RESULT_PATH \
		--dataroot $DATAROOT \
		--dataname $DATANAME \
		--modelname $MODELNAME \
		--seed ${SEED}
done

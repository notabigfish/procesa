## conda environment
`environment.yml`

## esm model
Download pretrained esm1b, esm1v, esm2 model from original esm github. Put these models in `dwnl_ckpts`.

## data generation

Use scripts in `/procesa/FLIP/baselines/scripts/` to generate dgl graph pkl files. Generated data will be saved in `/datasets`.

(Due to file size limit, the hotprotein-S dataset is in https://drive.google.com/file/d/1VvAXKw01hMrKBMsOMDB5OKcQlvSzN0TN/view?usp=drive_link.)

## Train and evaluate
Run scripts in `/procesa/scripts` to train and evaluate models. Results will be saved in `/procesa/results/`. The correspondence between results shown in paper and running scripts are shown in figure below.

![Table 3](images/table3.png)
![Table 4](images/table4.png)
![Table 5](images/table5.png)


For FLIP, all trainval scripts are listed.
For hotprotein-s2c2 and hotprotein-s2c5, you can change `DATANAME` and `EXPNAME` to run other experiments, like `s2c2_1` and `model31`.
For hotprotein-S, you can change `EXPNAME` to run other experiments, like `model116`.

## Results
Results are saved in `/procesa/results` folder.

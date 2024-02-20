Pytorch implementation of : https://arxiv.org/abs/2011.06704

Reference from Tensorflow implementation here: https://github.com/tcl9876/Diffusion-Handwriting-Generation


Steps to run inference:
1. Run inference.py to generate output samples on the pretrained weights. Modify argument as necessary

Steps to run train:
1. Go to https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database
2. Download data/ascii-all.tar.gz, data/lineStrokes-all.tar.gz, and data/lineImages-all.tar.gz
3. Extract those files to the project's data folder
4. Run processing.py to generate the pickle files
5. Run utils.py to cache style vectors
6. Run train.py to create retrain and create new weights


[//]: # (Current changes:)
[//]: # ()
[//]: # (| Tensorflow         | Pytorch         | Notes |)
[//]: # (|--------------------|-----------------|-------|)
[//]: # (| Dropout            | Dropout         | SP    |)
[//]: # (| LayerNormalization | LayerNorm       | MI    |)
[//]: # (| Dense              | Linear          | MI    |)
[//]: # (| swish              | SiLu            | SP    |)
[//]: # (| Embedding          | Embedding       | SP    |)
[//]: # (| Conv1D             | Conv1d          | MI    |)
[//]: # (| AveragePooling1D   | AvgPool1d       | SP    |)
[//]: # (| UpSampling1D       | Upsample        | DP    |)
[//]: # (| gather_nd          | simple indexing ||)
[//]: # (Notation:)
[//]: # (SP: Same parameters)
[//]: # (MI: Missing input shape, but with overall same parameters)
[//]: # (DP: Different parameters, require different things)

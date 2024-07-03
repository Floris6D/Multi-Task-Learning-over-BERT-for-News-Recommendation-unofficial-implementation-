
# Introduction
Hello there 👋🏽
This repo was made for the course Recommender Systems from the University of Amsterdam. Contributors are Jasper Eppink, Cedrik Blommestijn and Floris Six Dijkstra. All code within src/mtrec is created by us.

## EBNeRD 
The model in this repo is based on the paper: MTRec: Multi-Task Learning over BERT for News Recommendation. We provide an open source implementation of MTRec, with some additional functionalities such as automized hyperparameter tuning, and extended Named Entity Recognition labels. The dataset EBNeRD was provided by the Ekstra Bladet Recommender System repository, created for the RecSys'24 Challenge. 

## Getting started
We advise using conda for this, then first run:
```
conda env create -f recsys_env_JE.yaml
```
Then afterwards install via pip (not available in conda)
```
pip install ebrec
```
On some devices, the library optuna gives issues with the environment. Therefore it is not included in the .yaml file. Optuna is only used in hypertuning.py. If you do want to install it we advise:
```
conda activate recsys
conda install -c conda-forge optuna
```

# Using the code
For hypertuning options, the src/mtrec/configs/hypertune.yml file can be edited. Hypertuning is simply done by the following line, or submitting run_hypertuning.job : 
```
python src/mtrec/hypertune.py
```
Training is done using src/mtrec/train_mtrec.py. This takes as an argument the name of a config.yml file that should be located in src/mtrec/configs. The python file defaults to the standard training .yml file if none is provided. So it will train when running:
```
python src/mtrec/train_mtrec.py
```
With using a job file we have:
```
sbatch jobs/train/1_train_model.job
```
Different yml files have been provided for different implementation options. Testing the performance van be done with src/mtrec/test_predictions.py which needs the correct config.yml and model run name: for example:
```
python src/mtrec/test_predictions.py --config 1_train_model.yml --run_name run1
```

## Using Snellius
All jobfiles are located in ./jobs, and a job can be run with the command:
```
sbatch ./jobs/<job-file>
```
You can see jobs in the queue using the command:
```
squeue
```
You can then use a JOB-ID to show more information about a job with the command:
```
scontrol show job <JOB-ID>
```
Lastly, you can cancel a job using:
```
scancel <JOB-ID>
```

# Credit
This repo is not fully original code, which we want to disclose. src/ebrec is the entire repo from https://github.com/ebanalyse/ebnerd-benchmark , there is also a function in in src/mtrec/NeRD_data.py that is copied from ebrec, and tweaked for our purposes. src/mtrec/gradient_surgery.py is largly the code from https://github.com/WeiChengTseng/Pytorch-PCGrad , which is again tweaked for our purposes. Inside both files there's more detail provided.

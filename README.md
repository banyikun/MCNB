# Meta_Ban

## Prerequisites: 

python 3.8.8

CUDA 11.2

torch 1.9.0

torchvision 0.10.0

sklearn 0.24.1

numpy 1.20.1

scipy 1.6.2

pandas 1.2.4

* packages.py - all the needed packages
* load_data.py - load the datasets
* run.py - run all the methods

## Methods:
"locb", "club", "sclub", "cofiba", "neuucb_one", "neuucb_ind", "meta_ban"

## Datasets:
"mnist", "movie", "yelp"

## Run:
python run.py --dataset "dataset" --method "method"

For example,  to meta_ban on movie dataset:
python run.py --dataset movie  --method meta_ban

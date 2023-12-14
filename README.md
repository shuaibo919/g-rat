G-RAT Code
### 1.Environments
> Ubuntu 18.04.6 
> python 3.9

Requirements
> torch==2.0.1
> torchmetrics==1.0.1
> tqdm==4.65.0
> pandas==2.0.3
> numpy==1.25.1

### 2.Datasets
#### Embedding
run download_embedding.sh in the data/ directory.
#### BeerAdvocate
run download_data.sh in the data/beer directory.
#### HotelReview
run download_data.sh in the data/hotel directory.

### 3.Running Example
#### 1) Real-World Setting:
Run Re-RNP in Appearance aspect(BeerAdvocate) :
```
sh script/beer/a0.sh
```
Run FR in Appearance aspect(BeerAdvocate) :
```
sh script/beer/share/a0.sh
```
Run G-RAT in Appearance aspect(BeerAdvocate) :
```
sh script/beer/guide/a0.sh
```
Similarly, Run Re-RNP in Location aspect(HotelReview) :
```
bash script/hotel/a0.sh
```
Other aspects are similar. 

#### 2) Synthetic Setting:
First run the corresponding script for saving the skew model(selector/predictor) parameters:
```
sh script/beer/pretrain_skew_{selector/predictor}_a{0,1,2}.sh
```
For example, to run the skew-selector experiments on aspect smell, we first run this:
```
sh script/beer/pretrain_skew_selector_a1.sh
```
Then, you can run this to get the result of G-RAT on all skew thresholds:
```
python run_skew_selector.py --model guide --aspect 1
```

In this repo, we implemented three models that you can choose from using --model:

| arg_name |                         method                          |
|:--------:|:-------------------------------------------------------:|
|   sep    |                         Re-RNP                          |
|  share   | [FR(Liu et al.,2022)](https://arxiv.org/abs/2209.08285) |
|  guide   |                          G-RAT                          |

More details can be founded in run_skew_selector.py and run_skew_predictor.py

#### 3) Parameter Analysis:
The code used in our parameter analysis is all in run_analysis.py, and its usage is similar to that of run_skew_selector.py.

#### 4) Other:

If you want to try different parameters, please modify the script file in the script/ directory.
# Recommender System

### Introduction

This project consists of two kind of recommender systems as well as an evaluation system to evaluate their performance.<br>
The baseline recommender system implements the item-to-item collaborative filtering.<br>
The advanced recommender system implements the matrix factorization collaborative filtering.<br>
The evaluation system evaluates a certain recommender system with the k-fold manner.

### Usage

This project provides two command line tools: recommend.py and evaluate.py. <br>
There are two models to choose: BLRS and ALS.<br>
You need to prepare a rating file with each line in the format of
\<USER_ID\> \<ITEM_ID\> \<RATING\>. <br>
Some example usages are listed below. See -h for further details.


##### Recommendation

To recommend a certain item to a certain user:
```
python3 recommend.py -m model -f file -u user_id -i item_id
```
To recommend top k items to a certain user:
```
python3 recommend.py -m model -f file -u user_id -k top_k
```


##### Evaluation

To evaluate on a fix number of users using existing split files:
```
python3 evaluate.py -m model -f file [file ...] -k K [K ...] -u num_user
```

To evaluate on all users using existing split files:
```
python3 evaluate.py -m model -f file [file ...] -k K [K ...]
```


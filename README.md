# ARRR
This is our pytorch implementation for the paper "Aspect-level Recommendation Fused with Review and Rating Representations".
The code has been tested running under Python 3.6.5. The required packages are as follows:
* pytorch == 1.10.2
* python == 3.6.3

Data
-----------------
`python preprocessing/data_preprocessing.py ` 
  * Dataset:[Amazon Product Review dataset](http://jmcauley.ucsd.edu/data/amazon/links.html)/Musical Instruments
  * generate:
   musical_instruments_5___preprocessing_log.txt
   musical_instruments_5_env.pkl
   musical_instruments_5_iid_itemDoc.npy
   musical_instruments_5_info.pkl
   musical_instruments_5_split_dev.pkl
   musical_instruments_5_split_test.pkl
   musical_instruments_5_split_train.pkl
   musical_instruments_5_uid_userDoc.npy

`python preprocessing/pretrained_vectors.py `
  * Download the pretrained word2vec embeddings: GoogleNews-vectors-negative300.bin
  * generate:
   musical_instruments_5___pretrained_vectors_log.txt
   musical_instruments_5_wid_wordEmbed.npy

Run ARRR
-----------------
`python main.py `

### Named Entity Recognition on Conll 2003


#### Requirements
* Python 3.6
* pytorch>=0.41
* numpy >=1.14.1
* scikit-learn >=0.19.1



#### Data:
1) DOCSTART is used as an ordinary word with tag as 'O'

#### Questions

1  Explain in detail the process of feature extraction including the normalization?
Ans: todo.
   

2  Describe the hyperparameter choices?
Ans:I experimented on the validation set to see which hyper parameters gives better results.
See Jupyter notebook for details.


3  Apart from LSTM if given a choice what model do you think works better? and why?
Ans: LSTM-Conditional random fields would do a good job as they apply to an entire sequence(using dynamic programming)


4  How does the batch size affect your model?
Ans: See Jupyter notebook for details. Lower batch sizes give better f1 scores.


5  Report Recall, precision, and F1 measure
Ans: 
for validation sets they are 
0.90,0.89, 0.90 for batch size 32 on 5 epochs 
0.91,0.90,0.90 for batch size 32 on 5 epochs for hidden size 250 for lstm
    



Bonus(not mandatory):

*  Report entity wise Recall, precision, and F1 measure like PER, LOC
Ans:Printed at the end of training 
*  Report effect of imbalanced dataset if any
Ans: F1,precision ,recall are all dominated by 'O' category because almost 90% of the data is category 'O'

* Use word vectors to improve NER performance
Ans: Pretrained embedding from fasttext seem to give a slight edge  over randomly initialized embeddings from pytorch
The results seem to vary slightly with each run (based on differnet random seed). further optimizations may lead 
to a better result.

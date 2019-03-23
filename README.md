### Named Entity Recognition on Conll 2003


#### Requirements.
You should have the following installed.
* Python 3.6
* pytorch>=0.41
* numpy >=1.14.1
* scikit-learn >=0.19.1

[install pytorch as per your device](https://pytorch.org/get-started/locally)


#### Pretrained Word embedding:
If you want to use pretrained embeddings

Download [fasttext](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)  
Unzip and put it in 'data' folder.

If you use pretrained word embeddings use a smaller learning rate.

#### Usage:
1) Install the requirements 
2) Run ```python main.py ``` to see the results on model trained with default arguments. 
3) Different arguments can be specified while running. run ```python main.py --help``` for different arguments that can be passed 

#### Device
1) It works with either cpu or gpu seamlessly. I tested it on gcp where I ran ablation studies  
2) I suggest using gpu when using pretrained word vectors or large lstm hidden size.  

#### Process 
1) I explain the process and overview in process.ipynb
#### Questions

1  Explain in detail the process of feature extraction including the normalization?  
**Ans**: I downloaded the data from [here](https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003).
I didn't perform any normalization step. I felt lowercasing or stemming will lose the properties that I care about in NER task.
I didn't use the pos tags as well. My aim was to build an end to end model from words in sentences to tags.
Although normalization and using pos tags may producing different results.But that is left for future work.

   

2  Describe the hyperparameter choices?  
**Ans**:I experimented on the validation set to see which hyper parameters gives better results.
Lower batch size gives better result
larger hidden size is good upto certain size then it becomes bad.
See Ablation studies Jupyter notebook for details.


3  Apart from LSTM if given a choice what model do you think works better? and why?

**Ans**: LSTM-Conditional random fields would do a good job as they apply to an entire sequence(using dynamic programming)
Basic LSTM is similarly to greedy approach. Where as a LSTM-CRF is a more dp approach.

4  How does the batch size affect your model?

**Ans**:  Lower batch sizes give better f1 scores. See Ablation studies Jupyter notebook for details.
The results change sligtly with each run , even after setting a random seed


5  Report Recall, precision, and F1 measure  

**Ans** :  For different setting they are different.    
You can see them at the end of training when you run ```python main.py```  with specific arguments   



**Bonus(not mandatory)**:

1 Report entity wise Recall, precision, and F1 measure like PER, LOC  
**Ans** :Printed at the end of training   

2 Report effect of imbalanced dataset if any    
**Ans** : F1,precision ,recall are all dominated by 'O' category because almost 90% of the data is category 'O'.
But with training it gets better for other categories. The more number of samples a category has the better the model learns.

3 Use word vectors to improve NER performance    
**Ans**: Pretrained embedding from fasttext seem to give a slight edge  over randomly initialized embeddings from pytorch.
While Pretrained embeddings give a better f1 micro score,non pretrained embeddings give better macro score with only a small dip in micro score
The results seem to vary slightly with each run (based on differnet random seed). further optimizations may lead 
to better results.

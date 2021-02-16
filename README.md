# Data Scientist - Technical Assignment 

### Assignment 
A sample of labeled search term category data is provided in the CSV file, trainSet.csv. The file contains the two columns, the search term and the search term category. The search term category has been indexed. There are 606,823 examples in the data set with 1,419 different search term categories. There are roughly 427 examples in each category. 

> How to run it?
```sh
start.sh
```

My accuracy result based on cross-validation is 0.53

candidateTestSet with predictions is here: https://github.com/marynadorosh/test_task/blob/main/data/processed/candidateTestSet.csv

### Preprocessing and model description

* Since the dataset is imbalanced, I expanded categories, which contain less than average amount of examples, by generating new search term changing words to synonyms. 
* Each search term was replaced by average FastText word embedding.
* I selected Linear Support Vector Machine because it can work reasonable fast with a huge amount of data, also I tried Logistic Regression and Support Vector Machine with other kernels. 

### Runtime complexity:
* Text augmentation - O(n^2) in worse case
* Linear Support Vector Machine - O(n)

Since I selected only linear models, which can't catch non-linear dependencies and ignore word orders, the easiest way for improving is trying more a complex model like neural networks. 


### Possible improvements 
* Translate a dataset in one language or create a new model for every languages
* Define as features named entity, like countries, names, brand, etc.
* Generate features using some user context (country, browser, previous queries, etc.)
* Use transformer-based neural network.
* Add logging, config, tests, more data, more models, more fun:)

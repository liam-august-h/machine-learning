# Machine Learning Engineer Nanodegree
## Capstone Project
liam 
March 4th, 2018

## I. Definition
### Project Overview
**Mercari Price Suggestion Challenge** is a competition to predict products prices based on provided user-inputted text description of products, including details like product category name, brand name and item condition. It is a **regression problem** based on text features and category features. The major challenge of the project how to understand user inputted information with **Natural Language Processing**(NLP) technical.

According to wikipedia, **Natural Language Processing**(NLP) is a field of computer science, artificial intelligence concerned with the interactions between computers and human languages and concerned with programming computers to process large natural language data. The history of NLP generally started in the 1950s. Most of the notable early successes occurred in the field of machine translation with more complicated statistical models developed by IBM research. In recent years, there has been a flurry of result showing deep learning techniques, like Yoav Goldbery, A Primer on Neural Network Models for Natural Language Processing, achieving state-of-the-art result in many natural language tasks, like language modeling, parsing, document classification and so on.

Natural Language Understanding**](https://en.wikipedia.org/wiki/Natural_language_understanding) is a subtopic of *NLP* in artificial intelligence that deals with machine reading comprehension. Throughout the years various attempts at processing natural language or English-like sentences presented to computers have taken place ate varying degrees of complexity. Some attempts have not resulted in systems with deep understanding, but have helpd overall system usability. Regardless of the approach used, most natural language understanding system share some common components. The system needs a lexicon of the language and a parser and a grammar rules to break sentences into an internal representation.

> Several discussion about machine learning and natural language process 
>
> - *Sebastiani, F., 2002. Machine learning in automated text categorization. ACM computing surveys (CSUR), 34(1), pp.1–47.*
> - *Yang, Y. & Pedersen, J.O., 1997. A comparative study on feature selection in text categorization. In ICML. pp. 412–420.*
> - *Kohavi, R. & others, 1995. A study of cross-validation and bootstrap for accuracy estimation and model selection. In Ijcai. pp. 1137–1145.*
> - *Deerwester, S. et al., 1990. Indexing by latent semantic analysis. Journal of the American society for information science, 41(6), p.391.*

### Problem Statement
The goal of the project is to use **regression** algorithm to predict products prices, and such tasks are involved to achieve the goal:

- Download and preprocess **Mercari Price Suggestion Challenge** training dataset and test dataset

- Data cleaning and feature engineering.  Since most features are text type, the preprocessing includes tokenizing, lowercase, removing punctuation and so on. T

- There are problems need to be considered carefully:

  - there are almost 1.5 million observations in training set, which needs Out-of-Core technical to deal with that kind of huge data without any memory exception
  - how to pick right minimum and maximum document frequencies in order to reduce words account and reduce dimension value. Since a product price could be consider as *base_price* + *unique_price*. *base_price* is a normal price of a product decided by product category. For example, a *electrical device* is averagely  expensive than a *cloth*. *unique_price* is an additional item on price decided by some special quality of the product, like  shipping, brand new, handmade, signatures of famous people and so on. *base_price* will be represented by some words with high document frequencies but *unique_price* will be represented by some words with low document frequencies
  - some fields, like *shipping* and *item condition*, are category data. how to form new features with text data and category data is the problem. Maybe change *shipping* and *item condition* into text, or find a way to combine dense matrix and sparse compress matrix without out-of-memory exception

- Setup a baseline with linear regression algorithm and estimate more regression algorithms, linear and non-linear

- Tuning parameter and score

  ​

### Metrics
The evaluation metric for this competition is [Root Mean Squared Logarithmic Error](https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError), which is provided by the competition.

The RMSLE is calculated as
$$
\epsilon = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }
$$
Where:

$\epsilon$ is the RMSLE value (score)

$n$ is the total number of observations in the (public/private) data set,

$p_i$ is your prediction of price, and

$a_i$ is the actual sale price for $i$ 

$\log(x)$ is the natural logarithm of $x$

## II. Analysis
### Data Exploration
The dataset is provided by competition. There are *train.tsv* for training and *test.tsv* for testing. *test.csv* has almost same columns with *training.csv* except *price* which is the regression target. 

There are following fields in the dataset:

- *XXX_id*. an id of training record
- *name*. the name of product. it is a text field. no missing
- *item_condition_id*. to indicate a product condition. 5 is the best and 1 is the worst. it is a numeric category field. no missing
- *category_name*. the category of the product. in a form like "xxx/yyy/zzz", "xxx", "yyy" and "zzz" represent different category level. For example, a necklaces's *category_name* is *Women/Jewelry/Necklaces*. it is text. there is missing data. Around TODO records don't have such field
- *brand_name*. the brand of the product. it is text. there is missing data. Around TODO records don't have such filed
- *price*. the price of the product. it is numeric. also the regression target. the *price* fields has a long tail distribution. 95% products prices are lower than TODO
- *shipping*. a numeric value to indicate whether the product is shipping free or not. 1 means yes. 0 means no. no missing
- *item_description*. some user-inputed text to describe the product. there is missing data. Around TODO records don't have such filed

### Exploratory Visualization
It is a distribution of *price* and its *boxplot*. Based on below, 

- *price* is a long tail, skewed distribution
- the mean value of *price* is around 30, and 75% products *price* are lower than 30
- the maximum *price* is above 2,000, it is 100x higher than mean value

![price_distribution](/Users/lianghe/Documents/udacity/machine_learning/machine-learning/projects/capstone/img/price_distribution.png)

### Algorithms and Techniques

Because there are text fields, *word embeding* will be used to process such fields *name*, *brand_name*, *category* and *item_description*.  *Bag of words* is the first option.

Then it is a *regression* problem. *LinearRegression* is a good start point. *SVR* is a good candidate in case it is a non-linear regression model.

### Benchmark	
To use *LinearRegression* result as a benchmark

## III. Methodology
### Data Preprocessing
#### Train_id ####

it is just an index and going to drop all *train_id* to save memory storage.

#### Missing Data ####

Since only few observations lost fields *category_name*, *brand_name* and *item_condition*, going to simply drop all NAs.

Since some second products may not have an identical brand, to predict *brand_name* with *name* and *item_condition* will not be a good move. In other hand, if there are several key words in *name* and *item_condition* to indicate *brand_name*, they should be able to used to indicate *price* too

#### Category ####

All value of *category_name* are in a form like *XXX/YYY/ZZZ". They are an important feature to predict the price of a product. During text processing, *XXX/YYY/ZZZ" will be thought as a combination of word and punctuation and might be filtered out by a pure word combination pattern. To split *category_name* with */* and save *XXX*, *YYY* and *ZZZ* separately.

#### Shipping and Item Condition ####

*shipping* and *item_condition* are category data. Using their numeric value is a nature way. Considering the target of to combine them with other text features, maybe to translate them into strings and to use a sparse matrix to store is more reliable than to try to combine a sparse matrix and a dense matrix.

#### Text Processing ####

- Remove all non alphanumeric characters
- Tokenize text
- convert all the characters to lowercase

#### Feature ####

to combine all text fields, includes *name*, *brand_name*, *item_condition* and spitted  *category_name* into one. Plus *shipping* and *item_condition* to form new features.

###   Implementation
#### Embeddings

Using *Bags of Words* model to project texts into space. The only concern in this step is how to avoid memory exception. 

*CounterVectorizer* and *TfIdfVectorizer* are the simplest choice. Because of the huge number of observations and their in memory vocabulary, try to use *df* to ignore irrelevant words and achieve dimensions reduction in order to save memory. As mentioned above, the price information are connected with *base price* and *unique price*. In this project, *base price* is related with *category* and *brand_name*. Those are all words with high document frequency. *unique price* is related with *name* and *item_description*. Those are words with low document frequency. So, using document frequency to filter out to avoid memory exception is not a good way. 

*HashingVectorizer* is using *scipy.sparse* matrix to hold token occurrence counts. With the hash trick, it is very low memory scalable to large dataset as there is no need to store a vocabulary dictionary in memory.

#### Dimensionality Reduction

Using *SelectPercentile* to reduce feature account.

#### Regression

*LinearRegression* is used to setup the baseline. 

Using *Ridge* since the whole feature matrix is a big sparse matrix with high dimensions. The penalty is able to avoid variance.

Using *ElasticNet* to handle sparse module with L1 regularization and to maintain regularization properties with L2.

Using *SVM* since the price of product might be a non-linear model of some words.  And try *RamdomForesetTree* for the same reason and a better performance.

*RandomForestRegressor* has the best score.

### Refinement
Using *GridSearchCV* to *parameter* tuning.  

The main parameters to adjust is *n_estimators* and *max_features*. *n_estimators* is the number of trees in the forest, the larger the better, but also longer the time consumption. *n_estimators* will stop getting significantly better beyond a critical number of trees. *max_features* is the size of the random subsets of features to consider when splitting a node. The lower the greater the reduction of variance and greater the bias.

## IV. Results
### Model Evaluation and Validation
Split *train.tsv* dataset into *training* and *test* sub dataset.Use *training subset* to train regressor and use *test* subset to predict. Then use *rmsle* as score function to estimator models. 

*RandomForestRegressor* has the highest score and is chosen as the final model.

### Justification
The baseline, *LinearRegressor* has *rmsle* score value 2.4 and the final model *RandomForestRegressor* has *rmsle* score value 0.71. 

It turns out, a non-linear model will fit the project better. The final model is better than baseline.

## V. Conclusion
### Reflection
*HashingVectorizer* + *RandomForestRegressor* are the final problem solution. Using *HashingVectorizer* to solve *Out-Of-Core* problem and *RandomForest* to deal with non-linear model.

The *Out-Of-Core* problem is the most difficult part. It is not only slow down the whole process but also give people a false hope that it could be fixed if with a strong machine.

### Improvement

There are some spelling mistake and abbreviations should be taken care of. To combine misspelled or alternately spelled words to a single representation. Consider lemmatization and stemming is another choice to clean the text data. Using *PCA* or *SparsePCA* to reduce dimensions and improve performance.

Basically, a high dimension sparse matrix is the disadvantage of *Bag-of-Words*. To turn it over, we need a better text model, like *word2vec*, which needs a *Neural Network* solution.
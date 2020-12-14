# JHU_CS675_Project_SentimentAnalysisonMovieReviews
This document will introduce the main codes that we have used in each part. For details, please see the ipynb notebooks in the python_notebook folder
## What has been uploaded in the Github repo
There are two folders
### image
> image name type 1: vectorization_model_item

e.g.: tfidf_lr_tests: performance of logistic model with tdidf vectorization on nine testing sets

> image name type 2: vectorization_item

e.g.: tfidf_top15: the ranking of top 15 keywords in terms of TFIDF scores

### Python notebook
There are six ipynb notebooks in this section:

4 notebooks on modelling with different vectorizations

Each ipynb notebook can be runed independently from the beginning, as they include the complete sections from loading data, pre-processing, split dataset, vectorization, modeling, performance measure...and alll intermediate results

> tfidf.ipynb: all models with tfidf vectorization

> BagOfWord_Ngram: all models with baseline (bag of word) and ngram vectorization

> word2vec1

> word2vec2

Fifth: We also uploaded a python notebook using which we webclawed extra testing set.

Sixth: Final Project Writeup whcih follows the instructor's templete and only includes main results and analysis

## Problem Introduction

There are thousands of movies coming out each year. Movie reviews reflect the quality of movies, and influence many people in their choice of movie-watching. In this project, we want to perform a thorough sentiment analysis to identify the polarity of textual reviews. This research work can hopefully help viewers decide whether to watch a newly released movie or not. It may also be of interest to the movie industry to tell what kind of movies the market will like and help recommend movies to users based on previous reviews. Each input is a paragraph of movie review consisting of several English sentences. We will train and develop different SVM, neural network and LSTM models to predict whether the movie review is positive or negative. Then, with the trained models, we will investigate what are key features in a movie review that reveals most of its attitude (e.g., frequency of particular key words). Furthermore, we will conduct a comparison analysis in following two aspects: First, we will compare different texture feature engineering techniques such as word2vec, N-Gram, TFIDF. Second, we will compare those best performing models in each classes based not only on their test accuracy, time complexity, but also on their fairness and interpretability (details explained in the deliverables section of our writeup notebook).

## Pre-Processing

Different vectorization requires different pre-processing, so please see each ipynb notebook in the python_notebook folder for this part.

## For models with TFIDF
### TFIDF Transformation
```
tfidf_vec = TfidfVectorizer(sublinear_tf = True, min_df=5, max_features = 1500)
tfidf_fit = tfidf_vec.fit(X_train) # perform the tfidf feature engineering only on the training set
X_train_tfidf = tfidf_fit.transform(X_train)
X_valid_tfidf = tfidf_fit.transform(X_valid)
X_tv_tfidf = tfidf_fit.transform(X_tv)
```
### PCA
Based on data exploration, we reduce the dimension to 400.
```
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=400, n_iter=30, random_state=2020)
svd_tfidf_fit = svd.fit(X_train_tfidf)
X_tv_tfidf = svd_tfidf_fit.transform(X_tv_tfidf)
X_train_tfidf = svd_tfidf_fit.transform(X_train_tfidf)
X_valid_tfidf = svd_tfidf_fit.transform(X_valid_tfidf)
y_valid_int=y_valid.astype('int')
y_train_int=y_train.astype('int')
```

## For models with Ngram
### Ngram Transformation
```
from sklearn.feature_extraction.text import CountVectorizer
ngram_vectorizer = CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=(5, 5), max_features=1000, lowercase=True)
ngram_fit = ngram_vectorizer.fit(X_train)
X_train_ngram = ngram_fit.transform(X_train)
X_valid_ngram = ngram_fit.transform(X_valid)
X_tv_ngram = ngram_fit.transform(X_tv)
X_test_ngram = ngram_fit.transform(X_test)
```

## For models with word2vec
### word2vec Transformation
In general, there are two types of architecture options: skip-gram (default) and CBOW (continuous bag of words). Most of time, skip-gram is little bit slower but has more accuracy than CBOW. CBOW is the method to predict one word by whole text; therefore, small set of data is more favorable. On the other hand, skip-gram is totally opposite to CBOW. With the target word, skip-gram is the method to predict the words around the target words. The more data we have, the better it performs. As the architecture, there are two training algorithms for Word2Vec: Hierarchical softmax (default) and negative sampling. We will be using the default.
```
model = word2vec.Word2Vec(sentence, workers = num_processor, 
                         size = num_features, min_count = min_count,
                         window = context, sample = downsampling)
```
### Vector Averaging
The purpose of this function is to combine all the word2vec vector values of each word in each review if each review is given as input and divide by the total number of words. Each word can be represented as number of feature dimension space vector
```
def makeFeatureVec(review, model, num_features):
    featureVec = np.zeros((num_features,), dtype = "float32")
    word_index = set(model.wv.index2word)
    nword = 0
    for word in review:
        if word in word_index:
            nword += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nword)        
    return featureVec
```
While iterating over reviews, add the vector sums of each review from the function "makeFeatureVec" to the predefined vector whose size is the number of total reviews and the number of features in word2vec. The working principle is basically same with "makeFeatureVec" but this is a review basis and makeFeatureVec is word basis.
```
def getAvgFeatureVec(clean_reviews, model, num_features):
    review_th = 0
    reviewFeatureVecs = np.zeros((len(clean_reviews), num_features), dtype = "float32")
    for review in clean_reviews:
        reviewFeatureVecs[int(review_th)] = makeFeatureVec(review, model, num_features) 
        review_th += 1
    
    return reviewFeatureVecs
```

## Logistic Regression and Gridsearch
```
lr = LogisticRegression(random_state = 2020)
lr_para_grid = {
    'penalty':['l1','l2'],
    'C':[0.1, 1, 10, 15, 20],
    'solver':['newton-cg','lbfs','liblinear','sag','saga']
    }

lr_gs = GridSearchCV(lr, param_grid = [lr_para_grid], cv = pds, scoring = 'roc_auc', n_jobs = -1, verbose = 1)
tic()
lr_gs.fit(X_tv, y_tv_int)
toc()
lr_best = lr_gs.best_estimator_
print(lr_gs.best_params_)
tic()
y_pred_lr_test = lr_best.predict(X_test)
y_pred_lr_runtime_1_100=lr_best.predict(X_runtime_1_100)
y_pred_lr_runtime_101_600=lr_best.predict(X_runtime_101_600)
y_pred_lr_action=lr_best.predict(X_action)
y_pred_lr_adventure=lr_best.predict(X_adventure)
y_pred_lr_animation=lr_best.predict(X_animation)
y_pred_lr_biography=lr_best.predict(X_biography)
y_pred_lr_comedy=lr_best.predict(X_comedy)
y_pred_lr_horror=lr_best.predict(X_horror)
y_pred_lr_romance=lr_best.predict(X_romance)
y_pred_lr_scifi=lr_best.predict(X_scifi)
toc()
```

## SVM Model and Gridsearch
```
svm = LinearSVC(random_state=2020)
svm_para_grid = {
    'penalty':['l1','l2'],
    'loss':['hinge','squared_hinge'],
    'C': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 1]
}
svm_gs = GridSearchCV(svm, param_grid = [svm_para_grid], verbose = 1, cv = pds, n_jobs = -1, scoring = 'roc_auc')
tic()
svm_gs.fit(X_tv, y_tv_int)
toc()
svm_best = svm_gs.best_estimator_
print(svm_gs.best_params_)
svm_gs.best_score_
tic()
y_pred_svm_test = svm_best.predict(X_test)
y_pred_svm_runtime_1_100=svm_best.predict(X_runtime_1_100)
y_pred_svm_runtime_101_600=svm_best.predict(X_runtime_101_600)
y_pred_svm_action=svm_best.predict(X_action)
y_pred_svm_adventure=svm_best.predict(X_adventure)
y_pred_svm_animation=svm_best.predict(X_animation)
y_pred_svm_biography=svm_best.predict(X_biography)
y_pred_svm_comedy=svm_best.predict(X_comedy)
y_pred_svm_horror=svm_best.predict(X_horror)
y_pred_svm_romance=svm_best.predict(X_romance)
y_pred_svm_scifi=svm_best.predict(X_scifi)
toc()
```

## For Model Evaluation and Fairness Comparison based on Different Metrics
```
def evaluate_pred(ytrue, ypred):
  TN, FP, FN, TP = confusion_matrix(ytrue, ypred).ravel()
  Accuracy = round((TP + TN)/(TP + TN + FP + FN),3)
  Precision = round(TP/(TP + FP),3)
  Recall = round(TP/(TP + FN),3)
  Specificity = round(TN/(TN + FP),3)
  FPR = round(FP/(FP + TN),3)
  F1 = round(2*(Precision * Recall)/(Precision + Recall),3)
  Balanced_Accuracy = round((Precision + Specificity)/2,3)
  print("TP = "+str(TP))
  print("FP = "+str(FP))
  print("FN = "+str(FN))
  print("TN = "+str(TN))
  print("Accuracy = "+str(Accuracy))
  print("Precision = "+str(Precision))
  print("Recall = "+str(Recall))
  print("Specificity = "+str(Specificity))
  print("False_Positive_Rate = "+str(FPR))
  print("F1_Score = "+str(F1))
  print("Balanced_Accuracy = "+str(Balanced_Accuracy))
  res=pd.DataFrame([TN, FP, FN, TP, Accuracy, Precision, Recall, Specificity, FPR, F1, Balanced_Accuracy])
  return(res)
 ```

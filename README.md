# JHU_CS675_Project_SentimentAnalysisonMovieReviews
This document will introduce the main codes that we have used in each part. For details, please see the ipynb notebooks in the Python notebook folder
## What has been uploaded in the Github repo
There are two folders
>image

>Python notebook

## Pre-Processing

Different vectorization requires different pre-processing, so please see each ipynb notebook in the Python notebook folder for this part.

## For models with TFIDF Vecotrization

### TFIDF Transformation
```
tfidf_vec = TfidfVectorizer(sublinear_tf = True, min_df=5, max_features = 1500)
tfidf_fit = tfidf_vec.fit(X_train) # perform the tfidf feature engineering only on the training set
X_train_tfidf = tfidf_fit.transform(X_train)
X_valid_tfidf = tfidf_fit.transform(X_valid)
X_tv_tfidf = tfidf_fit.transform(X_tv)
```
### PCA
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
### Logistic Regression and Gridsearch
```
lr = LogisticRegression(random_state = 2020)
lr_para_grid = {
    'penalty':['l1','l2'],
    'C':[0.1, 1, 10, 15, 20],
    'solver':['newton-cg','lbfs','liblinear','sag','saga']
    }

tfidf_lr_gs = GridSearchCV(lr, param_grid = [lr_para_grid], cv = pds, scoring = 'roc_auc', n_jobs = -1, verbose = 1)
tic()
tfidf_lr_gs.fit(X_tv_tfidf, y_tv_int)
toc()
tfidf_lr_best = tfidf_lr_gs.best_estimator_
print(tfidf_lr_gs.best_params_)
tic()
y_pred_tfidf_lr_test = tfidf_lr_best.predict(X_test_tfidf)
y_pred_tfidf_lr_runtime_1_100=tfidf_lr_best.predict(X_tfidf_runtime_1_100)
y_pred_tfidf_lr_runtime_101_600=tfidf_lr_best.predict(X_tfidf_runtime_101_600)
y_pred_tfidf_lr_action=tfidf_lr_best.predict(X_tfidf_action)
y_pred_tfidf_lr_adventure=tfidf_lr_best.predict(X_tfidf_adventure)
y_pred_tfidf_lr_animation=tfidf_lr_best.predict(X_tfidf_animation)
y_pred_tfidf_lr_biography=tfidf_lr_best.predict(X_tfidf_biography)
y_pred_tfidf_lr_comedy=tfidf_lr_best.predict(X_tfidf_comedy)
y_pred_tfidf_lr_horror=tfidf_lr_best.predict(X_tfidf_horror)
y_pred_tfidf_lr_romance=tfidf_lr_best.predict(X_tfidf_romance)
y_pred_tfidf_lr_scifi=tfidf_lr_best.predict(X_tfidf_scifi)
toc()
```
### Logistic Regression and Gridsearch
```
svm = LinearSVC(random_state=2020)
svm_para_grid = {
    'penalty':['l1','l2'],
    'loss':['hinge','squared_hinge'],
    'C': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 1]
}
tfidf_svm_gs = GridSearchCV(svmm, param_grid = [svm_para_grid], verbose = 1, cv = pds, n_jobs = -1, scoring = 'roc_auc')
tic()
tfidf_svm_gs.fit(X_tv_tfidf, y_tv_int)
toc()
tfidf_svm_best = tfidf_svm_gs.best_estimator_
print(tfidf_svm_gs.best_params_)
tfidf_svm_gs.best_score_
tic()
y_pred_tfidf_svm_test = tfidf_svm_best.predict(X_test_tfidf)
y_pred_tfidf_svm_runtime_1_100=tfidf_svm_best.predict(X_tfidf_runtime_1_100)
y_pred_tfidf_svm_runtime_101_600=tfidf_svm_best.predict(X_tfidf_runtime_101_600)
y_pred_tfidf_svm_action=tfidf_svm_best.predict(X_tfidf_action)
y_pred_tfidf_svm_adventure=tfidf_svm_best.predict(X_tfidf_adventure)
y_pred_tfidf_svm_animation=tfidf_svm_best.predict(X_tfidf_animation)
y_pred_tfidf_svm_biography=tfidf_svm_best.predict(X_tfidf_biography)
y_pred_tfidf_svm_comedy=tfidf_svm_best.predict(X_tfidf_comedy)
y_pred_tfidf_svm_horror=tfidf_svm_best.predict(X_tfidf_horror)
y_pred_tfidf_svm_romance=tfidf_svm_best.predict(X_tfidf_romance)
y_pred_tfidf_svm_scifi=tfidf_svm_best.predict(X_tfidf_scifi)
toc()
```
## For models with Ngram 

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
### Logistic Regression and Gridsearch

### SVM and Gridsearch
```
# LinearSVC
tic()
sv = LinearSVC(random_state=2020)

param_grid1 = {
    'loss':['hinge', 'squared_hinge'],
    'class_weight':[{1:2},'balanced'],
    'C': [0.5,1,10,20],
    'penalty':['l2']
}

gs_sv = GridSearchCV(sv, param_grid = [param_grid1], verbose = 1, cv = pds, n_jobs = 1, scoring = 'roc_auc' )
gs_sv.fit(tvDataAvg, y_tv_int)
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)
print(gs_sv.best_score_)
toc()
tic()
y_svm1 = gs_sv.predict(testDataAvg)
y_svm2 = gs_sv.predict(testDataAvg_action)
y_svm3 = gs_sv.predict(testDataAvg_adventure)
y_svm4 = gs_sv.predict(testDataAvg_animation)
y_svm5 = gs_sv.predict(testDataAvg_biography)
y_svm6 = gs_sv.predict(testDataAvg_comedy)
y_svm7 = gs_sv.predict(testDataAvg_horror)
y_svm8 = gs_sv.predict(testDataAvg_romance)
y_svm9 = gs_sv.predict(testDataAvg_scifi)
y_svm10 = gs_sv.predict(testDataAvg_runtime_1_100)
y_svm11 = gs_sv.predict(testDataAvg_runtime_101_600)
toc()
```

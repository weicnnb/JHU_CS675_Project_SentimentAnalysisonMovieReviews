# JHU_CS675_Project_SentimentAnalysisonMovieReviews
This document will introduce the main codes that we have used in each part. For details, please see the ipynb notebooks in the Python notebook folder


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
svmm = LinearSVC(random_state=2020)
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

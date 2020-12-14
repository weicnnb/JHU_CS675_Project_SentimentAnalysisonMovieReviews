# JHU_CS675_Project_SentimentAnalysisonMovieReviews



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

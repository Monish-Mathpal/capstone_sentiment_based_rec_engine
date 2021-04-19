import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import nltk
from gensim.models import word2vec
from sklearn.metrics import f1_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import pairwise_distances 
from sklearn import preprocessing

#read raw data
df_raw = pd.read_csv("sample30.csv")

#check missing values by columns
df_raw.isna().sum()

#keep relevant columns for the text processing
feature_list = ['reviews_text','reviews_title', 'user_sentiment']

#make copy of the orignal data
df_clean = df_raw[feature_list].copy()

#merge or concat reviews text and reviews title columns
df_clean['reviews'] = df_clean['reviews_text'] + df_clean['reviews_title']

#delete reviews text and title columns and kee the merged reviews column
del df_clean['reviews_text']
del df_clean['reviews_title']

# drop na values
df_clean.dropna(axis='rows', inplace=True)

target_names = df_clean['user_sentiment'].unique()

# preprocess data
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    doc = [w.lemma_ for w in doc if not w.is_stop and not w.is_punct and w.text != "."]
    
    return ' '.join(doc)


# apply preprocess function on reviews column
df_clean['reviews'] = df_clean['reviews'].apply(preprocess_text)

def get_model(Model, train_dat = None, label_dat=None):
    model = Model
    model.fit(train_dat, label_dat)
    return model

def get_model_metrics(model, test_dat=None):
    y_pred = model.predict(test_dat)
    score = f1_score(y_test, y_pred, average = 'weighted')
    print(classification_report(y_pred, y_test, target_names=['Positive','Negative'])) 
    print("f1 score: {}".format(score))    
    

def get_bow(dat):
    bow_model = vectorizer.fit(dat)
    return bow_model

def save_model(model_obj, filename):
    with open(filename+".pkl","wb") as f:
        classifier_pickled_object = pickle.dump(model_obj, f)
    

# create tfidf vector through train and test data
tf_idf_obj = TfidfVectorizer(ngram_range=(1, 2), max_df=1, max_features=13000)

def transform_data_to_tfidf(dat):
    X_dat = tf_idf_obj.fit_transform(dat['reviews']).toarray()
    y_dat  = dat['user_sentiment']
    return (X_dat, y_dat)

X = df_clean['reviews'].values
y = df_clean['user_sentiment'].values

# Splitting Dataset into train and test set 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# vectorizer = CountVectorizer(max_features=13000)
# get_bow(list(X_train))
# vectorizer.vocabulary_


# vect_X_train = vectorizer.transform(list(X_train))
# vect_X_test = vectorizer.transform(list(X_test))

tf_idf_vectr_train = tf_idf_obj.fit_transform(X_train).toarray()
tf_idf_vectr_test  = tf_idf_obj.transform(X_test).toarray()
save_model(tf_idf_obj,"tf_idf_vect")

# # # apply smote to handle class imbalancing issue
# oversample = SMOTE(random_state=2)
# vect_X_train_smote, vect_y_train_smote = oversample.fit_resample(vect_X_train, y_train.ravel())

oversample = SMOTE(random_state=2)
X_train_smote, y_train_smote = oversample.fit_resample(tf_idf_vectr_train, y_train.ravel())


# Logistic Regression
LR = LogisticRegression( multi_class = 'multinomial', solver = 'newton-cg', C = 0.1, n_jobs = -1, random_state = 42, class_weight='balanced')


# model_obj = get_model(LR, vect_X_train_smote, vect_y_train_smote)
model_obj = get_model(LR, X_train_smote, y_train_smote)
get_model_metrics(model_obj, tf_idf_vectr_test)

save_model(model_obj, "classifier")


## Recommendation Model
#read raw data
df_raw = pd.read_csv("sample30.csv")

#keeping the relevant columns
df_clean = df_raw[['name', 'reviews_username', 'reviews_rating', 'reviews_title', 'reviews_text']].copy()
df_clean['reviews'] = df_raw['reviews_text'] + df_raw['reviews_title']
df_clean.drop(['reviews_title', 'reviews_text'], axis='columns', inplace=True)
df_clean.dropna(axis='rows', inplace=True)

le = preprocessing.LabelEncoder()

# encoding string based columns
le.fit(df_clean['name'])
df_clean['product_id'] = le.transform(df_clean['name'])
le.fit(df_clean['reviews_username'])
df_clean['user_id'] = le.transform(df_clean['reviews_username'])

train, test = train_test_split(df_clean, test_size=0.33, random_state=42)

user_prod_matrix = train.pivot_table(index='user_id', columns='product_id', values='reviews_rating').fillna(0)
user_corr = 1- pairwise_distances(user_prod_matrix, metric='cosine')
user_corr[np.isnan(user_corr)] = 0

# normalizing user_prod matrix
mean = np.mean(user_prod_matrix, axis=1)
user_prod_matrix_norm = (user_prod_matrix.T-mean).T

# adjust cosine similarity
user_corr = 1 - pairwise_distances(user_prod_matrix_norm.fillna(0), metric='cosine')
print(user_corr)

# consider only positvely corelated users
user_corr[user_corr<0] = 0
print(user_corr)

# make predictions
user_predicted_ratings = np.dot(user_corr, user_prod_matrix)
print(user_predicted_ratings)

# identify products which are not rated by products
user_not_rated_prods_train = train.copy()
user_not_rated_prods_train['reviews_rating'] = user_not_rated_prods_train['reviews_rating'].apply(lambda x: 0 if x>=1 else\
                                                                                                  1)

user_not_rated_prods_train = user_not_rated_prods_train.pivot_table(index='user_id', columns='product_id', \
                                                                    values='reviews_rating').fillna(1)

user_final_rating = np.multiply(user_predicted_ratings, user_not_rated_prods_train)

save_model(user_final_rating, "user_final")

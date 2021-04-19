from flask import Flask,jsonify,request, render_template
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pickle
import pandas as pd

# from sklearn.externals import joblib
app = Flask(__name__)
class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])

    @app.route("/", methods=['POST','GET'])
    def index():
        form = ReusableForm(request.form)
        with open('classifier.pkl', 'rb') as pickle_file:
            model_obj = pickle.load(pickle_file)
        
        with open('cleaned_data.pkl', 'rb') as pickle_file:
            dat_clean_obj = pickle.load(pickle_file)
        
        with open('vect.pkl', 'rb') as pickle_file:
            tf_idf_obj = pickle.load(pickle_file)
        
        with open('user_final.pkl', 'rb') as pickle_file:
            user_final_rating = pickle.load(pickle_file)
        
        # dat_clean_obj = pickle.load('cleaned_data.pkl')
        # tf_idf_obj = pickle.load('tfidf_model.pkl')
        # user_final_rating = pickle.load('user_final_rating_model.pkl')
        
        if(request.method == 'POST'):
            ip=int(request.form['name'])
            # ip = request.get_json()
            # get user id
                
            # generate top 20 recommendation of products the user id
            top_20_recommendation = user_final_rating.iloc[ip].sort_values(ascending=False)[0:20]
            clean_dat = pd.merge(top_20_recommendation, dat_clean_obj,left_on='product_id',right_on='product_id', how = 'left')
            clean_dat['predicted_sentiment'] = model_obj.predict(tf_idf_obj.transform(clean_dat['reviews']).toarray())
            # create top 5 strongly positive recommnedation for the user strongly positive sentiment for t
            print(clean_dat['predicted_sentiment'].value_counts())
            print(top_20_recommendation)
            top_5_rec = clean_dat[clean_dat.loc[:,'predicted_sentiment']=='Positive'].groupby('product_id')['predicted_sentiment'].\
                value_counts().sort_values(ascending=False)[:5].index
            
            prod_ind = []
            for ind in top_5_rec:
                prod_ind.append(ind[0])
            
            return jsonify(list(dat_clean_obj.loc[dat_clean_obj['product_id'].isin(prod_ind), :]['name'].unique()))
        else:
            return  render_template('home.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
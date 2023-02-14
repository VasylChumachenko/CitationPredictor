#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import matplotlib.pyplot as plt

def Stem_data(data, column_name):
    stemmer = SnowballStemmer("english")
    data['stemmed'] = data[column_name].apply(lambda x : filter(None, 
                                                                x.split(" ")))
    data['stemmed_2'] = data['stemmed'].apply(lambda x : [stemmer.stem(y) for y in x])
    data[column_name]=data['stemmed_2'].apply(lambda x : " ".join(x))
    data.drop(['stemmed', 'stemmed_2'], axis = 1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

class CitationPredictor:
    def __init__(self, path_to_model, path_to_data, 
                 path_to_vocabulary, path_to_features):
        self.model=self._load_model(path_to_model)
        self.raw_data=self._load_demo(path_to_data)
        self.vocabulary=self._load_vocabulary(path_to_vocabulary)
        self._important_features=list(pd.read_csv(path_to_features,
                                                  index_col = [0])['0'])
        self.titles=list(self.raw_data['Title'])
        self.data=None
        self._values=None
        self.velues=None
        self.preds=None
    
    def _load_model(self, path_to_model):
        if path_to_model==None:
            return
        model = lgb.Booster(model_file=path_to_model)
        return model
    
    def _load_vocabulary(self, path_to_vocabulary):
        if path_to_vocabulary==None:
            return
        return list(pd.read_csv(path_to_vocabulary)['0'])
    
    def _load_demo(self, path_to_data):
        if path_to_data==None:
            return
        demo_df = pd.read_csv(path_to_data)
        return demo_df
    
    def _preprocess(self):
        if self.raw_data is None:
            print('Load .csv data first')
            raise AttributeError
        for column_name in ['Title', 'Abstract']:
            self.raw_data=Stem_data(self.raw_data, column_name)
       
        tv_abst = TfidfVectorizer(
                            analyzer='word', 
                            ngram_range=(1,2), 
                            stop_words='english',
                            vocabulary = self.vocabulary,
                            smooth_idf=True,
                            )
        
        tv_title = TfidfVectorizer(
                            analyzer='word', 
                            ngram_range=(1,2), 
                            stop_words='english', 
                            vocabulary = self.vocabulary, 
                            smooth_idf=True,
                            norm='l2'
                            )
        
        abstract_text_features_sm = tv_abst.fit_transform(self.raw_data['Abstract'])
        abstract_text_features = pd.DataFrame(abstract_text_features_sm.todense())
        title_text_features_sm = tv_title.fit_transform(self.raw_data['Title'])
        title_text_features = pd.DataFrame(title_text_features_sm.todense())        
        text_features_tokens = []
        
        for k,v in tv_abst.vocabulary_.items():
            text_features_tokens.append(k + '_abst')
        
        for k,v in tv_title.vocabulary_.items():
            text_features_tokens.append(k + '_title')
        
        self.data = pd.concat([self.raw_data, abstract_text_features, 
                               title_text_features], axis = 1) 
        self.data.columns =  list(self.raw_data.columns) + text_features_tokens
        self._values = self.data['Citations_per_year']
        self.data = self.data.drop('Citations_per_year', axis=1)
        self.data = self.data[self._important_features]
    
    def predict(self, n_years=1):
        self._preprocess()
        self.values = np.array(n_years*self._values, dtype='float32')
        self.preds=np.array(n_years*self.model.predict(self.data, 
                                                       predict_contrib=True), 
                            dtype='float32')
        return self.preds
    
    def get_results(self):
        if self.values is None:
            print('Run .predict() method first')
            raise AttributeError
        result = pd.concat([self.raw_data['Title'], 
                            pd.DataFrame(self.values), 
                            pd.DataFrame(self.preds)], axis=1)
        result.columns = ['Title', 'Citations', 'Predicted_Citations']
        return result
    
    def explain_values(self, max_display=10):
        self.model.params["objective"] = "regression"
        explainer=shap.Explainer(self.model)
        shap_values=explainer(self.data)
        shap.initjs()
        for i in range(len(self.titles)):
            plt.title(self.titles[i])
            shap.waterfall_plot(shap_values[i], max_display=max_display)
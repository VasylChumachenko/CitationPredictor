# CitationPredictor

# Motivation

Number of citations of scientific paper is a key parameter that shows the significance of any publication. Citations of multiple publications therefore define the performance of academical researchers (quantified by so-called h-index https://en.wikipedia.org/wiki/H-index) and as a result carrier of particular researcher and funding avaliability for scientific teams and institutions. That is why pre-estimation of number of citations based on clear and well-defined input information could be a good tool for the researchers to gain some insights on possible outcome of certain publication without in-deep analysis of publication text itself. Potentially, tools like one in this repo can become a part of scientific management frameworks.

# Brief project description

Citation predictor estimates year-averaged citations number of scientific papers (in the fields of phisics, chemistry and biology) based on solely basic journal metrics (Impact factor, CiteScore etc) and key-word frequencies in article title and abstract. Trainig dataser was generated by scraping American Chemical Society (https://www.acs.org/) scientific database that includes 150+ peer-rewieved journals of different significance. 

Overall of 150k+ datapoints have been collected. Dataset covers works from 1990-2021. Vocabylary were disigned taking into account basic scientific terminology and Nobel Price names (from 1900-2021). Vocabylary allowed to extract 2k+ features among which 320+ have been chosen as most significant.

As a baseline author chose median per journal citation rate (gives 7.5 root mean squared error). Lightgbm gradient boosting showed the best performance among other ML techniques and some basic NN (Linear Regression, Random Forest, SVM and multi layer perceptrones) on 5 folds cross-validation. Lightgbm and XGBoost both beats chosen baceline and gave similar rmse of 6.5 on validation however lightgbm trained faster. 

Current model have a flows. It considerably underestimates high citation rates (>10 citations per year) and cannot properly predict zeros. However, overcoming these problems may certanly require experimanting with DL for NLP with no result garanteed. 

# How to use CitationPredictor:

First, we need to import modules, set path to model, dataset and additional required files and create CitationPredictor class instance:

```
from citation_predictor import CitationPredictor
import matplotlib.pyplot as plt

path_to_model='...'
path_to_data='...'
path_to_vocabulary='...'
path_to_features='...'
predictor = CitationPredictor(path_to_model=path_to_model, 
                       path_to_data=path_to_data,
                       path_to_vocabulary=path_to_vocabulary,
                       path_to_features=path_to_features)
```
Then we have to run predict() method and get prediction results as pandas DataFrame:

```
preds=predictor.predict()
res=predictor.get_results()
```

Finally, we can run explaine_values() method to have some interpretability through visualizing SHAP values:

```
predictor.explain_values()
```
Please note that matplotlib has to be imported in order to properly display SHAP

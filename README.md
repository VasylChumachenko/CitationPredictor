# CitationPredictor

# Motivation

The number of citations of a scientific paper is a key parameter that shows the significance of any publication. Citations of multiple publications, therefore, define the performance of academic researchers (quantified by the so-called h-index https://en.wikipedia.org/wiki/H-index) and as a result carrier of particular researchers and funding availability for scientific teams and institutions. That is why pre-estimation of the number of citations based on clear and well-defined input information could be a good tool for the researchers to gain some insights into the possible outcome of certain publications without in-deep analysis of the publication text itself. Potentially, tools like the one in this repo can become a part of scientific management frameworks.

# Brief project description

Citation predictor estimates year-averaged citations number of scientific papers (in the fields of physics, chemistry, and biology) based on solely basic journal metrics (Impact factor, CiteScore, etc) and keyword frequencies in article title and abstract. The Trainig dataset was generated by scraping the American Chemical Society (https://www.acs.org/) scientific database that includes 150+ peer-reviewed journals of different significance. 

Overall 150k+ datapoints have been collected. The dataset covers works from 1990-2021. Vocabulary was designed taking into account basic scientific terminology and Nobel Price names (from 1900-2021). Vocabulary allowed to extract of 2k+ features among which 320+ have been chosen as most significant.

As a baseline author chose the median per journal citation rate (gives 7.5 root mean squared error). Lightgbm gradient boosting showed the best performance among other ML techniques and some basic NN (Linear Regression, Random Forest, SVM, and multi-layer perceptron) on 5 folds cross-validation. Lightgbm and XGBoost both beats chosen baseline and gave a similar rmse of 6.5 on validation however lightgbm trained faster. 

The current model has problems. It considerably underestimates high citation rates (>10 citations per year) and cannot properly predict zeros. However, overcoming these problems may certainly require experimenting with DL for NLP with no result guaranteed. 

# How to use CitationPredictor:

First, we need to import modules, set the path to the model, dataset, and additional required files, and create CitationPredictor class instance:

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

Finally, we can run explain_values() method to have some interpretability through visualizing SHAP values:

```
predictor.explain_values()
```
Please note that matplotlib has to be imported in order to properly display SHAP

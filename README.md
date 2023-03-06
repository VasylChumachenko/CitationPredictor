# CitationPredictor

The number of citations of a scientific paper is a key parameter that shows the significance of any publication. Citations of multiple publications, therefore, define the performance of academic researchers (quantified by the so-called h-index https://en.wikipedia.org/wiki/H-index) and as a result carrier of particular researchers and funding availability for scientific teams and institutions. That is why pre-estimation of the number of citations based on clear and well-defined input information could be a good tool for the researchers to gain some insights into the possible outcome of certain publications without in-deep analysis of the publication text itself. Potentially, tools like the one in this repo can become a part of scientific management frameworks.

# How to use CitationPredictor:

First, we need to import modules, set the path to the model, dataset, and additional required files, and create CitationPredictor class instance:

```
from citation_predictor import CitationPredictor
import matplotlib.pyplot as plt

path_to_model='.../lgb_reg'
path_to_data='.../demo_data.csv'
path_to_vocabulary='.../vocabulary.csv'
path_to_features='.../features.csv'
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

## Demo

Demo notebook is here:

https://drive.google.com/file/d/1B3s5lBYJGYA5aDrtToVnRe0LsGmfY9UJ/view?usp=share_link

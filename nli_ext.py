from collections import defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import utils
import nli

def wordentail_experiment(
        train_data,
        assess_data,
        vector_func,
        vector_combo_func,
        train_func,
        featurize_func=nli.word_entail_featurize,
        verbose=True,
):
    X_train, y_train = featurize_func(
        train_data,  vector_func, vector_combo_func)
    X_dev, y_dev = featurize_func(
        assess_data, vector_func, vector_combo_func)
    
    model = train_func(X_train, y_train)    
    
    predictions = model.predict(X_dev)
    
    # Report:
    if verbose:
        print(classification_report(y_dev, predictions, digits=3))
    macrof1 = utils.safe_macro_f1(y_dev, predictions)
    return {
        'model': model,
        'train_data': train_data,
        'assess_data': assess_data,
        'macro-F1': macrof1,
        'vector_func': vector_func,
        'vector_combo_func': vector_combo_func}
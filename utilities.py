# Content: Utilities for LLM-enhanced AutoML
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: Sept, 2023

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)
from flaml import tune
from flaml import AutoML
import json


def metrics_display(y_test, y_pred, y_pred_proba):

    # Obtain confusion matrix
    cm = confusion_matrix(y_test, y_pred)
   
    # Output classification metrics
    tn, fp, fn, tp = cm.ravel()
   
    print(f'ROC_AUC score: {roc_auc_score(y_test, y_pred_proba):.3f}')
    print(f'f1 score: {f1_score(y_test, y_pred):.3f}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
    print(f'Precision: {precision_score(y_test, y_pred)*100:.2f}%')
    print(f'Detection rate: {recall_score(y_test, y_pred)*100:.2f}%')
    print(f'False alarm rate: {fp / (tn+fp)*100}%')
    print(f'MCC: {matthews_corrcoef(y_test, y_pred):.2f}')
   
    # Display confusion matrix
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format='.5g', colorbar=False)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()



def data_report(df, num_feats, bin_feats, nom_feats):
   
    # Last column is the label
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # General dataset info
    num_instances = len(df)
    num_features = features.shape[1]

    # Label class analysis
    class_counts = target.value_counts()
    class_distribution = class_counts/num_instances
    if any(class_distribution<0.3) or any(class_distribution>0.7):
        class_imbalance = True
    else:
        class_imbalance = False
   
    # Create a text report
    report = f"""Data Characteristics Report:

- General information:
  - Number of Instances: {num_instances}
  - Number of Features: {num_features}

- Class distribution analysis:
  - Class Distribution: {class_distribution.to_string()}
  {'Warning: Class imbalance detected.' if class_imbalance else ''}

- Feature analysis:
  - Feature names: {features.columns.to_list()}
  - Number of numerical features: {len(num_feats)}
  - Number of binary features: {len(bin_feats)}
  - Binary feature names: {bin_feats}
  - Number of nominal features: {len(nom_feats)}
  - Nominal feature names: {nom_feats}
"""
   
    return report



def auto_machine_learning(X_train, y_train, time_budget,
                          metric, custom_hp, starting_points=None):
    """Hyperparameter tuning API wrapping around FLAML library. 
    """
    
    # AutoML object
    automl = AutoML()
    
    # Tuning config
    config = {
        'task': 'classification',
        'metric': metric,
        'time_budget': time_budget,
        'estimator_list': ['xgboost'],
        'custom_hp': {'xgboost': custom_hp},
        'log_file_name': 'automl.log',
        'eval_method': 'cv',
        'log_type': 'all'
    }
    
    # Starting point
    if starting_points is not None:
        config['starting_points'] = starting_points
    
    # Fitting
    automl.fit(X_train, y_train, **config)
    
    # Tuning logs
    with open("automl.log", "r") as txt_file:
        log = txt_file.readlines()
    record = {
        'log': log,
        'best config': automl.best_config
    }
    
    return automl, record


def suggest_metrics(report):
    
    prompt = f"""
    The classification problem under investigation is based on a network intrusion detection dataset. 
    This dataset contains DOS, Probe, R2L, and U2R attack types, which are all grouped under the 
    "attack" class (label: 1). Conversely, the "normal" class is represented by label 0. 
    Below are the dataset's characteristics:
    {report}.

    For this specific inquiry, you are tasked with recommending a suitable hyperparameter optimization 
    metric for training a XGBoost model, which should deliver high detection rate and low false alarm 
    rate. Given the problem context and dataset characteristics:

    - suggest only the name of one of the built-in metrics: 'accuracy', 'roc_auc' (ROCAUC score), or 'f1' (F1 score).
    OR
    - if the built-in metrics do not fit, propose a customized metric function, which should 
    have the following signature:

    def custom_metric(
        X_val, y_val, estimator, labels,
        X_train, y_train, weight_val=None, weight_train=None,
        config=None, groups_val=None, groups_train=None,
    ):
        return metric_to_minimize, metrics_to_log
        
    Please first briefly explain your reasoning and then provide the recommended metric name or the custom function. 
    Your recommendation should be enclosed between markers [BEGIN] and [END], with either a standalone string for 
    indicating the metric name, or the definition of the custom function.
    Do not provide other settings or configurations.
    """

    return prompt



def suggest_initial_prompt():
    
    prompt = f"""
    Given your understanding of XGBoost and general best practices in machine learning, suggest an 
    initial search space for hyperparameters. Remember to consider interactions and dependencies 
    between hyperparameters.

    Tunable hyperparameters include:
    - n_estimators (integer): Number of boosting rounds or trees to be trained.
    - max_leaves (integer): Maximum number of terminal nodes (or leaves) in a single boosted tree. 
    - min_child_weight (integer or float): Minimum sum of instance weight (hessian) needed in a leaf node. 
    - learning_rate (float): Step size shrinkage used during each boosting round to prevent overfitting. 
    - subsample (float): Fraction of the training data sampled to train each tree. 
    - colsample_bylevel (float): Fraction of features that can be randomly sampled for building each level (or depth) of the tree.
    - colsample_bytree (float): Fraction of features that can be randomly sampled for building each tree. 
    - reg_alpha (float): L1 regularization term on weights. 
    - reg_lambda (float): L2 regularization term on weights. 

    The search space is defined as a nested dict with keys being hyperparameter names, and values 
    are dicts of info ("domain" and "init_value") about the search space associated with the 
    hyperparameter. For example:
        custom_hp = {{
            "learning_rate": {{
                "domain": tune.loguniform(lower=0.01, upper=20.0),
                "init_value": 1
            }}
        }}
        
    Remember, for tune.loguniform(), tune.qloguniform(), and tune.lograndint(), the bounds you 
    provide should be in their original scale, not log-transformed. For instance, if you intend 
    to explore values between 0.001 (1e-3) and 1 in a log scale, you'd suggest:
    tune.loguniform(lower=1e-3, upper=1).

    Examples of the available types of domains include:
    # Sample a float uniformly between -5.0 and -1.0
    tune.uniform(-5, -1),

    # Sample a float uniformly between a small value greater than 0 (e.g., 1e-4) and 1, 
    # while sampling in log space.
    tune.loguniform(1e-4, 1),

    # Sample a float uniformly between 0.0001 and 0.1, while
    # sampling in log space and rounding to increments of 0.00005
    tune.qloguniform(1e-4, 1e-1, 5e-5),

    # Sample a random float from a normal distribution with
    # mean=10 and sd=2
    tune.randn(10, 2),

    # Sample a integer uniformly between -9 (inclusive) and 15 (exclusive)
    tune.randint(-9, 15),

    # Sample a integer uniformly between 1 (inclusive) and 10 (exclusive),
    # while sampling in log space
    tune.lograndint(1, 10),

    # Sample an option uniformly from the specified choices
    tune.choice(["a", "b", "c"])  

    Please first briefly explain your reasoning, then provide the configurations of the initial 
    search space. Enclose your suggested configurations between markers 
    [BEGIN] and [END], and assign your configuration to a variable named custom_hp.
    """

    return prompt



def suggest_refine_search_space(logs, results):
    
    prompt = f"""
    Please refine the search space based on the search history of previous AutoML run:
    {simplify_logs(logs)}


    The current best configuration for XGBoost from this round:
    {results.best_config}

    Remember, tunable hyperparameters are: n_estimators, max_leaves, min_child_weight, learning_rate, 
    subsample, colsample_bylevel, colsample_bytree, reg_alpha, reg_lambda.

    Given the insights from the search history and the current best configuration, please suggest 
    refinements for the search space in the next optimization round to enhance the model's performance. 
    You don't need to provide initial values for the hyperparameters; the AutoML tool will leverage 
    results from the previous run.

    Please first briefly explain your reasoning and then provide the refined configurations. 
    Enclose your refined configuration suggestions between markers [BEGIN] and [END]. 
    Please assign your configuration recommendations to a variable named custom_hp.
    """
    return prompt





def simplify_logs(log_data):
    simplified_logs = []

    # Go through each log entry and extract essential information
    for log in log_data['log']:
        if 'curr_best_record_id' not in log:
            log_json = json.loads(log)
            simplified_entry = {
                'record_id': log_json['record_id'],
                'validation_loss': log_json['validation_loss'],
                'config': log_json['config']
            }
            simplified_logs.append(simplified_entry)

    return simplified_logs



def extract_insights(logs, N=5):
    
    records = [json.loads(log) for log in logs[:-1]]

    # Sort records by validation_loss
    sorted_records = sorted(records, key=lambda x: x['validation_loss'])

    # Get top-N configurations
    top_n_configs = [record['config'] for record in sorted_records[:N]]
    top_n_loss = [record['validation_loss'] for record in sorted_records[:N]]
    top_n = top_n_configs
    for i in range(len(top_n)):
        top_n[i]['validation_loss'] = top_n_loss[i]
    formatted = '\n'.join([f"Config {i + 1}: {' '.join([f'{k}={v}' for k, v in entry.items()])}" for i, entry in enumerate(top_n)])

    # Average validation loss among top-N
    avg_validation_loss = sum(record['validation_loss'] for record in sorted_records[:N]) / N

    # Identify hyperparameters with most variance in top-N
    variance = {}
    for key in top_n_configs[0].keys():
        values = [config[key] for config in top_n_configs]
        variance[key] = max(values) - min(values)

    # Sort hyperparameters by variance
    sorted_variance = sorted(variance.items(), key=lambda x: x[1], reverse=True)
    most_varied_hyperparam = sorted_variance[0][0]

    insights = {
        'average_validation_loss_top_N': avg_validation_loss,
        'most_varied_hyperparam': most_varied_hyperparam
    }
    

    return insights, formatted


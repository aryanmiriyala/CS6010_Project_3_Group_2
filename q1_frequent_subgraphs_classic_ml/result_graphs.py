import matplotlib.pyplot as plt
import pandas as pd
import os.path as path
from os import sep, getcwd, mkdir

results_path = getcwd() + sep + f'q1_frequent_subgraphs_classic_ml' + sep +  f'results'
graph_path = results_path + sep + 'graphs'


num_seeds = ['0', '1', '2']
supports = ['0.10', '0.20', '0.30', '0.40']

val_metrics = ['val_accuracy','val_precision','val_recall','val_f1','val_auc', 'val_inference_time_sec']
test_metrics = ['test_accuracy','test_precision','test_recall','test_f1','test_auc', 'test_inference_time_sec']

model_names = ['RandomForest','LinearSVM', 'RBFSVM'] 

# Read in results of a model from each seed
def read_file(seed_num, support_num):
    file_path = results_path + sep + f'seed_{seed_num}{sep}classic_ml_support_{support_num}.csv'
    if(path.exists(file_path)):
        model_results = pd.read_csv(file_path)
        return model_results
    else:
        print("Results not found for: ", file_path)
        return None


# read in all model results as one df
results_all = []
for seed in num_seeds:
    for support in supports:
        result = read_file(seed, support)
        result['support'] = support
        results_all.append(result)

results_df = pd.concat(results_all)
results_df.drop(columns = [ 'num_params', 'feature_dim'], inplace=True)

params = list(results_df['params'].unique())
param_dict = {model_names[0]: params[0:4], model_names[1]:params[4:6], model_names[2]: params[6:8]}


def rename_param(df, param_dict):
    new_param = {}
    for model, params in param_dict.items():
        param_num = 0   
        for par in params:
            key = f'{model} {param_num + 1}'
            param_num += 1
            new_param[key] = par
            df['params'][df['params'] == par] = key

    return df, new_param

results_df, params_dict = rename_param(results_df, param_dict=param_dict)


## Create a Visual Representation of Ablations studies for that model
def build_graphs(results_df, metric_name, supports = [], metrics = []):
    results_df = results_df.groupby(by = 'support')

    if(not path.exists(graph_path)):
        mkdir(graph_path)

    for support in supports:
        results_format = results_df.get_group(support).groupby('params')[metrics].agg('mean')

        ax = results_format.plot(kind='bar', width = 0.8, figsize=(14,14)) 
        metric_set = metric_name
        ft_size = 15
        plt.title(f"Average {metric_set} Metrics across SEEDs for Support {support} - Classic ML Abalation Studies", fontsize = ft_size*1.15)
        plt.xlabel("Model Parameter Set", fontsize = ft_size*1.15)
        plt.ylabel("Mean Metric Value", fontsize = ft_size*1.15)
        plt.xticks(rotation = 30, fontsize = ft_size)
        plt.yticks(fontsize = ft_size)
        filename = graph_path + sep + f'classic_ml_{support}_{metric_set}.png'
        plt.savefig(filename)

build_graphs(results_df, 'Validation', supports=supports, metrics=val_metrics)
build_graphs(results_df, 'Test', supports=supports, metrics=test_metrics)
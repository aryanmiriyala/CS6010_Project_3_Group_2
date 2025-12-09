import matplotlib.pyplot as plt
import pandas as pd
import os.path as path
from os import sep, getcwd, mkdir

results_path = getcwd() + sep + f'q2_gnn' + sep +  f'results'
graph_path = results_path + sep + 'graphs'
csv_path = results_path + sep + "csv"

num_seeds = ['0', '1', '2']

val_metrics = ['val_accuracy','val_precision','val_recall','val_f1','val_auc', 'val_inference_time_sec']
test_metrics = ['test_accuracy','test_precision','test_recall','test_f1','test_auc', 'test_inference_time_sec']

model_names = ['GIN', 'GCN'] 
supports = ['hidden_dim','layers','dropout','lr']
# Read in results of a model from each seed
def read_file(seed_num, model_name):
    file_path = results_path + sep + f'seed_{seed_num}{sep}' + sep + f'{model_name}_results.csv'
    if(path.exists(file_path)):
        model_results = pd.read_csv(file_path)
        return model_results
    else:
        print("Results not found for: ", file_path)
        return None


# read in all model results as one df
results_all = []
for seed in num_seeds:
        for model in model_names:
            result = read_file(seed, model)
            results_all.append(result)


results_df = pd.concat(results_all)
results_df['params'] = results_df['hidden_dim'].astype('str') + ',' + results_df['layers'].astype('str') + ',' + results_df['dropout'].astype('str') + ',' + results_df['lr'].astype('str')
## Create a Visual Representation of Ablations studies for that model
def build_graphs(results_df, metric_name, columns = [] , metrics = []):

    if(not path.exists(graph_path)):
        mkdir(graph_path)

    if(not path.exists(csv_path)):
        mkdir(csv_path)

    for col in columns:
        format_results = results_df.groupby(by = ['model', col])
        format_results = format_results[metrics].agg('mean')
        format_results.to_csv(csv_path + sep + f'GNN_{col}_{metric_name}.csv')

        ax = format_results.plot(kind='bar', width = 0.8, figsize=(14,14)) 
        metric_set = metric_name
        ft_size = 15
        plt.title(f"Average {metric_set} Metrics across SEEDs for {col} - GNN Abalation Studies", fontsize = ft_size*1.15)
        plt.xlabel("Model Parameter Set", fontsize = ft_size*1.15)
        plt.ylabel("Mean Metric Value", fontsize = ft_size*1.15)
        plt.xticks(rotation = 30, fontsize = ft_size)
        plt.yticks(fontsize = ft_size)
        filename = graph_path + sep + f'GNN_{col}_{metric_set}.png'
        plt.savefig(filename)

build_graphs(results_df, 'Validation', columns=supports, metrics=val_metrics)
build_graphs(results_df, 'Test', columns=supports, metrics=test_metrics)


## Get Summary Results
quality_metrics = test_metrics[0:-1] + val_metrics[0:-1]
efficiency_metrics = [test_metrics[-1], val_metrics[-1]]
all_metrics = quality_metrics + efficiency_metrics

format_results = results_df.groupby(by = ['model'])

# Get GCN results
GCN_results = format_results.get_group('GCN').groupby('params')[all_metrics].agg('mean')
GCN_results_test = GCN_results.loc[GCN_results[test_metrics].idxmax()]

GCN_results_test['model'] = "GCN"
model = GCN_results_test.pop('model')
GCN_results_test.insert(0, 'model', model)

# Get GIN results
GIN_results = format_results.get_group('GIN').groupby('params')[all_metrics].agg('mean')
GIN_results_test = GIN_results.loc[GIN_results[test_metrics].idxmax()]

GIN_results_test['model'] = "GIN"
model = GIN_results_test.pop('model')
GIN_results_test.insert(0, 'model', model)

results_save = pd.concat([GCN_results_test, GIN_results_test])
results_save.drop_duplicates(inplace=True)
results_save.to_csv(graph_path + sep + 'GNN_best_results.csv')

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from neural_network.config import pipeline_configs
from neural_network.torch_pipeline import Pipeline
from neural_network.config import PipeLineConfig

from get_logger import get_logger

logger = get_logger(__name__)

def hyperparameter_optimization(pipeline_config: PipeLineConfig, seed=None):
    if seed is not None:
        pipeline_config.train_data.seed = seed

    neuron_num_list = [16, 24, 32, 48, 64, 96, 128]
    layer_num_list = [2, 4, 8, 16]
    activ_func_list = ["tanh", "leaky_relu"]
    learning_rate_list = [0.001, 0.005, 0.01, 0.05]
    lambda_param_list = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]

    test_cases = set()
    while len(test_cases) < 20:
        neuron_num = np.random.choice(neuron_num_list)
        layer_num = np.random.choice(layer_num_list)
        activ_func = np.random.choice(activ_func_list)
        learning_rate = np.random.choice(learning_rate_list)
        lambda_param = np.random.choice(lambda_param_list)
        test_cases.add((neuron_num, layer_num, activ_func, learning_rate, lambda_param))

    results_list = []
    pipeline_instance = Pipeline(pipeline_config)

    for neuron_num, layer_num, activ_func, learning_rate, lambda_param in test_cases:
        pipeline_instance.config.model.neuron_per_layer = neuron_num
        pipeline_instance.config.model.layer_number = layer_num
        pipeline_instance.config.model.hidden_layer_activation = activ_func
        pipeline_instance.config.model.learning_rate = learning_rate
        pipeline_instance.config.model.lambda_param = lambda_param
        pipeline_instance.config.model.epochs = 5

        pipeline_instance.train(retrain=True)
        eval_res = pipeline_instance.evaluate(None)
        eval_res = {k: [v] for k, v in eval_res.items()}
        logger.info(f"Evaluation results: {eval_res}")
        results_list.append(pd.DataFrame(eval_res))

    first_round_results_df = pd.concat(results_list, axis=0).reset_index(drop=True)
    first_round_results_df["overall_loss"] = first_round_results_df[
        [col for col in first_round_results_df if col.startswith("out_sample_")]
    ].sum(axis=1).values
    first_round_results_df = first_round_results_df.sort_values(by=["overall_loss"], ascending=True)

    test_cases = set()
    for i in range(10):
        neuron_num = first_round_results_df.iloc[i]["neuron_per_layer"]
        layer_num = first_round_results_df.iloc[i]["layer_number"]
        activ_func = first_round_results_df.iloc[i]["hidden_layer_activation"]
        learning_rate = first_round_results_df.iloc[i]["learning_rate"]
        lambda_param = first_round_results_df.iloc[i]["lambda_param"]
        test_cases.add((neuron_num, layer_num, activ_func, learning_rate, lambda_param))

    results_list = []
    for neuron_num, layer_num, activ_func, learning_rate, lambda_param in test_cases:
        pipeline_instance.config.model.neuron_per_layer = neuron_num
        pipeline_instance.config.model.layer_number = layer_num
        pipeline_instance.config.model.hidden_layer_activation = activ_func
        pipeline_instance.config.model.learning_rate = learning_rate
        pipeline_instance.config.model.lambda_param = lambda_param
        pipeline_instance.config.model.epochs = 10

        pipeline_instance.train(retrain=True)
        eval_res = pipeline_instance.evaluate(None)
        eval_res = {k: [v] for k, v in eval_res.items()}
        logger.info(f"Evaluation results: {eval_res}")
        results_list.append(pd.DataFrame(eval_res))

    second_round_results_df = pd.concat(results_list, axis=0).reset_index(drop=True)
    second_round_results_df["overall_loss"] = second_round_results_df[
        [col for col in second_round_results_df if col.startswith("out_sample_")]
    ].sum(axis=1).values
    second_round_results_df = second_round_results_df.sort_values(by=["overall_loss"], ascending=True)

    test_cases = set()
    for i in range(5):
        neuron_num = second_round_results_df.iloc[i]["neuron_per_layer"]
        layer_num = second_round_results_df.iloc[i]["layer_number"]
        activ_func = second_round_results_df.iloc[i]["hidden_layer_activation"]
        learning_rate = second_round_results_df.iloc[i]["learning_rate"]
        lambda_param = second_round_results_df.iloc[i]["lambda_param"]
        test_cases.add((neuron_num, layer_num, activ_func, learning_rate, lambda_param))

    results_list = []
    for neuron_num, layer_num, activ_func, learning_rate, lambda_param in test_cases:
        pipeline_instance.config.model.neuron_per_layer = neuron_num
        pipeline_instance.config.model.layer_number = layer_num
        pipeline_instance.config.model.hidden_layer_activation = activ_func
        pipeline_instance.config.model.learning_rate = learning_rate
        pipeline_instance.config.model.lambda_param = lambda_param
        pipeline_instance.config.model.epochs = 20

        pipeline_instance.train(retrain=True)
        eval_res = pipeline_instance.evaluate(None)
        eval_res = {k: [v] for k, v in eval_res.items()}
        logger.info(f"Evaluation results: {eval_res}")
        results_list.append(pd.DataFrame(eval_res))

    third_round_results_df = pd.concat(results_list, axis=0).reset_index(drop=True)
    third_round_results_df["overall_loss"] = third_round_results_df[
        [col for col in third_round_results_df if col.startswith("out_sample_")]
    ].sum(axis=1).values
    third_round_results_df = third_round_results_df.sort_values(by=["overall_loss"], ascending=True)

    return first_round_results_df, second_round_results_df, third_round_results_df

if __name__ == "__main__":
    seed = 20250418
    for config_name, config in pipeline_configs.items():
        first_round_results_df, second_round_results_df, third_round_results_df = hyperparameter_optimization(config, seed=seed)
        os.makedirs(os.path.join("hyperparam_optim", str(seed)), exist_ok=True)

        first_round_results_df.to_csv(os.path.join("hyperparam_optim", str(seed), f"{config_name}_first_round.csv"), index=False)
        second_round_results_df.to_csv(os.path.join("hyperparam_optim", str(seed), f"{config_name}_second_round.csv"), index=False)
        third_round_results_df.to_csv(os.path.join("hyperparam_optim", str(seed), f"{config_name}_third_round.csv"), index=False)

        logger.info(f"Hyperparameter optimization completed for {config_name}")

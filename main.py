import pickle
import numpy as np
import torch
import itertools
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, recall_score, precision_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from data import get_data, load_and_preprocess_data, SequenceDataset
from params import optimizer_dict, loss_function_dict, hyperparameter_space, num_rounds
from helper_func import get_predictions, train_model, test_model, matrix_eval
from models import QShallowRegressionLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = get_data();

# Now, generate all combinations of hyperparameters
param_names = list(hyperparameter_space.keys())
param_values = list(hyperparameter_space.values())

# Use itertools.product to create combinations
all_combinations = list(itertools.product(*param_values))

# Initialize a list to store all results
all_results = [];

# for idx, hyperparams in enumerate(random_hyperparams_list):
for idx, combo in enumerate(all_combinations):
    hyperparams = dict(zip(param_names, combo))
    print(f"\nTrial {idx+1}/{len(all_combinations)} with hyperparameters: {hyperparams}")

    # Extract hyperparameters
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    sequence_length = hyperparams['sequence_length']
    num_hidden_units = hyperparams['num_hidden_units']
    num_qubits = hyperparams['num_qubits']
    num_qlayers = hyperparams['num_qlayers']
    optimizer_name = hyperparams['optimizer']
    loss_function_name = hyperparams['loss_function']
    backend = hyperparams['backend']
    num_epochs = hyperparams['num_epochs']  # Local epochs
    num_nodes =  hyperparams['num_nodes']

    # Load and preprocess data
    df_train_balanced, df_test, features, target = load_and_preprocess_data(data.copy(),
                                                                            hyperparams['sample_size'],
                                                                            hyperparams['isSelectedColumns'])

    # Split the training data into 10 chunks for federated learning
    node_data_size = len(df_train_balanced) // num_nodes
    node_datasets = []
    for i in range(num_nodes):
        start_idx = i * node_data_size
        if i != num_nodes - 1:
            end_idx = (i + 1) * node_data_size
        else:
            end_idx = len(df_train_balanced)
        node_df = df_train_balanced.iloc[start_idx:end_idx].reset_index(drop=True)
        node_dataset = SequenceDataset(
            node_df,
            target=target,
            features=features,
            sequence_length=sequence_length
        )
        node_datasets.append(node_dataset)

    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize global model
    global_model = QShallowRegressionLSTM(
        num_sensors=len(features),
        hidden_units=num_hidden_units,
        n_qubits=num_qubits,
        n_qlayers=num_qlayers,
        backend=backend
    )

    # Initialize loss function
    loss_function_class = loss_function_dict[loss_function_name]
    loss_function = loss_function_class()

    for round_num in range(num_rounds):
        print(f"\n--- Global Round {round_num+1}/{num_rounds} ---")

        # Store local models' state dictionaries
        local_state_dicts = []
        for node_idx in range(num_nodes):
            print(f"Training on node {node_idx+1}/{num_nodes}")
            # Copy global model to local model
            local_model = QShallowRegressionLSTM(
                num_sensors=len(features),
                hidden_units=num_hidden_units,
                n_qubits=num_qubits,
                n_qlayers=num_qlayers,
                backend=backend
            )
            local_model.load_state_dict(global_model.state_dict())

            # Initialize optimizer for local training
            optimizer_class = optimizer_dict[optimizer_name]
            optimizer = optimizer_class(local_model.parameters(), lr=learning_rate)

            # Create data loader for local data
            local_loader = DataLoader(node_datasets[node_idx], batch_size=batch_size, shuffle=True)

            # Train local model
            for epoch in range(num_epochs):
                train_loss = train_model(local_loader, local_model, loss_function, optimizer)
                print(f"Node {node_idx+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

            # Collect local model state dict
            local_state_dicts.append(local_model.state_dict())

        # Federated Averaging: Update global model
        global_state_dict = global_model.state_dict()
        # Initialize the global parameters to zeros
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])

        # Sum up the local models
        for state_dict in local_state_dicts:
            for key in global_state_dict.keys():
                global_state_dict[key] += state_dict[key]

        # Average the parameters
        for key in global_state_dict.keys():
            global_state_dict[key] = global_state_dict[key] / num_nodes

        # Load averaged parameters into global model
        global_model.load_state_dict(global_state_dict)

        # Evaluate global model on test data
        test_loss = test_model(test_loader, global_model, loss_function)
        output, target = get_predictions(test_loader, global_model)
        matrix_eval(output, target);
        print(f"Global Test Loss after Round {round_num+1}: {test_loss:.6f}")

    # Evaluation
    y_pred_logits, y_test = get_predictions(test_loader, global_model)
    y_pred_probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()

    # Compute optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_binary = [1 if prob >= optimal_threshold else 0 for prob in y_pred_probs]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_probs)

    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}")

    # Store results
    result = {
        'trial': idx+1,
        'hyperparameters': hyperparams,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'optimal_threshold': optimal_threshold,
        "model_parameters": [(k, v.shape) for k,v in global_model.state_dict().items()],
    }
    all_results.append(result)

    with open(f"exp/results", "wb") as fp:
      pickle.dump(all_results, fp)
      print("res saved")



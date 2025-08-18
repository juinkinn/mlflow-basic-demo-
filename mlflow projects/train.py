import os
import sys

import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Load MNIST
    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype("int")

    # Train/test split
    X_train, y_train = X[0:60000], y[0:60000]
    X_test, y_test = X[-10000:], y[-10000:]
    
    # Params

    solver = str(sys.argv[1]) if len(sys.argv) > 1 else "lbfgs"
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    model_params = {
        "solver": solver,
        "max_iter": max_iter,
        "random_state": 42,
    }


    # MLflow tracking
    with mlflow.start_run():
        model = LogisticRegression(**model_params)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)

        print(f"Test accuracy: {acc:.4f}")

        input_schema = Schema([
            TensorSpec(np.dtype('float64'), (-1, 784))
        ])
        output_schema = Schema([
            TensorSpec(np.dtype('int64'), (-1,))
        ])
        
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log model
        model_info = mlflow.sklearn.log_model(
            model,
            name="logistic_mnist",
            input_example=X_test[0].reshape(1,-1),
            registered_model_name="sklearn-lr-mnist"
        )
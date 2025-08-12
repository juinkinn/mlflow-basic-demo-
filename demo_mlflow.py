import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

# 1. Load dataset
data = fetch_california_housing(as_frame=False)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Models
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(random_state=42),
        "params": {"n_estimators": 100, "max_depth": 10}
    },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {"n_estimators": 100, "learning_rate": 0.1}
    }
}

# 3. MLflow experiment
mlflow.set_experiment("california_housing_regression")
client = MlflowClient()

for model_name, config in models.items():
    with mlflow.start_run(run_name=f"{model_name}-California") as run:
        model = config["model"]
        params = config["params"]

        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)

        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_test[0:1],
        )
        model_uri = f"runs:/{run.info.run_id}/model"

        # Evaluate
        eval_results = mlflow.evaluate(
            model_uri,
            data=X_test,
            targets=y_test,
            model_type="regressor"
        )

        # Register model
        registered_model_name = f"{model_name}_California"
        mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

import mlflow
import sys
import json

if __name__ == "__main__":
    model_uri = str(sys.argv[1]) if len(sys.argv) > 1 else None
    test_json = str(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        with open(test_json, 'r') as f:
            data = json.load(f)['inputs']
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(data)
        print(f"Prediction: {y_pred}")
    except Exception as e:
        raise e
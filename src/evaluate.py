import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def results_to_dataframe(results_dict):
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

from collections import Counter

import examination as ex

source = ex.in_data("test.csv")
print(len(source["CID"]))

xgb = ex.in_runtime("ModXGBoost.csv")
print(Counter(xgb["Predicted_Results"]))

lg = ex.in_runtime("ModLightGBM.csv")
print(Counter(lg["Predicted_Results"]))

mlp = ex.in_runtime("ModMLP.csv")
print(Counter(mlp["Predicted_Results"]))
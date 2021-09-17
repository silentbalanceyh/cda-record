from constant import *

dataDf = exam.Answer()\
    .put("Logistic", exam.ModLogistic)\
    .put("DTC", exam.ModDtc)\
    .put("LightGBM", exam.ModLightGBM)\
    .put("Catboost", exam.ModCatboost)\
    .put("XGBoost", exam.ModXGBoost)\
    .put("Mlp", exam.ModMLP)\
    .put("Svc", exam.ModSvc)\
    .put("RForest", exam.ModRForest)\
    .put("RForestXGB", exam.ModRForestXGB)\
    .output(run_cat)
print("\033[30m")
print(dataDf)
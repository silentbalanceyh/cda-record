from constant import *

dataDf = exam.Answer()\
    .put("Logistic", exam.ModLogistic)\
    .put("DTC", exam.ModDtc)\
    .put("Catboost", exam.ModCatboost)\
    .put("LightGBM", exam.ModLightGBM)\
    .put("XGBoost", exam.ModXGBoost)\
    .put("Mlp", exam.ModMLP)\
    .put("Svc", exam.ModSvc)\
    .put("RForest", exam.ModRForest)\
    .put("RForestXGB", exam.ModRForestXGB)\
    .build(run_txt)
print("\033[30m")
print(dataDf)
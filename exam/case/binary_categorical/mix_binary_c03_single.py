
from constant import *
dataDf = exam.Answer()\
    .put("Catboost", exam.ModCatboost)\
    .output(run_cat)
print("\033[30m")
print(dataDf)
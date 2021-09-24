
from constant import *
dataDf = exam.Answer()\
    .put("Catboost", exam.ModCatboost)\
    .build(run_cat)
print("\033[30m")
print(dataDf)
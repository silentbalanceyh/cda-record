# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase

i_train = exam.csv_train()
runner = exam.Actor(V_ID, V_TARGET).fn_pre(lambda df_train: exam.cat_feature_fn(
    df_train=df_train,
    f_categorical=V_FEATURES
))
runner.execute(i_train, RunPhase.Pre)


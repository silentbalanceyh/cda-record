# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase

i_data = exam.in_data("Train_all.csv")
runner = exam.Actor(V_ID, V_TARGET).fn_split(lambda df_data, p_case: exam.data_split_fn(
    df_data=df_data,
    p_case=p_case,
    p_radio=0.2
)).fn_pre(lambda df_train: exam.cat_feature_fn(
    df_train=df_train,
    f_categorical=V_FEATURES
))
runner.execute(i_data, RunPhase.Mix_SP)

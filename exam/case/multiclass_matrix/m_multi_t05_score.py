# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase

i_true = exam.csv_target()
i_pred = exam.in_runtime("actor_o_xgboost.csv")
runner = exam.Actor(V_ID, V_TARGET).fn_score(lambda df_true, df_pred: exam.data_score_fn(
    df_true=df_true,
    df_predict=df_pred,
    o_target=O_TARGET
))
runner.execute(i_true, RunPhase.Score, i_pred)

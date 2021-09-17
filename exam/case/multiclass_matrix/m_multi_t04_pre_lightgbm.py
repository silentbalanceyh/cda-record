# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase

i_test = exam.csv_test()
runner = exam.Actor(V_ID, V_TARGET).fn_predict(lambda df_test: exam.txt_predict_fn(
    df_test=df_test,
    f_model="ModMXGBoost.model",
    o_id=O_ID,
    o_target=O_TARGET,
    o_filename="actor_o_xgboost.csv"
))
runner.execute(i_test, RunPhase.Predict)

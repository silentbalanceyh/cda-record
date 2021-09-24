# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase

input_df = exam.in_data("training_ori.csv")
runner = exam.Actor(V_ID, V_TARGET).fn_split(lambda data_df, p_case: exam.data_split_fn(
    df_data=data_df,
    p_case=p_case,
    p_radio=0.2
))
runner.execute(input_df, RunPhase.Split)
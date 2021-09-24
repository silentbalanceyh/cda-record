# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase
i_feature = exam.csv_feature()
runner = exam.Actor(V_ID, V_TARGET, f_classes=len(V_TARGET))\
    .fn_run_before(exam.data_modeling_fn)\
    .fn_run(exam.ModVLightGBM)
runner.execute(i_feature, RunPhase.Model)
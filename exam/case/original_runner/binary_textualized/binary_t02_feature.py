# et: Exam Tool（考试专用工具包）
from constant import *
from examination import RunPhase

runner = exam.Actor(V_ID, V_TARGET)
i_train = exam.csv_train()
i_test = exam.csv_test()
runner.fn_pre(lambda df_train: exam.txt_feature_fn(
    df_train=df_train,
    df_test=i_test,        # 此处需要测试集同时执行分词操作
    f_title=V_TITLE,
    f_content=V_CONTENT
))
runner.execute(i_train, RunPhase.Pre)
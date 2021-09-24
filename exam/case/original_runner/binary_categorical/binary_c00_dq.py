# et: Exam Tool（考试专用工具包）
from constant import *

i_data = exam.in_data("Train_all.csv")
report = exam.DQReport(i_data,V_ID, V_TARGET)
report.NReport()
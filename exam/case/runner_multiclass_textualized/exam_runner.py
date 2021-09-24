# -----------------------------------------------------------------------------------------------------
#
# 「执行区」
#
# -----------------------------------------------------------------------------------------------------
"""
步骤：
1. 执行数据拆分，原始训练数据划分成训练集和验证集
2. 数据预处理
    2.1 - 空值错值填补
    2.2 - 类型的独热编码，数值的Min-Max
3.
"""
from exam_kit import *

# -----------------------------------------------------------------------------------------------------
#
# 「分步执行」
#
# -----------------------------------------------------------------------------------------------------
# 步骤0 - Revert
# run_revert()
# 步骤1 - Splitting
# run_splitting()

# 步骤2 - Feature
# run_feature()

# 步骤3 - Modeling
run_model()

# 步骤4 - Predict
# run_predict()

# 步骤5 - Score
# run_score()
# -----------------------------------------------------------------------------------------------------
#
# 「联合执行」
#
# -----------------------------------------------------------------------------------------------------
# 常用综合步骤：训练 + 预测 + 评分
# mix_modeling(MODELER, OUT_RESULT)

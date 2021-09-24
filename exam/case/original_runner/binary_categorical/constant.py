import examination as exam
# --------------- 配置字段
# 特定案子则修改 V_CASE，默认 actor
V_CASE = "actor"

# --------------- 目标字段
V_TARGET = "Target"
V_ID = "B0001"

# --------------- 特征字段
V_FEATURES = [
    'B0002',  # 性别
    # 'B0004',
    # 'B0005',
    # 'B0007',
    'B0009',  # 职业
    # 'B0011',
    'B0013',  # 县市别
    # 'B0014',
    # 'B0017',
    # 'B0018',
    'B0019'  # 主建筑用途
    # 'B0049'
]

# --------------- 输出
O_TARGET = "Predicted_Results"
O_ID = "id"

def run_cat(modeler, o_filename=None):
    return exam.report_cat(
        modeler,
        f_id=V_ID,
        f_target=V_TARGET,
        f_features=V_FEATURES,
        o_filename=o_filename,
        o_id=O_ID,
        o_target=O_TARGET
    )

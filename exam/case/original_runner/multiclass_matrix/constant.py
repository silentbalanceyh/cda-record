import examination as exam

# --------------- 目标字段
V_ID = "ID"
V_TARGET = ["none", "disgust", "like", "happiness", "sadness", "surprise", "anger", "fear"]

# --------------- 特征字段
# V_TITLE = "title"
V_CONTENT = "message"

# --------------- 输出字段
O_TARGET = ["none", "disgust", "like", "happiness", "sadness", "surprise", "anger", "fear"]
O_ID = "ID"

def run_txt(modeler, o_filename=None):
    return exam.report_txt(
        modeler,
        f_id=V_ID,
        f_target=V_TARGET,
        o_filename=o_filename,
        o_id=O_ID,
        o_target=O_TARGET
    )
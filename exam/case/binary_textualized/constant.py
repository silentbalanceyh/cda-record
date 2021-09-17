import examination as exam

# --------------- 目标字段
V_TARGET = "label"
V_ID = "id"

# --------------- 特征字段
V_TITLE = "title"
V_CONTENT = "text"

# --------------- 输出字段
O_TARGET = "target"
O_ID = "id"

def run_txt(modeler, o_filename=None):
    return exam.report_txt(
        modeler,
        f_id=V_ID,
        f_target=V_TARGET,
        o_filename=o_filename,
        o_id=O_ID,
        o_target=O_TARGET
    )
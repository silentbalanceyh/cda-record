# et: Exam Tool（考试专用工具包）
from sklearn.datasets import make_multilabel_classification

from constant import *
from examination import RunPhase

def modeling_defined(df_test):
    def executor(f_id, f_target):
        train_x, train_y = make_multilabel_classification(
            n_samples=500, n_labels=3, allow_unlabeled=False, random_state=0)
        test_x, test_y = make_multilabel_classification(
            n_samples=50, n_labels=3, allow_unlabeled=False, random_state=0)
        return train_x, test_x, train_y, test_y, []
    return executor

runner = exam.Actor(V_ID, V_TARGET, f_classes=len(V_TARGET_BINARY))\
    .fn_run_before(modeling_defined)\
    .fn_run(exam.ModLLightGBM)
runner.execute(None, RunPhase.Model)
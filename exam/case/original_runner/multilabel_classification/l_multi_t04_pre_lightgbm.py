# et: Exam Tool（考试专用工具包）
from sklearn.datasets import make_multilabel_classification

from constant import *
from examination import RunPhase

test_x, test_y = make_multilabel_classification(
            n_samples=50, n_labels=3, allow_unlabeled=False, random_state=0)

model = exam.in_model("ModLLightGBM.model")
predict_y = model.predict(test_x)
print(predict_y)
import examination as exam
TARGET = ["none", "disgust", "like", "happiness", "sadness", "surprise", "anger", "fear"]
y_true = exam.in_data("val_result.xlsx")
y_prod = exam.in_data("results.csv")
exam.data_score(y_true, y_prod, TARGET, TARGET)
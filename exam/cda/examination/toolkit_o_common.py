from sklearn.metrics import f1_score, precision_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from examination.toolkit_o_y import *
from examination.estimator_onezero import PreOneZeroEncoder

def __file_out(flag, p_case): return PATH_RUNTIME + (
    "%s.csv" % flag if p_case is None else "%s_%s.csv" % (p_case, flag))

# ------------------------- 数据集处理

def data_split(df_data, f_id, f_target, p_case=None, p_radio=0.2, f_target_binary=None):
    """
    :param df_data: 全数据集
    :param f_id:    ID字段名
    :param f_target:目标字段名
    :param f_target_binary: 原始
    :param p_case:   「Optional」文件前缀
    :param p_radio:  「Optional」拆分数据集的比例
    :return:  train, test, target
              train：训练集
              test：测试集
              target: 测试集的Y值
    """
    log_info("导入数据Shape：", df_data.shape)
    train_df, test_df = train_test_split(df_data, test_size=p_radio)
    indexes = y_columns(f_id, f_target)
    target_df = test_df[indexes]  # 测试集Y值

    if f_target_binary is not None:
        target_df = target_df.copy()
        if isinstance(f_target, str):
            encoder = LabelEncoder()
            target_df[f_target] = encoder.fit_transform(target_df[f_target])
            out_model("MultiEncoder.encoder", {'content': encoder})


    file_train = __file_out("train", p_case)
    log_message("训练数据集：%s" % file_train)
    log_message("训练集Y：", dict(Counter(train_df[indexes])))
    train_df.to_csv(file_train, index=False)

    file_test = __file_out("test", p_case)
    log_message("验证数据集：%s" % file_test)
    log_message("验证集Y：", dict(Counter(test_df[indexes])))
    test_df = test_df.drop(f_target, axis=1)  # 验证集中必须去掉 Target
    test_df.to_csv(file_test, index=False)

    file_target = __file_out("target", p_case)
    log_message("验证结果：%s" % file_target)
    target_df.to_csv(file_target, index=False)

    log_info("训练数据Shape：", train_df.shape)
    log_info("验证数据Shape：", test_df.shape)
    log_success("------>「第一步」数据拆分完成！比例：\033[31m%.2f" % p_radio)
    return train_df, test_df, target_df


def data_split_fn(df_data, p_case, p_radio=0.2, f_target_binary=None):
    return lambda f_id, f_target: data_split(df_data, f_id, f_target, p_case, p_radio, f_target_binary)

# ------------------------- 评分和报表

def data_score(df_true, df_predict, f_target, o_target):
    # ValueError: Classification metrics can't handle a mix of binary and continuous targets
    # 此处可以 astype("int") 或 astype("float") 来解决
    y_true = df_true[f_target].astype("int")
    y_pred = df_predict[o_target].astype("int")
    f1 = f1_score(y_true, y_pred, average=None)
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')
    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 精确率
    precision = precision_score(y_true, y_pred, average=None)
    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    # 召回率
    recall = recall_score(y_true, y_pred, average=None)
    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    log_info("F1宏观 Macro：\033[31m", macro)
    log_info("F2微观 Micro：\033[31m", micro)
    log_info("准确率 Accuracy：\033[31m", accuracy)
    log_info("精确率宏观 Precision Macro：\033[31m", precision_macro)
    log_info("精确率微观 Precision Micro：\033[31m", precision_micro)
    log_info("召回率宏观 Macro：\033[31m", recall_macro)
    log_info("召回率微观 Micro：\033[31m", recall_micro)
    return {
        'f1_macro': macro,
        'f1_micro': micro,
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro
    }


def data_score_fn(df_true, df_predict, o_target):
    return lambda f_id, f_target: data_score(df_true, df_predict, f_target, o_target)
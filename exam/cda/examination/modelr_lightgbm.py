from lightgbm import LGBMRegressor

from examination.toolkit import *


class ModRLightGBM(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()
        model = LGBMRegressor(
            objective='regression',
            max_depth=3,
            learning_rate=0.07,
            n_estimators=10000,
            metric='rmse',
            bagging_fraction=0.8,
            feature_fraction=0.8
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModRLightGBM）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration

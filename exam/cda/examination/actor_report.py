from examination.actor_function import *

class Answer:
    def __init__(self):
        self.report = {}

    def put(self, name, modeler):
        self.report[name] = modeler
        return self

    def output(self, executor):
        keys = self.report.keys()
        tpl = report_tpl(keys)
        for key in keys:
            print("\033[36m ---------> 执行算法：%s --------> \033[37m" % key)
            modeler = self.report[key]
            params = executor(modeler, key + ".csv")
            report_add(tpl, params)
        return pd.DataFrame(tpl)

import pdb
class Test_fun(object):

    default_config = 100

    def __init__(self):
        self.config = self.default_config
    def printConfig(self):
        print(self.config)


class Child_fun(Test_fun):

    default_config = 200

    def __init__(self):
        self.config = self.default_config



child_fun = Child_fun()
child_fun.config
child_fun.config = 300
pdb.set_trace()
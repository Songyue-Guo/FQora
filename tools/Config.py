class Config:
    def __init__(self):
        self.settings = {}
        self.default_values = {
            
        }

    def set(self, key, value):
        """single set"""
        self.settings[key] = value
    
    def set_dic(self,config_dic):
        """set from dic"""
        for key,value in config_dic.items():
            self.set(key,value)    
    def args2cfg(self,**kwargs):
        for key, value in kwargs.items():
            self.set(key,value)
    def get(self, key, default=None):
        """获取配置项，如果不存在则返回默认值"""
        return self.settings.get(key, default)

    def load_from_file(self, filename):
        """从文件加载配置项"""
        with open(filename, 'r') as f:
            # 假设配置文件是简单的key=value格式
            for line in f:
                key, value = line.strip().split('=', 1)
                self.set(key, value)

    def __str__(self):

        return str(self.settings)



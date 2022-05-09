class TabuList:

    def __init__(self, max_len=100000):
        self.max_len = max_len
        self.l = [None] * self.max_len
        self.d = dict()
        self.__counter = 0

    def add(self, key, value=None):
        el = self.l[self.__counter]
        if el is not None:
            del self.d[el]

        nb = len(self.d)
        self.d[key] = value
        if len(self.d) == nb:
            return
        self.l[self.__counter] = key
        self.__counter = (self.__counter + 1) % self.max_len

    def get(self, key, default_value):
        return self.d.get(key, default_value)

    def clear(self):
        self.l = [None] * self.max_len
        self.d.clear()
        self.__counter = 0

import pickle


class Log:
    def __init__(self, file_name, ga_info, param_limits):
        self.file = open(file_name, 'wb')
        pickle.dump({'algorithm_info': ga_info}, self.file)
        pickle.dump({'parameter limits': param_limits}, self.file)

    def log_generation(self, iteration, population):
        pickle.dump({'iteration': iteration, 'population': population}, self.file)

    def __del__(self):
        self.file.close()

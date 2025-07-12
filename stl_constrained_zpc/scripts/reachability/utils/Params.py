class Params():
    def __init__(self, tFinal = None, U = None, R0 = None, dt = None):
        self.params = {'tFinal': tFinal, 
                       'U'     : U,
                       'R0'    : R0, 
                       'dt'    : dt}

    
    def __str__(self):
        str_repr = "tFinal: {} \n U: {} \n R0: {}".format(self.params['tFinal'], self.params['U'], self.params['R0'])
        return str_repr
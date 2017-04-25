@register
class Plusminusone(Initializer):
    """Initialize the weight with random +1 or -1 """
    def __init__(self):
        super(Plusminusone, self).__init__()

    def _init_weight(self, _, arr):
        arr[:] = np.random.randint(0, 2, arr.shape)*2-1
        

@register
class Index(Initializer):
    """Initialize the weight within range [0, up_value]

    Parameters
    ----------
    up_value : int
        The range of the index
    """
    def __init__(self, up_value):
        super(Index, self).__init__(up_value = up_value)
        self.up_value = up_value

    def _init_weight(self, _, arr):
        arr[:] = np.random.randint(0, self.up_value,arr.shape)

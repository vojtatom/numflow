

class Geometry:
    def __init__(self):
        pass


class Points(Geometry):
    def __init__(self):
        pass


class Box(Geometry):
    def __init__(self, low, high):
        pass


class Streamline(Geometry):
    def __init__(self, dataset, points, t0=0, tbound=1, solver='RK45'):
        pass


class Glyphs(Geometry):
    def __init__(self, dataset, points):
        pass


class Layer(Geometry):
    def __init__(self, dataset, points):
        pass




from __future__ import annotations

import matplotlib


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        self.hexs = ('#FF0000', '#ad6be3', '#59e6e3', '#47918a', '#fabf00', '#00adeb', '#d982de')
        self.palette = [matplotlib.colors.to_rgb(c) for c in self.hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False, hexs=True):
        if not hexs:
            c = self.palette[int(i) % self.n]
            if bgr:
                c = (c[2], c[1], c[0])
        else:
            return self.hexs[int(i) % self.n]


colors = Colors()  # create instance for 'from utils.plots import colors'

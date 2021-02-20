import matplotlib.pyplot as plt


class Visualization:

    @staticmethod
    def plot_quadruple_diff(res_trad, res_extend, res_geom, res_extremal):
        diff_extremal = []
        diff_extend = []
        diff_geom = []
        for keys in res_trad:
            if res_trad[keys] != res_extend[keys]:
                diff_extend.append(keys)
            if res_trad[keys] != res_geom[keys]:
                diff_geom.append(keys)
            if res_trad[keys] != res_extremal[keys]:
                diff_extremal.append(keys)
        x = ['extended', 'geometric', 'extremal']
        y = [len(diff_extend), len(diff_geom), len(diff_extremal)]
        plt.bar(x, y)
        plt.show()

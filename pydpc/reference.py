import numpy as _np
import matplotlib.pyplot as _plt

class LaioCluster(object):
    def __init__(self, fraction=0.02):
        self.fraction = fraction
    def load(self, points):
        self.points = points
        self.npoints = self.points.shape[0]
        self._get_distance()
        self._get_kernel_size()
        self._get_density()
        self._get_delta_and_neighbour()
        self.draw_decision_graph()
    def draw_decision_graph(self, min_density=None, min_delta=None):
        fig, ax = _plt.subplots(figsize=(8, 4.5))
        ax.scatter(self.density, self.delta, s=40)
        if min_density is not None and min_delta is not None:
            ax.plot(
                [min_density, self.density.max()], [min_delta, min_delta], linewidth=2, color="red")
            ax.plot(
                [min_density, min_density], [min_delta, self.delta.max()], linewidth=2, color="red")
        ax.set_xlabel(r"density", fontsize=20)
        ax.set_ylabel(r"delta / a.u.", fontsize=20)
        ax.tick_params(labelsize=15)
    def assign(self, min_density, min_delta, border_only=False):
        self.min_density = min_density
        self.min_delta = min_delta
        self.border_only = border_only
        self.draw_decision_graph(self.min_density, self.min_delta)
        self._get_cluster_indices()
        self._get_membership()
        self._get_halo()
    def _get_distance(self):
        self.distance = _np.zeros(shape=(self.npoints, self.npoints), dtype=_np.float64)
        for i in range(self.npoints - 1):
            for j in range(i + 1, self.npoints):
                self.distance[i, j] = _np.linalg.norm(self.points[i, :] - self.points[j, :])
                self.distance[j, i] = self.distance[i, j]
    def _get_kernel_size(self):
        arr = self.distance[0, 1:]
        for i in range(1, self.npoints - 1):
            arr = _np.hstack((arr, self.distance[i, i + 1:]))
        arr = _np.sort(arr)
        imax = int(_np.floor(0.5 + self.fraction * arr.shape[0]))
        self.kernel_size = arr[imax]
    def _get_density(self):
        self.density = _np.zeros(shape=(self.npoints,), dtype=_np.float64)
        for i in range(self.npoints - 1):
            for j in range(i + 1, self.npoints):
                rho = _np.exp(-(self.distance[i, j] / self.kernel_size)**2)
                self.density[i] += rho
                self.density[j] += rho
    def _get_delta_and_neighbour(self):
        self.order = _np.argsort(self.density)[::-1]
        max_distance = self.distance.max()
        self.delta = _np.zeros(shape=self.order.shape, dtype=_np.float64)
        self.delta[self.order[0]] = -1.0
        self.delta[self.order[1:]] = max_distance
        self.neighbour = _np.empty_like(self.order)
        self.neighbour[:] = -1
        for i in range(1, self.npoints):
            for j in range(i):
                if self.distance[self.order[i], self.order[j]] < self.delta[self.order[i]]:
                    self.delta[self.order[i]] = self.distance[self.order[i], self.order[j]]
                    self.neighbour[self.order[i]] = self.order[j]
        self.delta[self.order[0]] = self.delta.max()
    def _get_cluster_indices(self):
        self.cluster = _np.intersect1d(
            _np.where(self.density > self.min_density)[0],
            _np.where(self.delta > self.min_delta)[0], assume_unique=True)
        self.ncl = self.cluster.shape[0]
    def _get_membership(self):
        self.membership = -1 * _np.ones(shape=self.order.shape, dtype=_np.intc)
        for i in range(self.ncl):
            self.membership[self.cluster[i]] = i
        for i in range(self.npoints):
            if self.membership[self.order[i]] == -1:
                self.membership[self.order[i]] = self.membership[self.neighbour[self.order[i]]]
    def _get_halo(self):
        self.halo = self.membership.copy()
        self.border_density = _np.zeros(shape=(self.ncl,), dtype=_np.float64)
        self.border_member = _np.zeros(shape=self.membership.shape, dtype=_np.bool)
        for i in range(self.npoints - 1):
            for j in range(i + 1, self.npoints):
                if (self.membership[i] != self.membership[j]) and (self.distance[i, j] < self.kernel_size):
                    average_density = 0.5 * (self.density[i] + self.density[j])
                    if self.border_density[self.membership[i]] < average_density:
                        self.border_density[self.membership[i]] = average_density
                    if self.border_density[self.membership[j]] < average_density:
                        self.border_density[self.membership[j]] = average_density
                    self.border_member[i] = True
                    self.border_member[j] = True
        if self.border_only:
            for i in range(self.npoints):
                if (self.density[i] < self.border_density[self.membership[i]]) and self.border_member[i]:
                    self.halo[i] = -1
        else:
            for i in range(self.npoints):
                if (self.density[i] < self.border_density[self.membership[i]]):
                    self.halo[i] = -1
        self.halo_idx = (self.halo == -1)
        self.core_idx = _np.invert(self.halo_idx)

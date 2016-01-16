# This file is part of pydpc.
#
# Copyright 2016 Christoph Wehmeyer
#
# pydpc is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""This module provides a previously tested development version. It is rather
slow and only used for consistency checks"""

import numpy as _np
import matplotlib.pyplot as _plt

class Cluster(object):
    def __init__(self, fraction=0.02, autoplot=True):
        self.fraction = fraction
        self.autoplot = autoplot
    def load(self, points):
        self.points = points
        self.npoints = self.points.shape[0]
        self._get_distances()
        self._get_kernel_size()
        self._get_density()
        self._get_delta_and_neighbour()
        if self.autoplot:
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
        if self.autoplot:
            self.draw_decision_graph(self.min_density, self.min_delta)
        self._get_cluster_indices()
        self._get_membership()
        self._get_halo()
    def _get_distances(self):
        self.distances = _np.zeros(shape=(self.npoints, self.npoints), dtype=_np.float64)
        for i in range(self.npoints - 1):
            for j in range(i + 1, self.npoints):
                self.distances[i, j] = _np.linalg.norm(self.points[i, :] - self.points[j, :])
                self.distances[j, i] = self.distances[i, j]
    def _get_kernel_size(self):
        arr = self.distances[0, 1:]
        for i in range(1, self.npoints - 1):
            arr = _np.hstack((arr, self.distances[i, i + 1:]))
        arr = _np.sort(arr)
        imax = int(_np.floor(0.5 + self.fraction * arr.shape[0]))
        self.kernel_size = arr[imax]
    def _get_density(self):
        self.density = _np.zeros(shape=(self.npoints,), dtype=_np.float64)
        for i in range(self.npoints - 1):
            for j in range(i + 1, self.npoints):
                rho = _np.exp(-(self.distances[i, j] / self.kernel_size)**2)
                self.density[i] += rho
                self.density[j] += rho
    def _get_delta_and_neighbour(self):
        self.order = _np.argsort(self.density)[::-1]
        max_distance = self.distances.max()
        self.delta = _np.zeros(shape=self.order.shape, dtype=_np.float64)
        self.delta[self.order[0]] = -1.0
        self.delta[self.order[1:]] = max_distance
        self.neighbour = _np.empty_like(self.order)
        self.neighbour[:] = -1
        for i in range(1, self.npoints):
            for j in range(i):
                if self.distances[self.order[i], self.order[j]] < self.delta[self.order[i]]:
                    self.delta[self.order[i]] = self.distances[self.order[i], self.order[j]]
                    self.neighbour[self.order[i]] = self.order[j]
        self.delta[self.order[0]] = self.delta.max()
    def _get_cluster_indices(self):
        self.clusters = _np.intersect1d(
            _np.where(self.density > self.min_density)[0],
            _np.where(self.delta > self.min_delta)[0], assume_unique=True)
        self.ncl = self.clusters.shape[0]
    def _get_membership(self):
        self.membership = -1 * _np.ones(shape=self.order.shape, dtype=_np.intc)
        for i in range(self.ncl):
            self.membership[self.clusters[i]] = i
        for i in range(self.npoints):
            if self.membership[self.order[i]] == -1:
                self.membership[self.order[i]] = self.membership[self.neighbour[self.order[i]]]
    def _get_halo(self):
        self.halo = self.membership.copy()
        self.border_density = _np.zeros(shape=(self.ncl,), dtype=_np.float64)
        self.border_member = _np.zeros(shape=self.membership.shape, dtype=_np.bool)
        for i in range(self.npoints - 1):
            for j in range(i + 1, self.npoints):
                if (self.membership[i] != self.membership[j]) and (self.distances[i, j] < self.kernel_size):
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
        self.halo_idx = _np.where(self.halo == -1)[0]
        self.core_idx = _np.where(self.halo != -1)[0]

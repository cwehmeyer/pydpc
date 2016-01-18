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

import numpy as _np
import matplotlib.pyplot as _plt
from . import core as _core

__all__ = ['Cluster']

class Distances(object):
    def __init__(self, points):
        self.points = points
        self.npoints = self.points.shape[0]
        self.distances = _core.get_distances(self.points)
        self.max_distance = self.distances.max()

class Density(Distances):
    def __init__(self, points, fraction):
        super(Density, self).__init__(points)
        self.fraction = fraction
        self.kernel_size = _core.get_kernel_size(self.distances, self.fraction)
        self.density = _core.get_density(self.distances, self.kernel_size)

class Graph(Density):
    def __init__(self, points, fraction):
        super(Graph, self).__init__(points, fraction)
        self.order = _np.ascontiguousarray(_np.argsort(self.density).astype(_np.intc)[::-1])
        self.delta, self.neighbour = _core.get_delta_and_neighbour(
            self.order, self.distances, self.max_distance)

class Cluster(Graph):
    def __init__(self, points, fraction=0.02, autoplot=True):
        super(Cluster, self).__init__(points, fraction)
        self.autoplot = autoplot
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
        return fig, ax
    def assign(self, min_density, min_delta, border_only=False):
        self.min_density = min_density
        self.min_delta = min_delta
        self.border_only = border_only
        if self.autoplot:
            self.draw_decision_graph(self.min_density, self.min_delta)
        self._get_cluster_indices()
        self.membership = _core.get_membership(self.clusters, self.order, self.neighbour)
        self.border_density, self.border_member = _core.get_border(
            self.kernel_size, self.distances, self.density, self.membership, self.nclusters)
        self.halo_idx, self.core_idx = _core.get_halo(
            self.density, self.membership,
            self.border_density, self.border_member.astype(_np.intc), border_only=border_only)
    def _get_cluster_indices(self):
        self.clusters = _np.intersect1d(
            _np.where(self.density > self.min_density)[0],
            _np.where(self.delta > self.min_delta)[0], assume_unique=True).astype(_np.intc)
        self.nclusters = self.clusters.shape[0]

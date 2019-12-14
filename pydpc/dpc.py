# This file is part of pydpc.
#
# Copyright 2016-2019 Christoph Wehmeyer
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

import numpy as np
import matplotlib.pyplot as plt
from . import core

__all__ = ['Cluster']


class Distances(object):
    """Compute pairwise distances.

    Parameters
    ----------
    points : np.ndarray of shape (N, D)
        N points in D-dimensional space to cluster.
    distances : np.ndarray of shape (N, N). Default is None.
        Precomputed distances (optional).

    """

    def __init__(self, points, distances=None):
        self.points = points
        self.npoints = self.points.shape[0]
        if distances is None:
            self.distances = core.get_distances(self.points)
        else:
            if not distances.shape == (self.npoints, self.npoints):
                raise ValueError(
                    "Distance matrix must have shape (N, N)"
                )
            self.distances = distances
        self.max_distance = self.distances.max()


class Density(Distances):
    """Compute local densities.

    Parameters
    ----------
    points : np.ndarray of shape (N, D)
        N points in D-dimensional space to cluster.
    fraction : float
        Hyper-parameter for kernel_size estimation.
    kernel_size : float. Default is None.
        Kernel size for density estimation (optional).

    """

    def __init__(self, points, fraction, kernel_size=None, **kwargs):
        super(Density, self).__init__(points, **kwargs)
        self.fraction = fraction
        if kernel_size is None:
            self.kernel_size = core.get_kernel_size(
                self.distances,
                self.fraction
            )
        else:
            self.kernel_size = kernel_size
        if self.kernel_size <= 0:
            raise ValueError(
                f"kernel_size = {kernel_size} is invalid; must be strictly"
                " positive. This can occur in the degenerate case where"
                " the distance matrix is all zeros, check your input."
            )
        self.density = core.get_density(self.distances, self.kernel_size)


class Graph(Density):
    """Compute point order.

    Parameters
    ----------
    points : np.ndarray of shape (N, D)
        N points in D-dimensional space to cluster.
    fraction : float
        Hyper-parameter for kernel_size estimation.

    """

    def __init__(self, points, fraction, **kwargs):
        super(Graph, self).__init__(points, fraction, **kwargs)
        self.order = np.ascontiguousarray(
            np.argsort(self.density).astype(np.intc)[::-1]
        )
        [
            self.delta,
            self.neighbour
        ] = core.get_delta_and_neighbour(
            self.order,
            self.distances,
            self.max_distance
        )


class Cluster(Graph):
    """Cluster a point cloud.

    Parameters
    ----------
    points : np.ndarray of shape (N, D)
        N points in D-dimensional space to cluster.
    fraction : float. Default is 0.02.
        Hyper-parameter for kernel_size estimation (optional).
    autoplot : bool. Default is True.
        Directly draw the decision graph upon object instanciation.

    """

    def __init__(self, points, fraction=0.02, autoplot=True, **kwargs):
        super(Cluster, self).__init__(points, fraction, **kwargs)
        self.autoplot = autoplot
        if self.autoplot:
            self.draw_decision_graph()

    def draw_decision_graph(self, min_density=None, min_delta=None):
        """Draw the decision graph.

        Parameters
        ----------
        min_density : float. Default is None.
            Minimal local density for cluster center selection.
        min_delta : float. Default is None.
            Minimal delta for cluster center selection.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object for the decision graph.
        matplotlib.axes._subplots.AxesSubplot
            The axes object for the decision graph.

        """
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.density, self.delta, s=40)
        if min_density is not None and min_delta is not None:
            ax.plot(
                [min_density, self.density.max()],
                [min_delta, min_delta],
                linewidth=2,
                color="red"
            )
            ax.plot(
                [min_density, min_density],
                [min_delta, self.delta.max()],
                linewidth=2,
                color="red"
            )
        ax.set_xlabel(r"density", fontsize=20)
        ax.set_ylabel(r"delta / a.u.", fontsize=20)
        ax.tick_params(labelsize=15)
        return fig, ax

    def _get_cluster_indices(self):
        self.clusters = np.intersect1d(
            np.where(self.density > self.min_density)[0],
            np.where(self.delta > self.min_delta)[0],
            assume_unique=True
        ).astype(np.intc)
        self.nclusters = self.clusters.shape[0]

    def assign(self, min_density, min_delta, border_only=False):
        """Assign points to cluster centers.

        Parameters
        ----------
        min_density : float
            Minimal local density for cluster center selection.
        min_delta : float
            Minimal delta for cluster center selection.
        border_only : bool. Default is False.
            Exclude all halo members from assignment or only those
            who are in border regions between clusters.

        """
        self.min_density = min_density
        self.min_delta = min_delta
        self.border_only = border_only
        if self.autoplot:
            self.draw_decision_graph(self.min_density, self.min_delta)
        self._get_cluster_indices()
        self.membership = core.get_membership(
            self.clusters,
            self.order,
            self.neighbour
        )
        [
            self.border_density,
            self.border_member
        ] = core.get_border(
            self.kernel_size,
            self.distances,
            self.density,
            self.membership,
            self.nclusters
        )
        [
            self.halo_idx,
            self.core_idx
        ] = core.get_halo(
            self.density,
            self.membership,
            self.border_density,
            self.border_member.astype(np.intc),
            border_only=border_only
        )

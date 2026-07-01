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

"""Clustering by fast search and find of density peaks (Rodriguez & Laio, 2014).

The public entry point is :class:`Cluster`; the other classes in this module
are internal building blocks that successively compute the pairwise
distances, the local density, and the decision-graph quantities it relies on.
"""

import matplotlib.pyplot as _plt
import numpy as _np

from . import core as _core

__all__ = ["Cluster"]


class Distances:
    """Compute (or accept) the pairwise Euclidean distance matrix for a point set.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_dim)
        Coordinates of the data points to cluster.
    distances : ndarray of shape (n_points, n_points), optional
        Precomputed pairwise distance matrix. If given, it is used as-is
        instead of being recomputed from `points`.

    Attributes
    ----------
    points : ndarray of shape (n_points, n_dim)
    npoints : int
        Number of data points.
    distances : ndarray of shape (n_points, n_points)
        Pairwise Euclidean distance matrix.
    max_distance : float
        Largest entry in `distances`.

    Raises
    ------
    ValueError
        If `distances` is given but its shape is not (n_points, n_points).
    """

    def __init__(self, points, distances=None):
        self.points = points
        self.npoints = self.points.shape[0]

        if distances is None:
            self.distances = _core.get_distances(self.points)
        else:
            if not distances.shape == (self.npoints, self.npoints):
                raise ValueError("Distance matrix must have shape (n_points, n_points)")
            self.distances = distances

        self.max_distance = self.distances.max()


class Density(Distances):
    """Estimate each point's local density from the distance matrix.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_dim)
        Coordinates of the data points to cluster.
    fraction : float
        Average fraction of all other points to treat as neighbours when
        estimating the density kernel size. Ignored if `kernel_size` is given.
    kernel_size : float, optional
        Explicit kernel size to use instead of estimating one from `fraction`.
    **kwargs
        Additional keyword arguments forwarded to :class:`Distances`.

    Attributes
    ----------
    fraction : float
    kernel_size : float
        Kernel size used to estimate the density.
    density : ndarray of shape (n_points,)
        Local density of each data point.

    Raises
    ------
    ValueError
        If the (estimated or given) `kernel_size` is not strictly positive,
        which can happen if the distance matrix is degenerate (all zeros).
    """

    def __init__(self, points, fraction, kernel_size=None, **kwargs):
        super().__init__(points, **kwargs)
        self.fraction = fraction
        if kernel_size is None:
            self.kernel_size = _core.get_kernel_size(self.distances, self.fraction)
        else:
            self.kernel_size = kernel_size
        if self.kernel_size <= 0:
            raise ValueError(
                f"kernel_size = {self.kernel_size} is invalid; must be "
                "strictly positive. This can occur in the degenerate case "
                "where the distance matrix is all zeros, check your input."
            )
        self.density = _core.get_density(self.distances, self.kernel_size)


class Graph(Density):
    """Compute the decision-graph quantities `delta` and `neighbour`.

    For each point, `delta` is the minimal distance to any point of higher
    density, and `neighbour` is the index of that nearer, denser point.
    Points of (locally) highest density are the candidate cluster centers
    that stand out with high `density` and high `delta`.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_dim)
        Coordinates of the data points to cluster.
    fraction : float
        Forwarded to :class:`Density`.
    **kwargs
        Additional keyword arguments forwarded to :class:`Density`.

    Attributes
    ----------
    order : ndarray of shape (n_points,)
        Point indices sorted by decreasing density.
    delta : ndarray of shape (n_points,)
        Minimal distance of each point to a point of higher density.
    neighbour : ndarray of shape (n_points,)
        Index of the nearest point of higher density for each point.
    """

    def __init__(self, points, fraction, **kwargs):
        super().__init__(points, fraction, **kwargs)
        self.order = _np.ascontiguousarray(_np.argsort(self.density).astype(_np.intc)[::-1])
        self.delta, self.neighbour = _core.get_delta_and_neighbour(
            self.order, self.distances, self.max_distance
        )


class Cluster(Graph):
    """Density-peak clustering of a set of points.

    Constructing a `Cluster` computes the pairwise distances, local density,
    and decision-graph quantities (`density`, `delta`) for `points`. Cluster
    centers are then chosen interactively by inspecting the decision graph
    and calling :meth:`assign` with density/delta thresholds that isolate the
    outlying points (the candidate centers).

    Examples
    --------
    >>> clu = Cluster(points)  # doctest: +SKIP
    >>> clu.assign(min_density=20, min_delta=1.5)  # doctest: +SKIP
    >>> clu.membership  # cluster index for each point  # doctest: +SKIP

    Parameters
    ----------
    points : ndarray of shape (n_points, n_dim)
        Coordinates of the data points to cluster.
    fraction : float, default=0.02
        Average fraction of all other points to treat as neighbours when
        estimating the density kernel size.
    autoplot : bool, default=True
        If True, automatically draw the decision graph on construction and
        again (with the chosen thresholds indicated) whenever :meth:`assign`
        is called.
    **kwargs
        Additional keyword arguments forwarded to :class:`Graph`, e.g. a
        precomputed `distances` matrix or an explicit `kernel_size`.

    Attributes
    ----------
    density : ndarray of shape (n_points,)
        Local density of each data point.
    delta : ndarray of shape (n_points,)
        Minimal distance of each point to a point of higher density.
    clusters : ndarray of int
        Indices of the chosen cluster centers. Set by :meth:`assign`.
    membership : ndarray of shape (n_points,)
        Cluster index assigned to each point. Set by :meth:`assign`.
    halo_idx : ndarray of int
        Indices of points classified as halo (low-confidence) members.
        Set by :meth:`assign`.
    core_idx : ndarray of int
        Indices of points classified as core (high-confidence) members.
        Set by :meth:`assign`.
    """

    def __init__(self, points, fraction=0.02, autoplot=True, **kwargs):
        super().__init__(points, fraction, **kwargs)
        self.autoplot = autoplot
        if self.autoplot:
            self.draw_decision_graph()

    def draw_decision_graph(self, min_density=None, min_delta=None):
        """Plot the decision graph (density vs. delta) for all points.

        Parameters
        ----------
        min_density : float, optional
            If given together with `min_delta`, draw red threshold lines
            marking the region density > `min_density` and delta > `min_delta`.
        min_delta : float, optional
            See `min_density`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        fig, ax = _plt.subplots(figsize=(5, 5))
        ax.scatter(self.density, self.delta, s=40)
        if min_density is not None and min_delta is not None:
            ax.plot(
                [min_density, self.density.max()],
                [min_delta, min_delta],
                linewidth=2,
                color="red",
            )
            ax.plot(
                [min_density, min_density],
                [min_delta, self.delta.max()],
                linewidth=2,
                color="red",
            )
        ax.set_xlabel(r"density", fontsize=20)
        ax.set_ylabel(r"delta / a.u.", fontsize=20)
        ax.tick_params(labelsize=15)
        return fig, ax

    def assign(self, min_density, min_delta, border_only=False):
        """Choose cluster centers via decision-graph thresholds and assign points.

        Points with `density` above `min_density` and `delta` above
        `min_delta` are taken as cluster centers; every other point is then
        assigned to the cluster of its nearest denser neighbour.

        Parameters
        ----------
        min_density : float
            Minimum density for a point to be considered a cluster center.
        min_delta : float
            Minimum delta for a point to be considered a cluster center.
        border_only : bool, default=False
            If False (default), classify as halo any point whose density is
            below its cluster's border density (the original Rodriguez/Laio
            criterion). If True, only classify points as halo where they
            border a *different* cluster, which is less strict and keeps
            more points in the core.

        Returns
        -------
        None
            Results are stored on the instance; see the `clusters`,
            `membership`, `border_density`, `border_member`, `halo_idx`, and
            `core_idx` attributes.
        """
        self.min_density = min_density
        self.min_delta = min_delta
        self.border_only = border_only
        if self.autoplot:
            self.draw_decision_graph(self.min_density, self.min_delta)
        self._get_cluster_indices()
        self.membership = _core.get_membership(self.clusters, self.order, self.neighbour)
        self.border_density, self.border_member = _core.get_border(
            self.kernel_size,
            self.distances,
            self.density,
            self.membership,
            self.nclusters,
        )
        self.halo_idx, self.core_idx = _core.get_halo(
            self.density,
            self.membership,
            self.border_density,
            self.border_member.astype(_np.intc),
            border_only=border_only,
        )

    def _get_cluster_indices(self):
        self.clusters = _np.intersect1d(
            _np.where(self.density > self.min_density)[0],
            _np.where(self.delta > self.min_delta)[0],
            assume_unique=True,
        ).astype(_np.intc)
        self.nclusters = self.clusters.shape[0]

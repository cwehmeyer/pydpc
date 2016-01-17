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
cimport numpy as _np

cdef extern from "_core.h":
    void _get_distances(double *points, int npoints, int ndim, double *distances)
    double _get_kernel_size(double *distances, int npoints, double fraction)
    void _get_density(double kernel_size, double *distances, int npoints, double *density)
    void _get_delta_and_neighbour(
        double max_distance, double *distances, int *order,
        int npoints, double *delta, int *neighbour)
    void _get_membership(
        int *clusters, int nclusters, int *order, int *neighbour, int npoints, int *membership)
    void _get_border(
        double kernel_size, double *distances, double *density, int *membership, int npoints,
        int *border_member, double *border_density)
    void _get_halo(
        int border_only, double *border_density,
        double *density, int *membership, int *border_member, int npoints, int *halo)

def get_distances(_np.ndarray[double, ndim=2, mode="c"] points not None):
    npoints = points.shape[0]
    ndim = points.shape[1]
    distances = _np.zeros(shape=(npoints, npoints), dtype=_np.float64)
    _get_distances(
        <double*> _np.PyArray_DATA(points),
        npoints, ndim,
        <double*> _np.PyArray_DATA(distances))
    return distances

def get_kernel_size(_np.ndarray[double, ndim=2, mode="c"] distances not None, fraction):
    return _get_kernel_size(<double*> _np.PyArray_DATA(distances), distances.shape[0], fraction)

def get_density(_np.ndarray[double, ndim=2, mode="c"] distances not None, kernel_size):
    npoints = distances.shape[0]
    density = _np.zeros(shape=(npoints,), dtype=_np.float64)
    _get_density(
        kernel_size,
        <double*> _np.PyArray_DATA(distances),
        npoints,
        <double*> _np.PyArray_DATA(density))
    return density

def get_delta_and_neighbour(
    _np.ndarray[int, ndim=1, mode="c"] order not None,
    _np.ndarray[double, ndim=2, mode="c"] distances not None,
    max_distance):
    npoints = distances.shape[0]
    delta = _np.zeros(shape=(npoints,), dtype=_np.float64)
    neighbour = _np.zeros(shape=(npoints,), dtype=_np.intc)
    _get_delta_and_neighbour(
        max_distance,
        <double*> _np.PyArray_DATA(distances),
        <int*> _np.PyArray_DATA(order),
        npoints,
        <double*> _np.PyArray_DATA(delta),
        <int*> _np.PyArray_DATA(neighbour))
    return delta, neighbour

def get_membership(
    _np.ndarray[int, ndim=1, mode="c"] clusters not None,
    _np.ndarray[int, ndim=1, mode="c"] order not None,
    _np.ndarray[int, ndim=1, mode="c"] neighbour not None):
    npoints = order.shape[0]
    membership = _np.zeros(shape=(npoints,), dtype=_np.intc)
    _get_membership(
        <int*> _np.PyArray_DATA(clusters),
        clusters.shape[0],
        <int*> _np.PyArray_DATA(order),
        <int*> _np.PyArray_DATA(neighbour),
        npoints,
        <int*> _np.PyArray_DATA(membership))
    return membership

def get_border(
    kernel_size,
    _np.ndarray[double, ndim=2, mode="c"] distances not None,
    _np.ndarray[double, ndim=1, mode="c"] density not None,
    _np.ndarray[int, ndim=1, mode="c"] membership not None,
    nclusters):
    npoints = distances.shape[0]
    border_density = _np.zeros(shape=(nclusters,), dtype=_np.float64)
    border_member = _np.zeros(shape=(npoints,), dtype=_np.intc)
    _get_border(
        kernel_size,
        <double*> _np.PyArray_DATA(distances),
        <double*> _np.PyArray_DATA(density),
        <int*> _np.PyArray_DATA(membership),
        npoints,
        <int*> _np.PyArray_DATA(border_member),
        <double*> _np.PyArray_DATA(border_density))
    return border_density, border_member.astype(_np.bool)

def get_halo(
    _np.ndarray[double, ndim=1, mode="c"] density not None,
    _np.ndarray[int, ndim=1, mode="c"] membership not None,
    _np.ndarray[double, ndim=1, mode="c"] border_density not None,
    _np.ndarray[int, ndim=1, mode="c"] border_member not None,
    border_only=False):
    halo = membership.copy()
    flag = 0
    if border_only:
        flag = 1
    _get_halo(
        flag,
        <double*> _np.PyArray_DATA(border_density),
        <double*> _np.PyArray_DATA(density),
        <int*> _np.PyArray_DATA(membership),
        <int*> _np.PyArray_DATA(border_member),
        density.shape[0],
        <int*> _np.PyArray_DATA(halo))
    halo_idx = _np.where(halo == -1)[0].astype(_np.intc)
    core_idx = _np.where(halo != -1)[0].astype(_np.intc)
    return halo_idx, core_idx

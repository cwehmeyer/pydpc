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

from pydpc._reference import Cluster as Ref
from pydpc import Cluster
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

class TestFourGaussians2D(object):
    @classmethod
    def setup_class(cls):
        # data generation
        cls.npoints = 1000
        cls.mux = 1.8
        cls.muy = 1.8
        cls.fraction = 0.02
        cls.points = np.zeros(shape=(cls.npoints, 2), dtype=np.float64)
        cls.points[:, 0] = np.random.randn(cls.npoints) + \
            cls.mux * (-1)**np.random.randint(0, high=2, size=cls.npoints)
        cls.points[:, 1] = np.random.randn(cls.npoints) + \
            cls.muy * (-1)**np.random.randint(0, high=2, size=cls.npoints)
        # cluster initialisation
        cls.ref = Ref(cls.fraction, autoplot=False)
        cls.ref.load(cls.points)
        cls.ref.assign(20, 1.5)
        cls.dpc = Cluster(cls.points, cls.fraction, autoplot=False)
        cls.dpc.assign(20, 1.5)
    @classmethod
    def teardown_class(cls):
        pass
    def setup(self):
        pass
    def teardown(self):
        pass
    def test_distances(self):
        assert_almost_equal(self.dpc.distances, self.ref.distances, decimal=10)
    def test_distances(self):
        assert_almost_equal(self.dpc.kernel_size, self.ref.kernel_size, decimal=10)
    def test_density(self):
        assert_almost_equal(self.dpc.density, self.ref.density, decimal=10)
    def test_order(self):
        assert_array_equal(self.dpc.order, self.ref.order)
    def test_delta(self):
        assert_almost_equal(self.dpc.delta, self.ref.delta, decimal=10)
    def test_neighbour(self):
        assert_array_equal(self.dpc.neighbour, self.ref.neighbour)
    def test_clusters(self):
        assert_array_equal(self.dpc.clusters, self.ref.clusters)
    def test_membership(self):
        assert_array_equal(self.dpc.membership, self.ref.membership)
    def test_border_density(self):
        assert_almost_equal(self.dpc.border_density, self.ref.border_density, decimal=10)
    def test_border_member(self):
        assert_array_equal(self.dpc.border_member, self.ref.border_member)
    def test_halo_idx(self):
        assert_array_equal(self.dpc.halo_idx, self.ref.halo_idx)
    def test_core_idx(self):
        assert_array_equal(self.dpc.core_idx, self.ref.core_idx)

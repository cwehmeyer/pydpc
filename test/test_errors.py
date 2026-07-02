# This file is part of pydpc.
#
# Copyright 2016-2026 Christoph Wehmeyer
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
import pytest

from pydpc import Cluster


class TestEmptyClusters:
    @classmethod
    def setup_class(cls):
        cls.npoints = 200
        cls.points = np.random.randn(cls.npoints, 2)
        cls.dpc = Cluster(cls.points, autoplot=False)

    def test_assign_raises_instead_of_propagating_empty_clusters(self):
        # thresholds above every point's density/delta select zero cluster
        # centers; previously this led to undefined behaviour (and could
        # segfault) inside the C extension instead of failing cleanly (GH-8)
        with pytest.raises(ValueError, match="no cluster centers"):
            self.dpc.assign(
                min_density=self.dpc.density.max() + 1,
                min_delta=self.dpc.delta.max() + 1,
            )

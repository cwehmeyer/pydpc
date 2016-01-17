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

from .dpc import Cluster

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = "Christoph Wehmeyer"
__copyright__ = "Copyright 2016 Christoph Wehmeyer"
__license__ = "LGPLv3+"
__email__ = "christoph.wehmeyer@fu-berlin.de"

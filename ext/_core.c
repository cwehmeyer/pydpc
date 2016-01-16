/*
* This file is part of pydpc.
* 
* Copyright 2016 Christoph Wehmeyer
* 
* pydpc is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <math.h>

/***************************************************************************************************
*   C99 compatibility for macros INFINITY and NAN
***************************************************************************************************/

#ifdef _MSC_VER
    /* handle Microsofts C99 incompatibility */
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#else
    /* if not available otherwise, define INFINITY/NAN in the GNU style */
    #ifndef INFINITY
        #define INFINITY (1.0/0.0)
    #endif
    #ifndef NAN
        #define NAN (INFINITY-INFINITY)
    #endif
#endif

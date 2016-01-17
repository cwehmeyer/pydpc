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

/***************************************************************************************************
*   static convenience functions (without cython wrappers)
***************************************************************************************************/

static double sqr(double x)
{
    return (x == 0.0) ? 0.0 : x * x;
}

static double distance(double *points, int n, int m, int ndim)
{
    int i, o = n * ndim, p = m * ndim;
    double sum = 0.0;
    for(i=0; i<ndim; ++i)
        sum += sqr(points[o + i] - points[p + i]);
    return sqrt(sum);
}

static void mixed_sort(double *array, int L, int R)
/* mixed_sort() is based on examples from http://www.linux-related.de (2004) */
{
    int l, r;
    double swap;
    if(R - L > 25) /* use quicksort */
    {
        l = L - 1;
        r = R;
        for(;;)
        {
            while(array[++l] < array[R]);
            while((array[--r] > array[R]) && (r > l));
            if(l >= r) break;
            swap = array[l];
            array[l] = array[r];
            array[r] = swap;
        }
        swap = array[l];
        array[l] = array[R];
        array[R] = swap;
        mixed_sort(array, L, l - 1);
        mixed_sort(array, l + 1, R);
    }
    else /* use insertion sort */
    {
        for(l=L+1; l<=R; ++l)
        {
            swap = array[l];
            for(r=l-1; (r >= L) && (swap < array[r]); --r)
                array[r + 1] = array[r];
            array[r + 1] = swap;
        }
    }
}

/***************************************************************************************************
*   pydpc core functions (with cython wrappers)
***************************************************************************************************/

extern void _get_distances(double *points, int npoints, int ndim, double *distances)
{
    int i, j, o;
    for(i=0; i<npoints-1; ++i)
    {
        o = i * npoints;
        for(j=i+1; j<npoints; ++j)
        {
            distances[o + j] = distance(points, i, j, ndim);
            distances[j * npoints + i] = distances[o + j];
        }
    }
}

extern double _get_kernel_size(double *distances, int npoints, double fraction)
{
    int i, j, o, m = 0, n = (npoints * (npoints - 1)) / 2;
    double kernel_size;
    double *scratch = (double*) malloc(n * sizeof(double));
    for(i=0; i<npoints-1; ++i)
    {
        o = i * npoints;
        for(j=i+1; j<npoints; ++j)
            scratch[m++] = distances[o + j];
    }
    mixed_sort(scratch, 0, n - 1);
    kernel_size = scratch[(int) floor(0.5 + fraction * n)];
    free(scratch);
    return kernel_size;
}

extern void _get_density(double kernel_size, double *distances, int npoints, double *density)
{
    int i, j, o;
    double rho;
    for(i=0; i<npoints-1; ++i)
    {
        o = i * npoints;
        for(j=i+1; j<npoints; ++j)
        {
            rho = exp(-sqr(distances[o + j] / kernel_size));
            density[i] += rho;
            density[j] += rho;
        }
    }
}

extern void _get_delta_and_neighbour(
    double max_distance, double *distances, int *order, int npoints, double *delta, int *neighbour)
{
    int i, j, o;
    double max_delta = 0.0;
    for(i=0; i<npoints; ++i)
    {
        delta[order[i]] = max_distance;
        neighbour[i] = -1;
    }
    delta[order[0]] = -1.0;
    for(i=1; i<npoints; ++i)
    {
        o = order[i] * npoints;
        for(j=0; j<i; ++j)
        {
            if(distances[o + order[j]] < delta[order[i]])
            {
                delta[order[i]] = distances[o + order[j]];
                neighbour[order[i]] = order[j];
            }
        }
        max_delta = (max_delta < delta[order[i]]) ? delta[order[i]] : max_delta;
    }
    delta[order[0]] = max_delta;
}

extern void _get_membership(
    int *clusters, int nclusters, int *order, int *neighbour, int npoints, int *membership)
{
    int i;
    for(i=0; i<npoints; ++i)
        membership[i] = -1;
    for(i=0; i<nclusters; ++i)
        membership[clusters[i]] = i;
    for(i=0; i<npoints; ++i)
    {
        if(membership[order[i]] == -1)
            membership[order[i]] = membership[neighbour[order[i]]];
    }
}

extern void _get_border(
    double kernel_size, double *distances, double *density, int *membership, int npoints,
    int *border_member, double *border_density)
{
    int i, j, o;
    double average_density;
    for(i=0; i<npoints-1; ++i)
    {
        o = i * npoints;
        for(j=i+1; j<npoints; ++j)
        {
            if((membership[i] != membership[j]) && (distances[o + j] < kernel_size))
            {
                average_density = 0.5 * (density[i] + density[j]);
                if(border_density[membership[i]] < average_density)
                    border_density[membership[i]] = average_density;
                if(border_density[membership[j]] < average_density)
                    border_density[membership[j]] = average_density;
                border_member[i] = 1;
                border_member[j] = 1;
            }
        }
    }
}

extern void _get_halo(
    int border_only, double *border_density,
    double *density, int *membership, int *border_member, int npoints, int *halo)
{
    int i;
    for(i=0; i<npoints; ++i)
    {
        if(density[i] < border_density[membership[i]])
        {
            if((0 == border_only) || (1 == border_member[i]))
                halo[i] = -1;
        }
    }
}

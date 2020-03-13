#include <iostream>
#include <algorithm>
#include "numflow.hpp"

using namespace std;

void index(const FTYPE value, const FTYPE * grid, const INTTYPE size, marker & mark)
{
    INTTYPE low = 0;
    INTTYPE high = size - 1;
    FTYPE min = grid[low];
    FTYPE max = grid[high];
    INTTYPE middle = high * (value - min) / (max - min);

    if (value < min || value > max)
    {
        mark.status = LookUp::outOfRange;
        return;
    }

    // try to predict the index on uniform
    if( value >= grid[middle] && value <= grid[middle + 1])
        low = middle, high = middle + 1 ;
    else {
        // if not guessed, perform binary search
        // has to have more than one layer!!
        middle = (high - low) / 2;
        while(high - low != 1)
        {
            if(value < grid[middle])
                high = middle;
            else
                low = middle;
            middle =  low + (high - low) / 2 ; 
        }
    }
    
    mark.i = low;
    mark.fac = (value - grid[low]) / (grid[high] - grid[low]);
    mark.status = LookUp::ok;
}


FTYPE * interpolate_3d(const dataset3D * dataset, FTYPE *points, const INTTYPE count)
{
    FTYPE * values = new FTYPE[count * 3]{};

    INTTYPE zy = dataset->dy * dataset->dz;
    INTTYPE zyx0, zyx1, zy0, zy1, zy0ind2, zy1ind2;
    FTYPE c00[3], c01[3], c10[3], c11[3], c0[3], c1[3];
    marker x, y, z;

    for (INTTYPE j = 0; j < count; j++)
    {
        //get indicies
        index(points[j * 3    ], dataset->ax, dataset->dx, x);
        if(x.status != LookUp::ok)
            continue;
        index(points[j * 3 + 1], dataset->ay, dataset->dy, y);
        if(y.status != LookUp::ok)
            continue;
        index(points[j * 3 + 2], dataset->az, dataset->dz, z);
        if(z.status != LookUp::ok)
            continue;


        //interpolate values
        zyx0 = zy * x.i * 3;
        zyx1 = zy * (x.i + 1) * 3;
        zy0 = dataset->dz * y.i * 3;
        zy1 = dataset->dz * (y.i  + 1) * 3;
        zy0ind2 = zy0 + z.i  * 3;
        zy1ind2 = zy1 + z.i * 3;

        for (INTTYPE i = 0; i < 3; ++i)
        {
            c00[i] = dataset->data[zyx0 + zy0ind2 + i] * (1.0 - x.fac) + 
                     dataset->data[zyx1 + zy0ind2 + i] * x.fac;
            c01[i] = dataset->data[zyx0 + zy0ind2 + 3 + i] * (1.0 - x.fac) + 
                     dataset->data[zyx1 + zy0ind2 + 3 + i] * x.fac;
            c10[i] = dataset->data[zyx0 + zy1ind2 + i] * (1.0 - x.fac) + 
                     dataset->data[zyx1 + zy1ind2 + i] * x.fac;
            c11[i] = dataset->data[zyx0 + zy1ind2 + 3 + i] * (1.0 - x.fac) + 
                     dataset->data[zyx1 + zy1ind2 + 3 + i] * x.fac;

            c0[i] = c00[i] * (1.0 - y.fac) + c10[i] * y.fac;
            c1[i] = c01[i] * (1.0 - y.fac) + c11[i] * y.fac;

            values[j * 3 + i] = c0[i] * (1.0 - z.fac) + c1[i] * z.fac;
        }
    }

    return values;
}

//-------------------------------------------------------------------------------

bool compare_datasets3D(const dataset_line3D &i1, const dataset_line3D &i2)
{
    if (i1.x == i2.x)
    {
        if (i1.y == i2.y)
            return i1.z < i2.z;
        else
            return i1.y < i2.y;
    }
    else
        return i1.x < i2.x;
}

bool compare_datasets2D(const dataset_line2D &i1, const dataset_line2D &i2)
{
    if (i1.x == i2.x)
        return i1.y < i2.y;
    else
        return i1.x < i2.x;
}

void dataset_sort(FTYPE *data, INTTYPE columns, INTTYPE rows)
{
    if (columns == 6)
        sort((dataset_line3D *) data, (dataset_line3D *) data + rows, compare_datasets3D);
    else if (columns == 4)
        sort((dataset_line2D *) data, (dataset_line2D *) data + rows, compare_datasets2D);
}

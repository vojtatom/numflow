#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <string>

#include "numflow.hpp"
#include "buffer.hpp"

using namespace std;

//-----------------------------------------------------------------------------------

struct dataset_line3D
{
    double vx, vy, vz;
    double x, y, z;
};

struct dataset_line2D
{
    double vx, vy;
    double x, y;
};

struct marker
{
    int32_t i;
    double fac;
    int32_t status;
};

enum LookUp
{
    ok,
    outOfRange,
};

//-----------------------------------------------------------------------------------

void index(const double value, const double *grid, const int32_t size, marker &mark)
{
    int32_t low = 0;
    int32_t high = size - 1;
    double min = grid[low];
    double max = grid[high];
    int32_t middle = high * (value - min) / (max - min);

    if (value < min || value > max)
    {
        mark.status = LookUp::outOfRange;
        return;
    }

    // try to predict the index on uniform
    if (value >= grid[middle] && value <= grid[middle + 1])
        low = middle, high = middle + 1;
    else
    {
        // if not guessed, perform binary search
        // has to have more than one layer!!
        middle = (high - low) / 2;
        while (high - low != 1)
        {
            if (value < grid[middle])
                high = middle;
            else
                low = middle;
            middle = low + (high - low) / 2;
        }
    }

    mark.i = low;
    mark.fac = (value - grid[low]) / (grid[high] - grid[low]);
    mark.status = LookUp::ok;
}

double *interpolate_3d(const Dataset3D *dataset, double *points, const int32_t count)
{
    double *values = new double[count * 3]{};

    int32_t zy = dataset->dy * dataset->dz;
    int32_t zyx0, zyx1, zy0, zy1, zy0ind2, zy1ind2;
    double c00[3], c01[3], c10[3], c11[3], c0[3], c1[3];
    marker x, y, z;

    for (int32_t j = 0; j < count; j++)
    {
        //get indicies
        index(points[j * 3], dataset->ax, dataset->dx, x);
        if (x.status != LookUp::ok)
            continue;
        index(points[j * 3 + 1], dataset->ay, dataset->dy, y);
        if (y.status != LookUp::ok)
            continue;
        index(points[j * 3 + 2], dataset->az, dataset->dz, z);
        if (z.status != LookUp::ok)
            continue;

        //interpolate values
        zyx0 = zy * x.i * 3;
        zyx1 = zy * (x.i + 1) * 3;
        zy0 = dataset->dz * y.i * 3;
        zy1 = dataset->dz * (y.i + 1) * 3;
        zy0ind2 = zy0 + z.i * 3;
        zy1ind2 = zy1 + z.i * 3;

        for (int32_t i = 0; i < 3; ++i)
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

bool compare_datasets_3d(const dataset_line3D &i1, const dataset_line3D &i2)
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

bool compare_datasets_2d(const dataset_line2D &i1, const dataset_line2D &i2)
{
    if (i1.x == i2.x)
        return i1.y < i2.y;
    else
        return i1.x < i2.x;
}

//sort matrix rows according to 3-6th or 2-4th columns
void dataset_sort(double *data, int32_t columns, int32_t rows)
{
    if (columns == 6)
        sort((dataset_line3D *)data, (dataset_line3D *)data + rows, compare_datasets_3d);
    else if (columns == 4)
        sort((dataset_line2D *)data, (dataset_line2D *)data + rows, compare_datasets_2d);
}

//try to add uniques to buffer of unique values, values remain sorted
void add_unique(Buffer<double> &uniques, Buffer<size_t> &counts, double value, double epsilon)
{
    size_t idx = uniques.lower_bound(value);
    //cout << value << " " << idx << endl;

    assert(idx < 700);

    if (idx == uniques.filled || abs(uniques[idx] - value) > epsilon)
    {
        //cout << " inserting " << idx << endl;
        uniques.insert(idx, value), counts.insert(idx, 1);
        //cout << " done " << endl;
    }
    else{
        counts[idx] += 1;
        //cout << " adding " << endl;

    }
}

//gets unique coordinates of all axis
//also test for rectilinear grid
bool is_rectilinear_3d(double *values, size_t count, double epsilon,
                       Buffer<double> &ux, Buffer<double> &uy, Buffer<double> &uz)
{
    Buffer<size_t> xcount, ycount, zcount;

    size_t idx = 3;
    while (idx < count)
    {
        add_unique(ux, xcount, values[idx++], epsilon);
        add_unique(uy, ycount, values[idx++], epsilon);
        add_unique(uz, zcount, values[idx++], epsilon);
        idx += 3;
        //cout << idx << endl;
    }

    size_t coutns = xcount[0];
    for (size_t i = 1; i < xcount.filled; i++)
        if (xcount[i] != coutns)
            return false;

    coutns = ycount[0];
    for (size_t i = 1; i < ycount.filled; i++)
        if (ycount[i] != coutns)
            return false;

    coutns = zcount[0];
    for (size_t i = 1; i < zcount.filled; i++)
        if (zcount[i] != coutns)
            return false;

    return true;
}

//load 3D dataset
Dataset3D *load_rectilinear_3d(const DataMatrix *mat, double epsilon)
{
    Buffer<double> ux, uy, uz;
    size_t size = mat->columns * mat->rows;

    if (!is_rectilinear_3d(mat->data, size, epsilon, ux, uy, uz))
        return nullptr;

    dataset_sort(mat->data, 6, mat->rows);

    //create dataset
    Dataset3D *dataset = new Dataset3D();
    dataset->dx = ux.filled;
    dataset->dy = uy.filled;
    dataset->dz = uz.filled;

    dataset->ax = ux.data;
    dataset->ay = uy.data;
    dataset->az = uz.data;

    ux.data = nullptr, ux.filled = 0, ux.allocated = 0;
    uy.data = nullptr, uy.filled = 0, uy.allocated = 0;
    uz.data = nullptr, uz.filled = 0, uz.allocated = 0;

    //dataset->data = mat->data; // not necessary here
    return dataset;
}

//parse .csv file
DataMatrix *parse_file(const char *filename, const char *sep)
{
    ifstream file(filename);
    size_t dims = 0, clines = 0;
    string line;
    unsigned char s1, s2, s3, s4, s5;
    float x, y, z, vx, vy, vz;
    Buffer<double> values;

    //number of dimensions
    getline(file, line);
    for (const char c : line)
        if (c == *sep)
            dims++;

    file.seekg(0, file.beg);
    while (getline(file, line))
        clines++;

    file.clear();
    file.seekg(0, file.beg);

    //3d dataset
    if (dims == 5)
    {
        values.reserve(6 * clines);
        while (!file.eof() && file >> vx >> s1 >> vy >> s2 >> vz >> s3 >> x >> s4 >> y >> s5 >> z)
        {
            /*if (!(s1 == *sep && s2 == *sep && s3 == *sep 
                  && s4 == *sep && s5 == *sep))
                {
                    cout << s1 << " " << s2 << " " << s3 << " " << s4 << " " << s5 << endl;
                    return nullptr;
                }*/

            values.push_back(vx, vy, vz, x, y, z);
        }
    }
    //2d dataset
    else if (dims == 3)
    {
        return nullptr; //not implemented
    }
    else
        return nullptr; //unknown

    //now all data is in memory
    if (!file.eof())
        return nullptr;
    file.close();

    DataMatrix *mat = new DataMatrix();
    mat->data = values.data;
    mat->columns = dims + 1;
    mat->rows = values.filled / mat->columns;
    values.data = nullptr, values.filled = 0, values.allocated = 0;
    return mat;
}

void delete_dataset_3d(Dataset3D * ds)
{
    delete ds;
}

void delete_datamatrix(DataMatrix * dm)
{
    delete dm;
}

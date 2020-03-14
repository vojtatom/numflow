#pragma once

struct Dataset3D
{
    int32_t dx, dy, dz;
    double *ax, *ay, *az;
    double *data;
};

struct DataMatrix
{
    int32_t rows, columns;
    double * data;
};

//loading functions
Dataset3D * load_rectilinear_3d(const DataMatrix * mat, double epsilon);
DataMatrix * parse_file(const char * filename, const char * sep);

//interpolation
double * interpolate_3d(const Dataset3D * dataset, double *points, const int32_t count);

void delete_dataset_3d(Dataset3D * ds);
void delete_datamatrix(DataMatrix * dm);
#pragma once

#define FTYPE double
#define INTTYPE int32_t

struct dataset_line3D
{
    FTYPE vx, vy, vz;
    FTYPE x, y, z;
};

struct dataset_line2D
{
    FTYPE vx, vy;
    FTYPE x, y;
};

struct dataset3D
{
    INTTYPE dx, dy, dz;
    FTYPE *ax, *ay, *az;
    FTYPE *data;
};

struct marker
{
    INTTYPE i;
    FTYPE fac;
    INTTYPE status;
};

enum LookUp {
    ok,
    outOfRange,
};

void dataset_sort(FTYPE *data, INTTYPE columns, INTTYPE rows);
FTYPE * interpolate_3d(const dataset3D * dataset, FTYPE *points, const INTTYPE count);


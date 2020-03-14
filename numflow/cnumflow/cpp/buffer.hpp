#pragma once
#pragma once
#include <cstdlib>
#include <cstring>
#include <iostream>

#define INITBUFFERSIZE 1024
#define THRESHOLDBUFFERSIZE 25000000

template <typename T>
struct Buffer
{
    Buffer()
    {
        //data = (T *)std::malloc(sizeof(T) * INITBUFFERSIZE);
        //std::memset(data, 0, sizeof(T) * INITBUFFERSIZE);
        data = new T[INITBUFFERSIZE];
        filled = 0;
        allocated = INITBUFFERSIZE;
    }

    ~Buffer()
    {
        if (data != nullptr)
            //std::free(data);
            delete [] data;
    }

    T &operator[](size_t idx)
    {
        return data[idx];
    }

    const T &operator[](size_t idx) const
    {
        return data[idx];
    }

    void reserve(const size_t size)
    {
        std::free(data);
        //data = (T *)std::malloc(sizeof(T) * size);
        data = new T[size];

        /*if (data == nullptr)
        {
            std::cerr << size << std::endl;
            exit(1);
        }*/

        //std::memset(data, 0, sizeof(T) * size);
        allocated = size;
        filled = 0;
    }

    void push_back(const T v)
    {
        if (filled == allocated)
            realloc();

        data[filled++] = v;
    }

    void push_back(const T v1, const T v2)
    {
        if (filled >= allocated - 1)
            realloc();

        data[filled++] = v1;
        data[filled++] = v2;
    }

    void push_back(const T v1, const T v2, const T v3)
    {
        if (filled >= allocated - 2)
            realloc();

        data[filled++] = v1;
        data[filled++] = v2;
        data[filled++] = v3;
    }

    void push_back(const T v1, const T v2, const T v3, const T v4, const T v5, const T v6)
    {
        if (filled >= allocated - 5)
            realloc();

        data[filled++] = v1;
        data[filled++] = v2;
        data[filled++] = v3;
        data[filled++] = v4;
        data[filled++] = v5;
        data[filled++] = v6;
    }

    void insert(const size_t idx, const T v)
    {
        if (filled == allocated)
            realloc();

        for (size_t i = filled; i > idx; --i)
            data[i] = data[i - 1];

        data[idx] = v;
        filled++;
    }

    size_t lower_bound(const T v)
    {
        if (filled == 0)
            return 0;

        size_t low = 0;
        size_t high = filled - 1;
        T min = data[0];
        T max = data[high];

        if (v < min)
            return 0;

        if (v > max)
            return filled;

        if (high == low)
            return low;

        size_t middle = high * (v - min) / (max - min);

        middle = (high - low) / 2;
        while (high - low != 1)
        {
            if (v < data[middle])
                high = middle;
            else
                low = middle;
            middle = low + (high - low) / 2;
        }
        
        if (v <= data[low])
            return low;
        return high;
        
        //fac = (value - data[low]) / (data[high] - data[low]);
    }

    void realloc()
    {
        T *tmp = data;

        /*try
        {*/
        if (allocated > THRESHOLDBUFFERSIZE)
        {
            //data = (T * ) std::realloc(data, sizeof(T) * (allocated + THRESHOLDBUFFERSIZE));
            data = new T[allocated + THRESHOLDBUFFERSIZE];
            allocated += THRESHOLDBUFFERSIZE;
        }
        else
        {
            //data = (T * ) std::realloc(data, sizeof(T) * (allocated + THRESHOLDBUFFERSIZE));
            data = new T[allocated * 2];
            allocated *= 2;
        }

        /*if (data == nullptr)
        {
            std::cerr << "reall" << allocated << std::endl;
            exit(1);
        }*/

        for (size_t i = 0; i < filled; i++)
            data[i] = tmp[i];

        //if (tmp != data)
        //std::free(tmp);

        delete[] tmp;

        /*}
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << allocated << " " << allocated + THRESHOLDBUFFERSIZE  << std::endl;
            exit(1);
        }*/
    }

    void rob()
    {
        data = nullptr;
        filled = 0;
        allocated = 0;
    }

    T *data;
    size_t filled;
    size_t allocated;
};

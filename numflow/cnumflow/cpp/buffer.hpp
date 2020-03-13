#pragma once
#pragma once
#include <cstdlib>

#define INITBUFFERSIZE  1024



template <typename T>
struct Buffer {
    Buffer() {
        data = new T[INITBUFFERSIZE];
        filled = 0;
        allocated = INITBUFFERSIZE;
    }

    ~Buffer()
    {
        delete [] data;
    }

    void push_back(const T v)
    {
        if (filled == allocated)
            realloc();

        data[filled++] = v;
    }

    void realloc() {
        T * tmp = data;
        data = new T[allocated * 2];
        allocated *= 2;

        for (size_t i = 0; i < filled; i++)
            data[i] = tmp[i];

        delete [] tmp;
    }


    T * data;
    size_t filled;
    size_t allocated;
};


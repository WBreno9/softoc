#ifndef _PPM_H
#define _PPM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

//Fits in a 32bit register
typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} pixel_t;

typedef struct {
    uint8_t* data;

    size_t w;
    size_t h;
} image_t;

image_t* create_image(uint32_t w, uint32_t h);

void free_image(image_t* t);

void write_pixel(image_t* t, pixel_t p, uint32_t x, uint32_t y);

void clear_image(image_t* t, pixel_t p);

int write_image(image_t* t, const char* file_path);

#endif //_PPM_H

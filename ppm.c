#include "ppm.h"

image_t* create_image(uint32_t w, uint32_t h) {
    image_t* t = (image_t*)malloc(sizeof(image_t));
    t->w = w;
    t->h = h;

        t->data = (uint8_t*)malloc(w*h*3);

    return t;
}

void free_image(image_t* t) {
    free(t->data);
    free(t);
}

void write_pixel(image_t* t, pixel_t p, uint32_t x, uint32_t y) {
    uint32_t index = 3*(x + y * t->w);

    t->data[index    ] = p.r;
    t->data[index + 1] = p.g;
    t->data[index + 2] = p.b;
}

void clear_image(image_t* t, pixel_t p) {
    for (int x = 0; x < t->w; x++) {
        for (int y = 0; y < t->h; y++) {
            write_pixel(t, p, x, y);
        }
    }
}

int write_image(image_t* t, const char* file_path) {
    FILE* file = fopen(file_path, "w");

    if (!file) {
        fprintf(stderr, "ERROR: %s\n", strerror(errno));
        return 0;
    }

    //PPM image
    //Byte (P6) header
    fprintf(file, "P6\n %lu %lu\n 255\n", t->w, t->h);
    fwrite(t->data, 1, t->w * t->h * 3, file);

    return 1;
}

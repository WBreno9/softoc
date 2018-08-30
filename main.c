#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <smmintrin.h>

//Fits in a 32bit register
typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} pixel_t;

typedef struct {
    pixel_t* pixels;

    uint32_t w;
    uint32_t h;
} image_t;

image_t* create_image(uint32_t w, uint32_t h) {
    image_t* t = (image_t*)malloc(sizeof(image_t));
    t->w = w;
    t->h = h;

    t->pixels = (pixel_t*)malloc(sizeof(pixel_t)*w*h);

    return t;
}

void free_image(image_t* t) {
    free(t->pixels);
    free(t);
}

void write_pixel(image_t* t, pixel_t p, uint32_t x, uint32_t y) {
    uint32_t index = x + y * t->w;

    t->pixels[index] = p;
}

void clear_image(image_t* t, pixel_t p) {
    for (int i = 0; i < t->w * t->h; i++) {
        t->pixels[i] = p;
    }
}

int write_image(image_t* t, const char* file_path) {
    FILE* file = fopen(file_path, "w");

    if (!file) {
        fprintf(stderr, "ERROR: %s\n", strerror(errno));
        return 0;
    }

    //Byte (P6) header
    fprintf(file, "P6\n %d %d\n 255\n", t->w, t->h);
    fwrite(t->pixels, sizeof(pixel_t), t->w * t->h, file);

    return 1;
}

typedef __m128 vec4;

#define vec4_set                _mm_setr_ps
#define vec4_get                _mm_store_ps
#define vec4_add                _mm_add_ps
#define vec4_sub                _mm_sub_ps
#define vec4_mul                _mm_mul_ps
#define vec4_div                _mm_mul_ps
#define vec4_dot                _mm_dp_ps
#define vec4_xor                _mm_xor_ps

#define VEC2_DOT                0x30
#define VEC3_DOT                0x70
#define VEC4_DOT                0xF0

#define VEC_STORE_0             0x01
#define VEC_STORE_1             0x02
#define VEC_STORE_2             0x04
#define VEC_STORE_3             0x08
#define VEC_STORE_ALL           0x0F

static inline void vec4_print(vec4 a) {
    printf("vec4 = [%.2f, %.2f, %.2f, %.2f]\n", a[0], a[1], a[2], a[3]);
}

typedef vec4 mat4[4];

typedef struct {
    vec4 pos;
    vec4 forward;
    vec4 up;

    float fov;
} camera_t;

static inline void mat4_print(mat4 m) {
    printf("mat4:\n");
    for (int i = 0; i < 4; i++) {
        printf("        ");
        vec4_print(m[i]);
    }
}

static inline void mat4_transpose(mat4 a, mat4 b) {
    b[0] = vec4_set(a[0][0], a[1][0], a[2][0], a[3][0]);
    b[1] = vec4_set(a[0][1], a[1][1], a[2][1], a[3][1]);
    b[2] = vec4_set(a[0][2], a[1][2], a[2][2], a[3][2]);
    b[3] = vec4_set(a[0][3], a[1][3], a[2][3], a[3][3]);
}

static inline void mat4_cpy(mat4 a, mat4 b) {
    for (int i = 0; i < 4; i++) 
        b[i] = a[i];
}

static inline void mat4_mul(mat4 a, mat4 b, mat4 c) {
    mat4 t;
    mat4_transpose(b, t);

    for (int i = 0; i < 4; i++) {
        c[i] = vec4_xor(vec4_xor(vec4_dot(a[i], t[0], VEC4_DOT | VEC_STORE_0),
                                 vec4_dot(a[i], t[1], VEC4_DOT | VEC_STORE_1)),
                        vec4_xor(vec4_dot(a[i], t[2], VEC4_DOT | VEC_STORE_2),
                                 vec4_dot(a[i], t[3], VEC4_DOT | VEC_STORE_3)));
    }
}

static inline vec4 vec4_mul_mat4(vec4 a, mat4 b) {
}

int main() {

    mat4 m;

    m[0] = vec4_set(1.0f, 0.0f, 0.0f, 0.0f);
    m[1] = vec4_set(0.0f, 1.0f, 0.0f, 0.0f);
    m[2] = vec4_set(0.0f, 0.0f, 1.0f, 0.0f);
    m[3] = vec4_set(0.0f, 0.0f, 0.0f, 1.0f);

    mat4 m0;

    m0[0] = vec4_set(2.0f, 1.0f, 0.0f, 0.0f);
    m0[1] = vec4_set(0.0f, 1.0f, 0.0f, 0.0f);
    m0[2] = vec4_set(0.0f, 0.0f, 2.0f, 0.0f);
    m0[3] = vec4_set(0.0f, 0.0f, 0.0f, 1.0f);

    mat4_mul(m, m0, m);
    mat4_print(m);
    return 0;
}

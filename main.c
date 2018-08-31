#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>

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

typedef __m128 vec4;

#define vec4_set                _mm_setr_ps
#define vec4_get                _mm_store_ps
#define vec4_add                _mm_add_ps
#define vec4_sub                _mm_sub_ps
#define vec4_mul                _mm_mul_ps
#define vec4_div                _mm_div_ps
#define vec4_dot                _mm_dp_ps
#define vec4_xor                _mm_xor_ps
#define vec4_crs                _vec4_crs
#define vec4_nrm                _vec4_nrm
#define vec4_lng                _vec4_lng
#define vec4_sum                _mm_hadds_ps

#define VEC2_DOT                0x30
#define VEC3_DOT                0x70
#define VEC4_DOT                0xF0

#define VEC_STORE_0             0x01
#define VEC_STORE_1             0x02
#define VEC_STORE_2             0x04
#define VEC_STORE_3             0x08
#define VEC_STORE_ALL           0x0F

static inline void vec4_print(vec4 a) {
    printf("vec4 = [%.4f, %.4f, %.4f, %.4f]\n", a[0], a[1], a[2], a[3]);
}

static inline vec4 _vec4_crs(vec4 a, vec4 b) {
    return vec4_sub(vec4_mul(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)),
                             _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
                    vec4_mul(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)),
                             _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))));
}

/*TODO*/
//Ugly, unecessary AND lacks optimization
static inline vec4 _vec4_lng(vec4 a, const short m) {
    vec4 dst;

    vec4 tmp = vec4_mul(a, a);
    float len = sqrt(tmp[0] + tmp[1] + tmp[2]);

    for (int i = 0; i < 4; i++) {
        if (m & VEC_STORE_0) dst[0] = len;
        if (m & VEC_STORE_1) dst[1] = len;
        if (m & VEC_STORE_2) dst[2] = len;
        if (m & VEC_STORE_3) dst[3] = len;
    }

    return dst;
}

static inline vec4 _vec4_nrm(vec4 a) {
    return vec4_div(a, vec4_lng(a, VEC_STORE_ALL));
}

typedef vec4 mat4[4];

static inline void mat4_print(mat4 m) {
    printf("mat4:\n");
    for (int i = 0; i < 4; i++) {
        printf("        ");
        vec4_print(m[i]);
    }
}

/*TODO*/
//This can be a macro
static inline void mat4_transpose(mat4 a) {
    _MM_TRANSPOSE4_PS(a[0], a[1], a[2], a[3]);
}

static inline void mat4_cpy(mat4 a, mat4 b) {
    for (int i = 0; i < 4; i++) b[i] = a[i];
}

static inline vec4 vec4_mul_mat4(vec4 a, mat4 b) {
    mat4 t;
    mat4_cpy(b, t);
    mat4_transpose(t);

    return vec4_xor(vec4_xor(vec4_dot(a, t[0], VEC4_DOT | VEC_STORE_0),
                             vec4_dot(a, t[1], VEC4_DOT | VEC_STORE_1)),
                    vec4_xor(vec4_dot(a, t[2], VEC4_DOT | VEC_STORE_2),
                             vec4_dot(a, t[3], VEC4_DOT | VEC_STORE_3)));
}

//Row major order matrix multiplication
static inline void mat4_mul(mat4 a, mat4 b, mat4 c) {
    mat4 t;
    mat4_cpy(a, t);
    mat4_transpose(t);

    for (int i = 0; i < 4; i++) {
        c[i] = vec4_xor(vec4_xor(vec4_dot(t[0], b[i], VEC4_DOT | VEC_STORE_0),
                                 vec4_dot(t[1], b[i], VEC4_DOT | VEC_STORE_1)),
                        vec4_xor(vec4_dot(t[2], b[i], VEC4_DOT | VEC_STORE_2),
                                 vec4_dot(t[3], b[i], VEC4_DOT | VEC_STORE_3)));
    }
}

typedef struct {
    vec4 pos;
    vec4 forward;
    vec4 up;
    vec4 right;

    float fov;
    //Maybe add padding
} camera_t;

void update_camera(camera_t* cam, vec4 up) {
    //Can this be optimized?
    cam->right = vec4_nrm(vec4_crs(up          , cam->forward));
    cam->up    =          vec4_crs(cam->forward, cam->right);
    cam->right =          vec4_crs(cam->up     , cam->forward);
}

//Row major order matrix
void get_view(camera_t* cam, mat4 m) {
    m[0] = cam->right;
    m[1] = cam->up;
    m[2] = cam->forward;
    //Maybe use a shuffle
    m[3] = vec4_set(-cam->pos[0], -cam->pos[1], -cam->pos[2], 1.f);
}

void look_at(camera_t* cam) {
}

//Row major order matrix
void perspective(mat4 m, float fovy, float aspect, float n, float f) {
    float h = tan(fovy/2.f);

    m[0] = vec4_set(1.f/(aspect*h), 0.f   , 0.f             ,  0.f);
    m[1] = vec4_set(0.f           , 1.f/h , 0.f             ,  0.f);
    m[2] = vec4_set(0.f           , 0.f   , -(f+n)/(f-n)    , -1.f);
    m[3] = vec4_set(0.f           , 0.f   , -(2.f*f*n)/(f-n),  0.f);
}

typedef float vec3[3];

static inline void vec3_to_vec4(vec4 v4, vec3 v3, float w) {
    v4 = vec4_set(v3[0], v3[1], v4[2], w);
}

typedef float rect[2];

typedef struct {
    uint8_t* data;
    size_t w;
    size_t h;
}buffer_t;

void scanline_raster_triangle(vec4 v0, vec4 v1, vec4 v2, buffer_t* buff) {
        
}

#define IMAGE_WIDTH  512
#define IMAGE_HEIGHT 512
int main() {
    image_t* img = create_image(IMAGE_WIDTH, IMAGE_HEIGHT);
    memset(img->data, 0, img->w*img->h*3);

    buffer_t buff;
    buff.data = img->data;
    buff.w    = img->w;
    buff.h    = img->h;

    vec4 v0 = vec4_set( 0.f, 1.f, 0.f, 1.f);
    vec4 v1 = vec4_set(-1.f, 0.f, 0.f, 1.f);
    vec4 v2 = vec4_set( 1.f, 0.f, 0.f, 1.f);

    scanline_raster_triangle(v0, v1, v2, &buff);

    write_image(img, "test.ppm");
    free_image(img);
    return 0;
}

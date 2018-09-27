#ifndef _VMAT_H
#define _VMAT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>

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

static inline void vec4_print(vec4 a) 
{
    printf("vec4 = [%.4f, %.4f, %.4f, %.4f]\n", a[0], a[1], a[2], a[3]);
}

static inline vec4 _vec4_crs(vec4 a, vec4 b) 
{
    return vec4_sub(vec4_mul(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)),
                             _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
                    vec4_mul(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)),
                             _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))));
}

/*TODO*/
//Ugly, unecessary AND lacks optimization
static inline vec4 _vec4_lng(vec4 a, const short m) 
{
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

static inline void mat4_print(mat4 m) 
{
    printf("mat4:\n");

    for (int i = 0; i < 4; i++) {
        printf("        ");
        vec4_print(m[i]);
    }
}

/*TODO*/
//This can be a macro
static inline void mat4_transpose(mat4 a) 
{
    _MM_TRANSPOSE4_PS(a[0], a[1], a[2], a[3]);
}

static inline void mat4_cpy(mat4 a, mat4 b) 
{
    for (int i = 0; i < 4; i++) b[i] = a[i];
}

static inline vec4 vec4_mul_mat4(vec4 a, mat4 b) 
{
    mat4 t;
    mat4_cpy(b, t);
    mat4_transpose(t);

    return vec4_xor(vec4_xor(vec4_dot(a, t[0], VEC4_DOT | VEC_STORE_0),
                             vec4_dot(a, t[1], VEC4_DOT | VEC_STORE_1)),
                    vec4_xor(vec4_dot(a, t[2], VEC4_DOT | VEC_STORE_2),
                             vec4_dot(a, t[3], VEC4_DOT | VEC_STORE_3)));
}

//Row major order matrix multiplication
static inline void mat4_mul(mat4 a, mat4 b, mat4 c) 
{
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

void look_at(vec4 pos, vec4 center, vec4 up, mat4 m) 
{
    vec4 f = vec4_nrm(vec4_sub(center, pos));
    vec4 s = vec4_nrm(vec4_crs(f, up));
    vec4 u = vec4_crs(s, f);

    m[0] = s;
    m[1] = u;
    m[2] = vec4_sub(_mm_set1_ps(0.f), f);
    m[3] = vec4_xor(vec4_xor(vec4_dot(u, pos, VEC3_DOT | VEC_STORE_0),
                             vec4_dot(s, pos, VEC3_DOT | VEC_STORE_1)),
                             vec4_dot(f, pos, VEC3_DOT | VEC_STORE_2));
    m[3][3] = 1.f;
}

//Row major order matrix
static inline void perspective(mat4 m, float fovy, float aspect, float n, float f) 
{
    float h = tan(fovy/2.f);

    m[0] = vec4_set(1.f/(aspect*h), 0.f   , 0.f             ,  0.f);
    m[1] = vec4_set(0.f           , 1.f/h , 0.f             ,  0.f);
    m[2] = vec4_set(0.f           , 0.f   , -(f+n)/(f-n)    , -1.f);
    m[3] = vec4_set(0.f           , 0.f   , -(2.f*f*n)/(f-n),  0.f);
}

//Row major order matrix
static inline void screen_space(mat4 m, float w, float h) 
{
    m[0] = vec4_set((w-1)/2.f, 0.f             , 0.f, 0.f);
    m[1] = vec4_set(0.f      , (-1.f*(h-1))/2.f, 0.f, 0.f);
    m[2] = vec4_set(0.f      , 0.f             , 1.f, 0.f);
    m[3] = vec4_set((w-1)/2.f, (h-1)/2.f       , 0.f, 1.f);
}

#endif //_VMAT_H

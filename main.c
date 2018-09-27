#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "vmat.h"
#include "ppm.h"

struct buffer_container {
    float *data;
    size_t w;
    size_t h;
};

#define FP_BITS 12
#define FP_MASK ((1 << FP_BITS) - 1)

typedef int32_t fp32_t;

#define FP_FLOAT(x) ((x) * (float)(1 << FP_BITS))
#define FP_TO_FLOAT(x) ((float)(x) / (float)(1 << FP_BITS))
#define FP_MUL(a, b) ((((a)>>8)*((b)>>8))>>0)

#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)

void raster_triangle(fp32_t *x, fp32_t *y, float *z, struct buffer_container *buff) 
{
    uint32_t max_x = MIN(MAX(x[0], MAX(x[1], x[2])), buff->w << FP_BITS) >> FP_BITS;
    uint32_t max_y = MIN(MAX(y[0], MAX(y[1], y[2])), buff->w << FP_BITS) >> FP_BITS;

    uint32_t min_x = MAX(MIN(x[0], MIN(x[1], x[2])), 0) >> FP_BITS;
    uint32_t min_y = MAX(MIN(y[0], MIN(y[1], y[2])), 0) >> FP_BITS;

    fp32_t A01 = y[0] - y[1];
    fp32_t A12 = y[1] - y[2];
    fp32_t A20 = y[2] - y[0];

    fp32_t B01 = x[1] - x[0];
    fp32_t B12 = x[2] - x[1];
    fp32_t B20 = x[0] - x[2];

    fp32_t w0_row = FP_MUL((min_x - x[1]),  (y[2] - y[1])) - FP_MUL((min_y - y[1]), (x[2] - x[1]));
    fp32_t w1_row = FP_MUL((min_x - x[2]),  (y[0] - y[2])) - FP_MUL((min_y - y[2]), (x[0] - x[2]));
    fp32_t w2_row = FP_MUL((min_x - x[0]),  (y[1] - y[0])) - FP_MUL((min_y - y[0]), (x[1] - x[0]));

    /*
    float depth;
    float area = (x[2] - x[0]) * (y[1] - y[0]) - (y[2] - y[0]) * (x[1] - x[0]);
    */

    float xx[3] = {x[0] >> FP_BITS, x[1] >> FP_BITS, x[2] >> FP_BITS};
    float yy[3] = {y[0] >> FP_BITS, y[1] >> FP_BITS, y[2] >> FP_BITS};

    for (uint32_t py = min_y; py < max_y; py++) {
        fp32_t w0 = w0_row;
        fp32_t w1 = w1_row;
        fp32_t w2 = w2_row;

        for (uint32_t px = min_x; px < max_y; px++) {
            fp32_t pxx = px << FP_BITS;
            fp32_t pyy = py << FP_BITS;

            fp32_t w0 = FP_MUL((pxx - x[1]),  (y[2] - y[1])) - FP_MUL((pyy - y[1]), (x[2] - x[1]));
            fp32_t w1 = FP_MUL((pxx - x[2]),  (y[0] - y[2])) - FP_MUL((pyy - y[2]), (x[0] - x[2]));
            fp32_t w2 = FP_MUL((pxx - x[0]),  (y[1] - y[0])) - FP_MUL((pyy - y[0]), (x[1] - x[0]));

            if (w0 > 0 && w1 > 0 && w2 > 0) {
                /*
                w0 /= area;
                w1 /= area;
                w2 /= area;

                depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2];
                depth = fabs(depth);

                //Demcompress Z
                //depth = 1.f/depth;

                uint32_t index = x + y * buff->w;


                if (buff->data[index] > depth)
                    buff->data[index] = depth;
                    */

                uint32_t index = px + py * buff->w;
                buff->data[index] = 1.0f;
            }

            w0 += A12;
            w1 += A20;
            w2 += A01;
        }

        w0_row += B12;
        w1_row += B20;
        w2_row += B01;
    }
}

void raster_triangle_array(vec4 *vecs, size_t size, struct buffer_container *buff)
{
    mat4 ss;
    screen_space(ss, buff->w, buff->h);

    for (size_t i = 0; i < size; i++) {
        fp32_t x[3];
        fp32_t y[3];
        float  z[3];

        vec4 v0 = vec4_mul_mat4(vecs[i*3  ], ss);
        vec4 v1 = vec4_mul_mat4(vecs[i*3+1], ss);
        vec4 v2 = vec4_mul_mat4(vecs[i*3+2], ss);

        x[0] = FP_FLOAT(v0[0] / v0[3]);
        x[1] = FP_FLOAT(v1[0] / v1[3]);
        x[2] = FP_FLOAT(v2[0] / v2[3]);

        y[0] = FP_FLOAT(v0[1] / v0[3]);
        y[1] = FP_FLOAT(v1[1] / v1[3]);
        y[2] = FP_FLOAT(v2[1] / v2[3]);

        z[0] = v0[2] / v0[3];
        z[1] = v1[2] / v1[3];
        z[2] = v2[2] / v2[3];

        raster_triangle(x, y, z, buff);
    }
}

void render(struct buffer_container *buff)
{
    vec4 v0 = vec4_set( 0.f,  1.f, 1.f, 1.f);
    vec4 v1 = vec4_set(-1.f, -1.f, 1.f, 1.f);
    vec4 v2 = vec4_set( 1.f, -1.f, 1.f, 1.f);

    vec4 vs[3] = {v0, v1, v2};

    /*
    mat4 v;
    look_at(vec4_set(0.f, 1.f, 2.f, 1.f),
            vec4_set(0.f, 0.f, 0.f, 0.f),
            vec4_set(0.f, -1.f, 0.f, 0.f), v);

    mat4 p;
    perspective(p, 90.f, 1.33f, 0.1f, 100.f);

    mat4 mvp;

    mat4_mul(v, p, mvp);

    vs[0] = vec4_mul_mat4(v0, mvp);
    vs[1] = vec4_mul_mat4(v1, mvp);
    vs[2] = vec4_mul_mat4(v2, mvp);
    */

    raster_triangle_array(vs, 1, buff);
}

int main()
{
    struct buffer_container buff;
    buff.w = 512;
    buff.h = 512;
    buff.data = malloc(sizeof(float)*buff.w*buff.h);

    for (size_t i = 0; i < buff.h*buff.w; i++) {
        buff.data[i] = .1f;
    }

    render(&buff);
    
    /*
    fp32_t a = FP_FLOAT(3.f);
    fp32_t b = FP_FLOAT(6.8f);

    fp32_t c = ((uint64_t)a * (uint64_t)b) >> FP_BITS;
    */

    /*
    int32_t a = 3.0f *  (1 << FP_BITS);
    int32_t b = -1.5f * (1 << FP_BITS);

    uint64_t c = ((uint64_t)a * (uint64_t)b) >> FP_BITS;

    int32_t mask_2 = 0xffffffff >> (32-FP_BITS);

    printf("%i %i\n", c >> FP_BITS, c & mask_2);
    */

    GLFWwindow *wnd = NULL;

    glfwInit();

    glfwWindowHint(GLFW_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    wnd = glfwCreateWindow(buff.w, buff.h, "Z Buffer", NULL, NULL);

    glfwMakeContextCurrent(wnd);
    glfwSwapInterval(0);

    glClearColor(1.f, 1.f, 1.f, 1.f);

    float delta_time;
    float curr_time = 0.f;
    float past_time = 0.f;


    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glEnable(GL_TEXTURE_2D);

    while (!glfwWindowShouldClose(wnd)) {
        glfwPollEvents();

        curr_time = (float)glfwGetTime();
        delta_time = past_time - curr_time;
        past_time = curr_time;

        glClear(GL_COLOR_BUFFER_BIT);

        render(&buff);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, buff.w, buff.h, 0, GL_RED, GL_FLOAT, buff.data);

        glBegin(GL_TRIANGLES);
            glTexCoord2f(0.f, 1.f);
            glVertex3f(-1.f, -1.f, 0.f);
            glTexCoord2f(1.f, 0.f);
            glVertex3f( 1.f,  1.f, 0.f);
            glTexCoord2f(0.f, 0.f);
            glVertex3f(-1.f,  1.f, 0.f);

            glTexCoord2f(0.f, 1.f);
            glVertex3f(-1.f, -1.f, 0.f);
            glTexCoord2f(1.f, 1.f);
            glVertex3f( 1.f, -1.f, 0.f);
            glTexCoord2f(1.f, 0.f);
            glVertex3f( 1.f,  1.f, 0.f);
        glEnd();

        if (glfwGetKey(wnd, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(wnd, GLFW_TRUE);

        glfwSwapBuffers(wnd);
    }

    glDeleteTextures(1, &tex);

    glfwTerminate();

    free(buff.data);
}

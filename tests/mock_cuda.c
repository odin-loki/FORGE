/* A CPU stand-in for libcuda.so.1, used only by tests: it implements the driver entry points FORGE's
   CUDA runtime (grid/cuda.hpp) calls and "launches" each kernel by running a C version of it over the
   argument array exactly as cuLaunchKernel would pass it. This exercises the dlopen path, symbol set,
   argument packing, grid sizing and memcpy directions without a GPU. It does not execute the PTX. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t dptr;
static float* P(dptr p) { return (float*)(uintptr_t)p; }
int cuInit(unsigned f) { (void)f; return 0; }
int cuDeviceGetCount(int* n) { *n = 1; return 0; }
int cuDeviceGetName(char* b, int n, int d) { (void)d; strncpy(b, "forge mock cuda", (size_t)n); return 0; }
int cuCtxCreate_v2(void** c, unsigned f, int d) { (void)f; (void)d; *c = (void*)1; return 0; }
int cuCtxDestroy_v2(void* c) { (void)c; return 0; }
int cuModuleLoadData(void** m, const void* img) { if (!strstr((const char*)img, ".entry forge_")) return 200; *m = (void*)2; return 0; }
int cuModuleUnload(void* m) { (void)m; return 0; }
int cuModuleGetFunction(void** f, void* m, const char* name) { (void)m; *f = strdup(name); return 0; }
int cuMemAlloc_v2(dptr* p, size_t n) { *p = (dptr)(uintptr_t)malloc(n); return *p ? 0 : 2; }
int cuMemFree_v2(dptr p) { free(P(p)); return 0; }
int cuMemcpyHtoD_v2(dptr d, const void* s, size_t n) { memcpy(P(d), s, n); return 0; }
int cuMemcpyDtoH_v2(void* d, dptr s, size_t n) { memcpy(d, P(s), n); return 0; }
int cuCtxSynchronize(void) { return 0; }

static float un(const char* op, float x) {
    if (!strcmp(op, "neg")) return -x; if (!strcmp(op, "abs")) return fabsf(x); if (!strcmp(op, "sqrt")) return sqrtf(x);
    if (!strcmp(op, "rsqrt")) return 1.0f / sqrtf(x); if (!strcmp(op, "recip")) return 1.0f / x; if (!strcmp(op, "square")) return x * x;
    if (!strcmp(op, "exp")) return expf(x); if (!strcmp(op, "exp2")) return exp2f(x); if (!strcmp(op, "log")) return logf(x);
    if (!strcmp(op, "log2")) return log2f(x); if (!strcmp(op, "tanh")) return tanhf(x); if (!strcmp(op, "sigmoid")) return 1.0f / (1.0f + expf(-x));
    if (!strcmp(op, "silu")) return x / (1.0f + expf(-x)); if (!strcmp(op, "relu")) return x > 0 ? x : 0;
    if (!strcmp(op, "softplus")) return fmaxf(x, 0) + log1pf(expf(-fabsf(x)));
    if (!strcmp(op, "gelu_tanh")) return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    return NAN;
}
static float bin(const char* op, float a, float b) {
    if (!strcmp(op, "add")) return a + b; if (!strcmp(op, "sub")) return a - b; if (!strcmp(op, "mul")) return a * b;
    if (!strcmp(op, "div")) return a / b; if (!strcmp(op, "maximum")) return a > b ? a : b; if (!strcmp(op, "minimum")) return a < b ? a : b;
    if (!strcmp(op, "silu_mul")) return a / (1.0f + expf(-a)) * b;
    return NAN;
}
#define ARG(i, T) (*(T*)args[i])
int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned sh,
                   void* stream, void** args, void** extra) {
    (void)gy; (void)gz; (void)by; (void)bz; (void)sh; (void)stream; (void)extra;
    const char* op = (const char*)f + 6;   /* skip "forge_" */
    unsigned threads = gx * bx, i;
    if (!strcmp(op, "scale")) { float *x = P(ARG(0, dptr)), *y = P(ARG(1, dptr)); unsigned n = ARG(2, unsigned); float a = ARG(3, float);
        for (i = 0; i < threads && i < n; ++i) y[i] = a * x[i]; return 0; }
    if (!strcmp(op, "softmax")) { float *x = P(ARG(0, dptr)), *y = P(ARG(1, dptr)); unsigned rows = ARG(2, unsigned), cols = ARG(3, unsigned);
        for (unsigned r = 0; r < rows && r < threads; ++r) { float m = x[r * cols], s = 0; unsigned j;
            for (j = 1; j < cols; ++j) m = fmaxf(m, x[r * cols + j]); for (j = 0; j < cols; ++j) s += expf(x[r * cols + j] - m);
            for (j = 0; j < cols; ++j) y[r * cols + j] = expf(x[r * cols + j] - m) / s; } return 0; }
    if (!strcmp(op, "rms_norm")) { float *x = P(ARG(0, dptr)), *w = P(ARG(1, dptr)), *y = P(ARG(2, dptr));
        unsigned rows = ARG(3, unsigned), cols = ARG(4, unsigned); float eps = ARG(5, float);
        for (unsigned r = 0; r < rows && r < threads; ++r) { float s = 0; unsigned j; for (j = 0; j < cols; ++j) s += x[r * cols + j] * x[r * cols + j];
            float inv = 1.0f / sqrtf(s / (float)cols + eps); for (j = 0; j < cols; ++j) y[r * cols + j] = x[r * cols + j] * inv * w[j]; } return 0; }
    if (!strcmp(op, "matmul_t")) { float *a = P(ARG(0, dptr)), *b = P(ARG(1, dptr)), *y = P(ARG(2, dptr));
        unsigned rows = ARG(3, unsigned), cols = ARG(4, unsigned), k = ARG(5, unsigned);
        for (i = 0; i < rows * cols && i < threads; ++i) { float s = 0; for (unsigned t = 0; t < k; ++t) s = fmaf(a[(i / cols) * k + t], b[(i % cols) * k + t], s); y[i] = s; } return 0; }
    if (!isnan(bin(op, 1, 1)) || !strcmp(op, "div")) { float *a = P(ARG(0, dptr)), *b = P(ARG(1, dptr)), *y = P(ARG(2, dptr)); unsigned n = ARG(3, unsigned);
        for (i = 0; i < threads && i < n; ++i) y[i] = bin(op, a[i], b[i]); return 0; }
    { float *x = P(ARG(0, dptr)), *y = P(ARG(1, dptr)); unsigned n = ARG(2, unsigned);
      for (i = 0; i < threads && i < n; ++i) y[i] = un(op, x[i]); return 0; }
}

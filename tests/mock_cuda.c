/* A CPU stand-in for libcuda.so.1 (and, built with -DMOCK_HIP, libamdhip64.so.6), used only by tests: it
   implements the driver entry points FORGE's CUDA/HIP runtime (grid/cuda.hpp) calls and "launches" each
   kernel by running a C version of it over the argument array exactly as cuLaunchKernel would pass it. This
   exercises the dlopen path, symbol set, argument packing, grid sizing, streams, residency copies, graph
   capture/replay and multi-device without a GPU. It does not execute the PTX / code object.
   MOCK_DEVICES (env) sets the device count (default 1). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MOCK_HIP
#define N_(cu, hip) hip
#else
#define N_(cu, hip) cu
#endif
typedef uint64_t dptr;
static float* P(dptr p) { return (float*)(uintptr_t)p; }
static int g_cur = 0;
int N_(cuInit, hipInit)(unsigned f) { (void)f; return 0; }
int N_(cuDeviceGetCount, hipGetDeviceCount)(int* n) { const char* e = getenv("MOCK_DEVICES"); *n = e ? atoi(e) : 1; return 0; }
int N_(cuDeviceGetName, hipDeviceGetName)(char* b, int n, int d) { snprintf(b, (size_t)n, "forge mock %s %d", N_("cuda", "hip"), d); return 0; }
#ifndef MOCK_HIP
int cuDeviceGet(int* d, int o) { *d = o; return 0; }
int cuCtxCreate_v2(void** c, unsigned f, int d) { (void)f; *c = (void*)(uintptr_t)(d + 1); return 0; }
int cuCtxDestroy_v2(void* c) { (void)c; return 0; }
int cuCtxSetCurrent(void* c) { g_cur = (int)(uintptr_t)c - 1; return 0; }
#else
int hipSetDevice(int d) { g_cur = d; return 0; }
#endif
int N_(cuModuleLoadData, hipModuleLoadData)(void** m, const void* img) {
#ifdef MOCK_HIP
    (void)img;
#else
    if (!strstr((const char*)img, ".entry forge_")) return 200;
#endif
    *m = (void*)2; return 0; }
int N_(cuModuleUnload, hipModuleUnload)(void* m) { (void)m; return 0; }
int N_(cuModuleGetFunction, hipModuleGetFunction)(void** f, void* m, const char* name) { (void)m; *f = strdup(name); return 0; }
static size_t g_live = 0;
int N_(cuMemAlloc_v2, hipMalloc)(dptr* p, size_t n) { size_t* b = malloc(n + 16); if (!b) return 2; b[0] = n; g_live += n; *p = (dptr)(uintptr_t)(b + 2); return 0; }
int N_(cuMemFree_v2, hipFree)(dptr p) { if (!p) return 0; size_t* b = (size_t*)(uintptr_t)p - 2; g_live -= b[0]; free(b); return 0; }
int N_(cuMemcpyHtoD_v2, hipMemcpyHtoD)(dptr d, const void* s, size_t n) { memcpy(P(d), s, n); return 0; }
int N_(cuMemcpyDtoH_v2, hipMemcpyDtoH)(void* d, dptr s, size_t n) { memcpy(d, P(s), n); return 0; }
int N_(cuMemcpyHtoDAsync_v2, hipMemcpyHtoDAsync)(dptr d, const void* s, size_t n, void* st) { (void)st; memcpy(P(d), s, n); return 0; }
int N_(cuMemcpyDtoHAsync_v2, hipMemcpyDtoHAsync)(void* d, dptr s, size_t n, void* st) { (void)st; memcpy(d, P(s), n); return 0; }
int N_(cuMemcpyDtoDAsync_v2, hipMemcpyDtoDAsync)(dptr d, dptr s, size_t n, void* st) { (void)st; memmove(P(d), P(s), n); return 0; }
int N_(cuMemcpyPeer, hipMemcpyPeer)(dptr d, uintptr_t dc, dptr s, uintptr_t sc, size_t n) { (void)dc; (void)sc; memcpy(P(d), P(s), n); return 0; }
int N_(cuMemGetInfo_v2, hipMemGetInfo)(size_t* fr, size_t* tot) { *tot = (size_t)1 << 30; *fr = *tot - g_live; return 0; }
int N_(cuCtxSynchronize, hipDeviceSynchronize)(void) { return 0; }
int N_(cuStreamCreate, hipStreamCreateWithFlags)(void** s, unsigned f) { (void)f; *s = malloc(1); return 0; }
int N_(cuStreamDestroy_v2, hipStreamDestroy)(void* s) { free(s); return 0; }
int N_(cuStreamSynchronize, hipStreamSynchronize)(void* s) { (void)s; return 0; }

/* graph capture: launches between begin/end are recorded (with copies of their argument values) */
#define MAXARGS 16
typedef struct { void* f; unsigned gx, bx; int nargs; uint64_t vals[MAXARGS]; } rec;
typedef struct { int n; rec* r; } graph;
static int g_capturing = 0; static graph* g_cap = NULL;
int N_(cuStreamBeginCapture_v2, hipStreamBeginCapture)(void* s, int mode) { (void)s; (void)mode; g_capturing = 1; g_cap = calloc(1, sizeof(graph)); return 0; }
int N_(cuStreamEndCapture, hipStreamEndCapture)(void* s, void** g) { (void)s; g_capturing = 0; *g = g_cap; g_cap = NULL; return 0; }
int N_(cuGraphInstantiateWithFlags, hipGraphInstantiateWithFlags)(void** e, void* g, unsigned long long f) { (void)f; *e = g; return 0; }
int N_(cuGraphExecDestroy, hipGraphExecDestroy)(void* e) { (void)e; return 0; }
int N_(cuGraphDestroy, hipGraphDestroy)(void* g) { graph* gr = g; free(gr->r); free(gr); return 0; }
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
static float fp8dec(unsigned b) { unsigned e = (b >> 3) & 15u, m = b & 7u; float v = e == 0 ? ldexpf((float)m, -9) : ldexpf((float)(8u + m), (int)e - 10); return (b & 0x80u) ? -v : v; }
static float kvat(const float* base, unsigned codec, unsigned e) {
    if (!codec) return base[e];
    uint32_t w; memcpy(&w, &base[1 + (e >> 2)], 4); unsigned b = (w >> (8u * (e & 3u))) & 255u;
    return (codec == 1 ? (float)(signed char)b : fp8dec(b)) * base[0]; }
/* forge_qmatmul_t (grid/cuda.hpp run_qmatmul): same decode as the kernel / flow/codecs.hpp */
static float m_h2f(unsigned h) {
    unsigned s = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u, b; float f;
    if (e == 0u) { if (m == 0u) b = s; else { e = 113u; while (!(m & 1024u)) { m <<= 1; --e; } m &= 1023u; b = s | (e << 23) | (m << 13); } }
    else if (e == 31u) b = s | 0x7f800000u | (m << 13);
    else b = s | ((e + 112u) << 23) | (m << 13);
    memcpy(&f, &b, 4); return f; }
static unsigned m_ld16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
static void m_scmin(unsigned j, const unsigned char* q, unsigned* d, unsigned* m) {
    if (j < 4u) { *d = q[j] & 63u; *m = q[j + 4u] & 63u; }
    else { *d = (q[j + 4u] & 15u) | ((q[j - 4u] >> 6) << 4); *m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4); } }
static void m_qgroup(unsigned fmt, const unsigned char* row, unsigned g, float* v) {
    unsigned l, s = g & 7u;
    if (fmt == 1u) { const unsigned char* b = row + g * 34u; float d = m_h2f(m_ld16(b)); for (l = 0; l < 32u; ++l) v[l] = d * (float)(signed char)b[2u + l]; return; }
    if (fmt == 3u) { const unsigned char* b = row + (g >> 3) * 210u; unsigned h = s >> 2, t = s & 3u;
        const unsigned char* ql = b + 64u * h; const unsigned char* qh = b + 128u + 32u * h; const signed char* sc = (const signed char*)(b + 192u + 8u * h);
        float d = m_h2f(m_ld16(b + 208u));
        for (l = 0; l < 32u; ++l) { unsigned lo = (t & 1u) ? ql[l + 32u] : ql[l]; int q; lo = (t & 2u) ? (lo >> 4) : (lo & 15u);
            q = (int)(lo | (((qh[l] >> (2u * t)) & 3u) << 4)) - 32; v[l] = d * (float)sc[(l >> 4) + 2u * t] * (float)q; }
        return; }
    { const unsigned char* b = row + (g >> 3) * (fmt == 2u ? 144u : 176u);
      float d = m_h2f(m_ld16(b)), dm = m_h2f(m_ld16(b + 2u)), d1, m1; unsigned sc, mn, hi = s & 1u; m_scmin(s, b + 4u, &sc, &mn);
      d1 = d * (float)sc; m1 = dm * (float)mn;
      if (fmt == 2u) { const unsigned char* q = b + 16u + (s >> 1) * 32u; for (l = 0; l < 32u; ++l) v[l] = d1 * (float)(hi ? (q[l] >> 4) : (q[l] & 15u)) - m1; }
      else { const unsigned char* qh = b + 16u; const unsigned char* q = b + 48u + (s >> 1) * 32u; unsigned u = 1u << s;
          for (l = 0; l < 32u; ++l) v[l] = d1 * (float)((hi ? (q[l] >> 4) : (q[l] & 15u)) + ((qh[l] & u) ? 16u : 0u)) - m1; } } }
static int mock_qmm(const float* a, const unsigned char* w, float* y, unsigned rows, unsigned cols, unsigned k, unsigned fmt, unsigned rb) {
    unsigned r, c, g, l; float v[32];
    for (r = 0; r < rows; ++r) for (c = 0; c < cols; ++c) { float s = 0;
        for (g = 0; g < k / 32u; ++g) { m_qgroup(fmt, w + (size_t)c * rb, g, v); for (l = 0; l < 32u; ++l) s = fmaf(a[(size_t)r * k + g * 32u + l], v[l], s); }
        y[(size_t)r * cols + c] = s; }
    return 0; }
static int nargs_of(const char* op) {
    if (!strcmp(op, "qmatmul_t") || !strcmp(op, "qmatmul_mmq")) return 8;
    if (!strcmp(op, "delta_rule")) return 14; if (!strcmp(op, "causal_conv")) return 7;
    if (!strcmp(op, "attention") || !strcmp(op, "attention_fa")) return 12; if (!strcmp(op, "rope")) return 8; if (!strcmp(op, "slice")) return 7;
    if (!strcmp(op, "gather")) return 6; if (!strcmp(op, "rms_norm") || !strcmp(op, "matmul_t")) return 6;
    if (!strcmp(op, "scale") || !strcmp(op, "softmax")) return 4;
    return 0; }
static int run_kernel(void* f, unsigned gx, unsigned bx, void** args);
int N_(cuLaunchKernel, hipModuleLaunchKernel)(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned sh,
                   void* stream, void** args, void** extra) {
    (void)gy; (void)gz; (void)by; (void)bz; (void)sh; (void)stream; (void)extra;
    if (g_capturing) {
        const char* op = (const char*)f + 6; int n = nargs_of(op);
        if (!n) n = (!isnan(un(op, 1)) && isnan(bin(op, 1, 1))) ? 3 : 4;
        g_cap->r = realloc(g_cap->r, sizeof(rec) * (size_t)(g_cap->n + 1));
        rec* r = &g_cap->r[g_cap->n++]; r->f = f; r->gx = gx; r->bx = bx; r->nargs = n;
        for (int i = 0; i < n; ++i) memcpy(&r->vals[i], args[i], 8);   /* every arg is <= 8 bytes */
        return 0;
    }
    return run_kernel(f, gx, bx, args);
}
int N_(cuGraphLaunch, hipGraphLaunch)(void* e, void* s) {
    (void)s; graph* g = e;
    for (int k = 0; k < g->n; ++k) { void* a[MAXARGS]; for (int i = 0; i < g->r[k].nargs; ++i) a[i] = &g->r[k].vals[i]; run_kernel(g->r[k].f, g->r[k].gx, g->r[k].bx, a); }
    return 0; }
static int run_kernel(void* f, unsigned gx, unsigned bx, void** args) {
    const char* op = (const char*)f + 6;   /* skip "forge_" */
    unsigned threads = gx * bx, i;
    if (!strcmp(op, "rope")) { float *x = P(ARG(0, dptr)), *pos = P(ARG(1, dptr)), *inv = P(ARG(2, dptr)), *y = P(ARG(3, dptr));
        unsigned rows = ARG(4, unsigned), cols = ARG(5, unsigned), d = ARG(6, unsigned), neox = ARG(7, unsigned), half = d / 2;
        for (i = 0; i < rows * cols * half && i < threads; ++i) { unsigned j = i % half, row = i / half, s = row / cols, o = row * d;
            double th = (double)pos[s] * (double)inv[j]; float c = (float)cos(th), sn = (float)sin(th);
            unsigned i0 = neox ? j : 2 * j, i1 = neox ? j + half : 2 * j + 1; float x0 = x[o + i0], x1 = x[o + i1];
            y[o + i0] = x0 * c - x1 * sn; y[o + i1] = x0 * sn + x1 * c; } return 0; }
    if (!strcmp(op, "gather")) { float *t = P(ARG(0, dptr)), *idx = P(ARG(1, dptr)), *y = P(ARG(2, dptr)); unsigned n = ARG(3, unsigned), rows = ARG(4, unsigned), cols = ARG(5, unsigned);
        for (i = 0; i < n && i < threads; ++i) { unsigned r = (unsigned)idx[i / cols]; y[i] = r < rows ? t[r * cols + i % cols] : 0; } return 0; }
    if (!strcmp(op, "slice")) { float *x = P(ARG(0, dptr)), *y = P(ARG(1, dptr)); unsigned n = ARG(2, unsigned), lin = ARG(3, unsigned), inner = ARG(4, unsigned), st = ARG(5, unsigned), len = ARG(6, unsigned);
        for (i = 0; i < n && i < threads; ++i) { unsigned in_ = i % inner, r = (i / inner) % len, o = i / (inner * len); y[i] = x[(o * lin + st + r) * inner + in_]; } return 0; }
    if (!strcmp(op, "attention") || !strcmp(op, "attention_fa")) { float *q = P(ARG(0, dptr)), *k = P(ARG(1, dptr)), *v = P(ARG(2, dptr)), *y = P(ARG(3, dptr));
        unsigned s = ARG(4, unsigned), hq = ARG(5, unsigned), d = ARG(6, unsigned), t = ARG(7, unsigned), hk = ARG(8, unsigned), dv = ARG(9, unsigned), fl = ARG(10, unsigned);
        float sc = ARG(11, float);
        unsigned causal = fl & 1, kc = (fl >> 1) & 3, vc = (fl >> 3) & 3, window = fl >> 5;
        unsigned kw = kc ? 1 + (d + 3) / 4 : d, vw = vc ? 1 + (dv + 3) / 4 : dv;
        for (unsigned id = 0; id < s * hq && id < threads; ++id) {
            unsigned qi = id / hq, h = id % hq, kh = h / (hq / hk); int pos = (int)t - (int)s + (int)qi;
            float m = -INFINITY, sum = 0; float* yp = y + id * dv; const float* qp = q + id * d; unsigned j, e;
            for (j = 0; j < t; ++j) { if (causal && (int)j > pos) continue; if (window && pos - (int)j >= (int)window) continue;
                float dot = 0; for (e = 0; e < d; ++e) dot = fmaf(qp[e], kvat(k + (j * hk + kh) * kw, kc, e), dot); if (dot * sc > m) m = dot * sc; }
            for (e = 0; e < dv; ++e) yp[e] = 0;
            for (j = 0; j < t; ++j) { if (causal && (int)j > pos) continue; if (window && pos - (int)j >= (int)window) continue;
                float dot = 0; for (e = 0; e < d; ++e) dot = fmaf(qp[e], kvat(k + (j * hk + kh) * kw, kc, e), dot);
                float w = expf(dot * sc - m); sum += w; for (e = 0; e < dv; ++e) yp[e] = fmaf(w, kvat(v + (j * hk + kh) * vw, vc, e), yp[e]); }
            for (e = 0; e < dv; ++e) yp[e] *= sum > 0 ? 1 / sum : 0; }
        return 0; }
    if (!strcmp(op, "qmatmul_t")) { float *a = P(ARG(0, dptr)), *y = P(ARG(2, dptr)); const unsigned char* w = (const unsigned char*)(uintptr_t)ARG(1, dptr);
        unsigned rows = ARG(3, unsigned), cols = ARG(4, unsigned), k = ARG(5, unsigned), fmt = ARG(6, unsigned), rb = ARG(7, unsigned);
        if (gx < cols * ((rows + 3) / 4) || bx != 128) return 1;   /* run_qmatmul's launch shape */
        return mock_qmm(a, w, y, rows, cols, k, fmt, rb); }
    if (!strcmp(op, "qmatmul_mmq")) { float *a = P(ARG(0, dptr)), *y = P(ARG(2, dptr)); const unsigned char* w = (const unsigned char*)(uintptr_t)ARG(1, dptr);
        unsigned rows = ARG(3, unsigned), cols = ARG(4, unsigned), k = ARG(5, unsigned), fmt = ARG(6, unsigned), rb = ARG(7, unsigned);
        if (gx < ((cols + 63) / 64) * ((rows + 63) / 64) || bx != 256) return 1;   /* MMQ 64x64-tile launch shape */
        return mock_qmm(a, w, y, rows, cols, k, fmt, rb); }
    if (!strcmp(op, "delta_rule")) {   /* flow/state_ops.hpp delta_rule; one block per value head */
        float *q = P(ARG(0, dptr)), *k = P(ARG(1, dptr)), *v = P(ARG(2, dptr)), *g = P(ARG(3, dptr)), *be = P(ARG(4, dptr)), *S = P(ARG(5, dptr)), *o = P(ARG(6, dptr));
        unsigned s = ARG(7, unsigned), hk = ARG(8, unsigned), dk = ARG(9, unsigned), hv = ARG(10, unsigned), dv = ARG(11, unsigned), l2 = ARG(12, unsigned); float sc = ARG(13, float);
        unsigned h, t, a, j; if (gx < hv || bx != 128 || dk > 256) return 1;
        for (h = 0; h < hv; ++h) { float* Sh = S + (size_t)h * dk * dv; unsigned kh = h / (hv / hk);
            for (t = 0; t < s; ++t) { const float *qt = q + ((size_t)t * hk + kh) * dk, *kt = k + ((size_t)t * hk + kh) * dk, *vt = v + ((size_t)t * hv + h) * dv;
                float qs = 0, ks = 0, qf, kf, dec = expf(g[t * hv + h]), bt = be[t * hv + h]; float* ot = o + ((size_t)t * hv + h) * dv;
                for (a = 0; a < dk; ++a) { qs += qt[a] * qt[a]; ks += kt[a] * kt[a]; }
                qf = l2 ? sc / sqrtf(qs + 1e-6f) : sc; kf = l2 ? 1.0f / sqrtf(ks + 1e-6f) : 1.0f;
                for (j = 0; j < dv; ++j) { float kv = 0, d, acc = 0;
                    for (a = 0; a < dk; ++a) { Sh[a * dv + j] *= dec; kv = fmaf(Sh[a * dv + j], kt[a] * kf, kv); }
                    d = (vt[j] - kv) * bt;
                    for (a = 0; a < dk; ++a) { Sh[a * dv + j] = fmaf(kt[a] * kf, d, Sh[a * dv + j]); acc = fmaf(Sh[a * dv + j], qt[a] * qf, acc); }
                    ot[j] = acc; } } }
        return 0; }
    if (!strcmp(op, "causal_conv")) {
        float *x = P(ARG(0, dptr)), *w = P(ARG(1, dptr)), *st = P(ARG(2, dptr)), *o = P(ARG(3, dptr));
        unsigned s = ARG(4, unsigned), C = ARG(5, unsigned), K = ARG(6, unsigned), c, t, j, P_ = K - 1; float nw[8];
        if (gx * bx < C || K > 9) return 1;
        for (c = 0; c < C; ++c) {
            for (t = 0; t < s; ++t) { float acc = 0; for (j = 0; j < K; ++j) { unsigned p = t + j; acc = fmaf(w[c * K + j], p < P_ ? st[p * C + c] : x[(size_t)(p - P_) * C + c], acc); } o[(size_t)t * C + c] = acc; }
            for (j = 0; j < P_; ++j) { unsigned p = s + j; nw[j] = p < P_ ? st[p * C + c] : x[(size_t)(p - P_) * C + c]; }
            for (j = 0; j < P_; ++j) st[j * C + c] = nw[j]; }
        return 0; }
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

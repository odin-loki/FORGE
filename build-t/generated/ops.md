# FORGE op reference (generated)

62 ops.

| op | class | inputs | shape rule | tolerance | fusion | cost | attrs | semantics |
|---|---|---|---|---|---|---|---|---|
| `matmul` | contraction | 2 | matmul | rel:1e-4 | epilogue | 2 |  | batched a[..., m, k] @ b[k, n] or b[..., k, n] |
| `matmul_t` | contraction | 2 | matmul_t | rel:1e-4 | epilogue | 2 |  | a[..., m, k] @ b[n, k]^T (weight layout of HF linear layers) |
| `attention` | contraction | 3 | attention | abs:1e-5 | no | 4 | causal,groups,scale,window | naive scaled-dot-product attention; q[s,hq,d] k[t,hk,d] v[t,hk,d] -> [s,hq,d]; GQA via groups |
| `add` | elementwise | 2 | broadcast | exact | yes | 1 |  | y = a + b (numpy broadcasting) |
| `sub` | elementwise | 2 | broadcast | exact | yes | 1 |  | y = a - b |
| `mul` | elementwise | 2 | broadcast | exact | yes | 1 |  | y = a * b |
| `div` | elementwise | 2 | broadcast | ulp:1 | yes | 4 |  | y = a / b |
| `maximum` | elementwise | 2 | broadcast | exact | yes | 1 |  | y = max(a, b) |
| `minimum` | elementwise | 2 | broadcast | exact | yes | 1 |  | y = min(a, b) |
| `pow` | elementwise | 2 | broadcast | rel:1e-5 | yes | 20 |  | y = a ^ b |
| `fma` | elementwise | 3 | broadcast | exact | yes | 2 |  | y = a * b + c (single rounding) |
| `select` | elementwise | 3 | broadcast | exact | yes | 1 |  | y = c != 0 ? a : b  (inputs: a, b, cond) |
| `neg` | elementwise | 1 | same | exact | yes | 1 |  | y = -x |
| `abs` | elementwise | 1 | same | exact | yes | 1 |  | y = |x| |
| `sqrt` | elementwise | 1 | same | ulp:1 | yes | 4 |  | y = sqrt(x) |
| `rsqrt` | elementwise | 1 | same | rel:1e-6 | yes | 4 |  | y = 1 / sqrt(x) |
| `recip` | elementwise | 1 | same | ulp:1 | yes | 4 |  | y = 1 / x |
| `square` | elementwise | 1 | same | exact | yes | 1 |  | y = x * x |
| `exp` | elementwise | 1 | same | rel:1e-6 | yes | 10 |  | y = e^x |
| `exp2` | elementwise | 1 | same | rel:1e-6 | yes | 10 |  | y = 2^x |
| `expm1` | elementwise | 1 | same | rel:1e-6 | yes | 10 |  | y = e^x - 1 |
| `log` | elementwise | 1 | same | rel:1e-6 | yes | 10 |  | y = ln x |
| `log2` | elementwise | 1 | same | rel:1e-6 | yes | 10 |  | y = log2 x |
| `log1p` | elementwise | 1 | same | rel:1e-6 | yes | 10 |  | y = ln(1 + x) |
| `sin` | elementwise | 1 | same | abs:1e-6 | yes | 12 |  | y = sin x |
| `cos` | elementwise | 1 | same | abs:1e-6 | yes | 12 |  | y = cos x |
| `tan` | elementwise | 1 | same | rel:1e-5 | yes | 14 |  | y = tan x |
| `tanh` | elementwise | 1 | same | abs:1e-6 | yes | 12 |  | y = tanh x |
| `sinh` | elementwise | 1 | same | rel:1e-5 | yes | 12 |  | y = sinh x |
| `cosh` | elementwise | 1 | same | rel:1e-5 | yes | 12 |  | y = cosh x |
| `asin` | elementwise | 1 | same | abs:1e-6 | yes | 14 |  | y = asin x |
| `acos` | elementwise | 1 | same | abs:1e-6 | yes | 14 |  | y = acos x |
| `atan` | elementwise | 1 | same | abs:1e-6 | yes | 14 |  | y = atan x |
| `erf` | elementwise | 1 | same | abs:1e-6 | yes | 14 |  | y = erf x |
| `erfc` | elementwise | 1 | same | abs:1e-6 | yes | 14 |  | y = erfc x |
| `sigmoid` | elementwise | 1 | same | abs:1e-6 | yes | 12 |  | y = 1 / (1 + e^-x) |
| `silu` | elementwise | 1 | same | abs:1e-5 | yes | 12 |  | y = x * sigmoid(x) |
| `gelu` | elementwise | 1 | same | abs:1e-5 | yes | 14 |  | y = 0.5 x (1 + erf(x / sqrt 2)) |
| `gelu_tanh` | elementwise | 1 | same | abs:1e-5 | yes | 14 |  | tanh approximation of gelu |
| `relu` | elementwise | 1 | same | exact | yes | 1 |  | y = max(x, 0) |
| `softplus` | elementwise | 1 | same | abs:1e-5 | yes | 14 |  | y = ln(1 + e^x) |
| `scale` | elementwise | 1 | same | exact | yes | 1 | alpha | y = alpha * x |
| `convert` | elementwise | 1 | same | exact | yes | 1 | to | round-trip x through dtype `to` (f16/bf16/fp8_e4m3/fp8_e5m2) |
| `transpose` | data_movement | 1 | transpose | exact | no | 0 | perm | permute axes |
| `reshape` | data_movement | 1 | reshape | exact | no | 0 | shape | reshape (one -1 allowed) |
| `gather` | data_movement | 2 | gather | exact | no | 0 |  | rows of table[n, d] selected by idx[...] -> [..., d] (embedding lookup) |
| `concat` | data_movement | 2 | concat | exact | no | 0 | axis | concatenate along axis |
| `slice` | data_movement | 1 | slice | exact | no | 0 | axis,start,end | x[start:end] along axis |
| `rope` | data_movement | 2 | same | abs:1e-5 | no | 6 | base,neox | rotary embedding of x[s, h, d] at positions pos[s] |
| `silu_mul` | elementwise | 2 | broadcast | abs:1e-5 | yes | 13 |  | silu(a) * b (fused SwiGLU gate) |
| `expand` | data_movement | 1 | expand | exact | no | 0 | axis,n | repeat a size-1 axis n times (broadcast_to along one axis) |
| `reduce_sum` | reduction | 1 | reduce | rel:1e-5 | no | 1 | axis | sum over axis (keepdims=false) |
| `reduce_max` | reduction | 1 | reduce | exact | no | 1 | axis | max over axis |
| `reduce_min` | reduction | 1 | reduce | exact | no | 1 | axis | min over axis |
| `reduce_mean` | reduction | 1 | reduce | rel:1e-5 | no | 1 | axis | mean over axis |
| `argmax` | reduction | 1 | reduce | exact | no | 1 | axis | index of max over axis (as float) |
| `logsumexp` | reduction | 1 | reduce | rel:1e-5 | no | 12 | axis | log(sum(exp(x))) over axis, stable |
| `softmax` | reduction | 1 | same | abs:1e-6 | no | 14 |  | softmax over last axis |
| `rms_norm` | reduction | 2 | same | rel:1e-5 | no | 4 | eps | x / sqrt(mean(x^2) + eps) * w over last axis |
| `layer_norm` | reduction | 3 | same | rel:1e-5 | no | 6 | eps | (x - mean) / sqrt(var + eps) * w + b over last axis |
| `cumsum` | reduction | 1 | same | rel:1e-5 | no | 1 | axis | inclusive prefix sum over axis |
| `moe_gate` | reduction | 1 | same | abs:1e-6 | no | 16 | k,norm | MoE router: softmax over last axis, keep top-k (ties -> lower index), renormalise if norm |

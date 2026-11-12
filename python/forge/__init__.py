"""FORGE Python bindings (ctypes over the stable C ABI in libforge; docs/bindings.md, docs/capi.md).

    import forge
    with forge.Model("model.fpk") as m:
        print(m.complete("Hello", max_tokens=16))
        for piece in m.stream("Once upon", max_tokens=32, temperature=0.8): print(piece, end="", flush=True)
        logits = m.forward(m.tokenize("abc"))            # numpy array [vocab] (if numpy is installed)

    with forge.Weights("model.fpk", device="auto") as w:   # one set of weights, several sequences
        a, b = w.session(), w.session(context=512)
        with forge.Batch(w, max_active=4) as batch:        # continuous batching
            outs = batch.generate_many([a.tokenize("Hi"), a.tokenize("Hello")], max_tokens=16)

The library is found via $FORGE_LIB, next to this package, in the build tree, or on the loader path
(libforge.so / libforge.dylib / forge.dll, by platform).
"""
from __future__ import annotations

import ctypes, json, os, queue, sys, threading
from ctypes import c_char_p, c_float, c_int, c_int32, c_int64, c_uint32, c_uint64, c_void_p, POINTER
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = ["Model", "Weights", "Batch", "ForgeError", "abi_version", "version", "json_schema_to_grammar", "grammar_check",
           "set_deterministic", "aot_precompile"]

TokenCallback = Callable[[int, str], Any]
Prompt = Union[str, Sequence[int]]

class ForgeError(RuntimeError):
    pass

def _lib_names() -> List[str]:
    if sys.platform.startswith("win") or sys.platform == "cygwin": return ["forge.dll", "libforge.dll"]
    if sys.platform == "darwin": return ["libforge.dylib", "libforge.1.dylib", "libforge.so"]
    return ["libforge.so", "libforge.so.1"]

def _load() -> ctypes.CDLL:
    cands: List[str] = []
    if os.environ.get("FORGE_LIB"): cands.append(os.environ["FORGE_LIB"])
    here = os.path.dirname(os.path.abspath(__file__))
    names = _lib_names()
    for d in (here, os.path.join(here, "..", "..", "build", "dev"), os.path.join(here, "..", "..", "build", "release")):
        cands += [os.path.join(d, n) for n in names]
    cands += names   # the loader path (LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH)
    err: Optional[OSError] = None
    for c in cands:
        try: return ctypes.CDLL(c)
        except OSError as e: err = e
    raise ForgeError(f"cannot load {names[0]} (set FORGE_LIB to its path): {err}")

_lib = _load()

class _Sampling(ctypes.Structure):
    _fields_ = [("temperature", c_float), ("top_k", c_int32), ("top_p", c_float), ("min_p", c_float),
                ("repetition_penalty", c_float), ("seed", c_uint64)]

class _GenOpts(ctypes.Structure):
    _fields_ = [("grammar", c_char_p), ("draft", c_void_p), ("spec_k", c_int32)]

class _SamplingExt(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("typical_p", c_float), ("mirostat_tau", c_float), ("mirostat_eta", c_float)]

_CB = ctypes.CFUNCTYPE(c_int, c_int32, c_char_p, c_void_p)

def _sig(name: str, res: Any, *args: Any) -> Any:
    f = getattr(_lib, name); f.restype = res; f.argtypes = list(args); return f

_abi = _sig("forge_abi_version", c_uint32)
_ver = _sig("forge_version", c_char_p)
_err = _sig("forge_last_error", c_char_p)
_ABI = int(_abi())
if _ABI < 2:
    raise ForgeError(f"libforge ABI {_ABI} is too old (need >= 2)")
_open = _sig("forge_open", c_void_p, c_char_p, c_int64, c_int)
_close = _sig("forge_close", None, c_void_p)
_reset = _sig("forge_reset", None, c_void_p)
_vocab = _sig("forge_vocab_size", c_int64, c_void_p)
_tok = _sig("forge_tokenize", c_int64, c_void_p, c_char_p, c_int, POINTER(c_int32), c_int64)
_detok = _sig("forge_detokenize", c_int64, c_void_p, POINTER(c_int32), c_int64, c_char_p, c_int64)
_fwd = _sig("forge_forward", c_int, c_void_p, POINTER(c_int32), c_int64, POINTER(c_float), c_int64)
_pos = _sig("forge_position", c_int64, c_void_p)
_rew = _sig("forge_rewind", c_int, c_void_p, c_int64)
_fwd_all = _sig("forge_forward_all", c_int, c_void_p, POINTER(c_int32), c_int64, POINTER(c_float), c_int64)
_gen = _sig("forge_generate_ex", c_int64, c_void_p, POINTER(c_int32), c_int64, c_int32, POINTER(_Sampling), POINTER(_GenOpts),
            POINTER(c_int32), c_int64, _CB, c_void_p)
_gen2 = None
if _ABI >= 3:
    _gen2 = _sig("forge_generate_ex2", c_int64, c_void_p, POINTER(c_int32), c_int64, c_int32, POINTER(_Sampling), POINTER(_GenOpts),
                 POINTER(_SamplingExt), POINTER(c_int32), c_int64, _CB, c_void_p)

# ---- ABI 4
class _OpenParams(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("verify_hashes", c_int32), ("device", c_char_p), ("deterministic", c_int32), ("no_aot", c_int32)]

class _SessionParams(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("kv_offload", c_int32), ("context", c_int64), ("kv_codec", c_char_p)]

class _LogitBias(ctypes.Structure):
    _fields_ = [("token", c_int32), ("bias", c_float)]

class _SamplerParams(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("temperature", c_float), ("top_k", c_int32), ("top_p", c_float), ("min_p", c_float),
                ("typical_p", c_float), ("mirostat_tau", c_float), ("mirostat_eta", c_float), ("repetition_penalty", c_float),
                ("presence_penalty", c_float), ("frequency_penalty", c_float), ("repetition_window", c_int32), ("seed", c_uint64),
                ("logit_bias", POINTER(_LogitBias)), ("n_logit_bias", c_int64)]

class _GenParams(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("spec_k", c_int32), ("sampler", POINTER(_SamplerParams)), ("grammar", c_char_p),
                ("json_schema", c_char_p), ("draft", c_void_p), ("stop_tokens", POINTER(c_int32)), ("n_stop_tokens", c_int64),
                ("ignore_eos", c_int32), ("reset", c_int32), ("on_token", _CB), ("user", c_void_p)]

class _BatchParams(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("max_active", c_int32), ("context", c_int64), ("prefill_chunk", c_int64),
                ("prefix_cache", c_int32), ("reserved", c_int32), ("kv_codec", c_char_p)]

_DONE = ctypes.CFUNCTYPE(None, c_uint64, POINTER(c_int32), c_int64, c_char_p, c_void_p)

class _BatchRequest(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("max_new", c_int32), ("prompt", POINTER(c_int32)), ("n_prompt", c_int64),
                ("sampler", POINTER(_SamplerParams)), ("grammar", c_char_p), ("json_schema", c_char_p),
                ("stop_tokens", POINTER(c_int32)), ("n_stop_tokens", c_int64), ("adapter", c_char_p), ("priority", c_int32),
                ("reserved", c_int32), ("on_token", _CB), ("on_done", _DONE), ("user", c_void_p)]

class _BatchStats(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("reserved", c_uint32)] + [(n, c_uint64) for n in
                ("steps", "completed", "failed", "prefix_hits", "prefix_tokens_reused", "peak_active", "peak_kv_bytes", "waiting", "active")]

_STOP_REASONS = {0: "length", 1: "eos", 2: "callback", 3: "grammar"}

if _ABI >= 4:
    _w_open = _sig("forge_weights_open", c_void_p, c_char_p, POINTER(_OpenParams))
    _w_close = _sig("forge_weights_close", None, c_void_p)
    _w_of = _sig("forge_model_weights", c_void_p, c_void_p)
    _w_err = _sig("forge_weights_error", c_char_p, c_void_p)
    _w_dev = _sig("forge_weights_device", c_char_p, c_void_p)
    _w_set_dev = _sig("forge_weights_set_device", c_char_p, c_void_p, c_char_p)
    _set_det = _sig("forge_set_deterministic", c_int, c_int)
    _s_open = _sig("forge_session_open", c_void_p, c_void_p, POINTER(_SessionParams))
    _m_err = _sig("forge_model_error", c_char_p, c_void_p)
    _ctx = _sig("forge_context_size", c_int64, c_void_p)
    _kind = _sig("forge_session_kind", c_char_p, c_void_p)
    _manifest = _sig("forge_manifest_json", c_int64, c_void_p, c_char_p, c_int64)
    _t_bos = _sig("forge_token_bos", c_int32, c_void_p)
    _t_eos = _sig("forge_token_eos", c_int32, c_void_p)
    _t_text = _sig("forge_token_text", c_int64, c_void_p, c_int32, c_char_p, c_int64)
    _t_id = _sig("forge_token_id", c_int32, c_void_p, c_char_p)
    _t_special = _sig("forge_token_is_special", c_int, c_void_p, c_int32)
    _stops = _sig("forge_stop_tokens", c_int64, c_void_p, c_int, POINTER(c_int32), c_int64)
    _tmpl = _sig("forge_chat_template", c_int64, c_void_p, c_char_p, c_int64)
    _tmpl_apply = _sig("forge_chat_template_apply_ex", c_int64, c_void_p, c_char_p, c_char_p, c_char_p, c_int, c_char_p, c_int64)
    _gen4 = _sig("forge_generate_v4", c_int64, c_void_p, POINTER(c_int32), c_int64, c_int32, POINTER(_GenParams), POINTER(c_int32), c_int64, POINTER(c_int32))
    _schema = _sig("forge_json_schema_to_grammar", c_int64, c_char_p, c_char_p, c_int64)
    _gcheck = _sig("forge_grammar_check", c_int, c_char_p)
    _ad_load = _sig("forge_adapter_load", c_int, c_void_p, c_char_p, c_char_p)
    _ad_has = _sig("forge_adapter_exists", c_int, c_void_p, c_char_p)
    _ad_set = _sig("forge_adapter_set", c_int, c_void_p, c_char_p)
    _ad_get = _sig("forge_adapter_get", c_char_p, c_void_p)
    _st_save = _sig("forge_state_save", c_int64, c_void_p, c_int64, c_void_p, c_int64)
    _st_load = _sig("forge_state_load", c_int, c_void_p, c_void_p, c_int64)
    _st_copy = _sig("forge_state_copy", c_int, c_void_p, c_void_p, c_int64)
    _st_bytes = _sig("forge_state_bytes", c_int64, c_void_p)
    _e_dim = _sig("forge_embed_dim", c_int64, c_void_p)
    _e_tokens = _sig("forge_embed_tokens", c_int64, c_void_p, POINTER(c_int32), c_int64, POINTER(c_int32), POINTER(c_float), c_int64)
    _e_text = _sig("forge_embed_text", c_int64, c_void_p, c_char_p, c_char_p, POINTER(c_float), c_int64, POINTER(c_int64))
    _e_texts = _sig("forge_embed_texts", c_int64, c_void_p, POINTER(c_char_p), c_int64, POINTER(c_float), c_int64)
    _b_open = _sig("forge_batch_open", c_void_p, c_void_p, POINTER(_BatchParams))
    _b_close = _sig("forge_batch_close", None, c_void_p)
    _b_err = _sig("forge_batch_error", c_char_p, c_void_p)
    _b_submit = _sig("forge_batch_submit", c_uint64, c_void_p, POINTER(_BatchRequest))
    _b_cancel = _sig("forge_batch_cancel", c_int, c_void_p, c_uint64)
    _b_step = _sig("forge_batch_step", c_int, c_void_p)
    _b_run = _sig("forge_batch_run", c_int, c_void_p)
    _b_stats = _sig("forge_batch_stats_get", c_int, c_void_p, POINTER(_BatchStats))
    _b_evict = _sig("forge_batch_evict_prefix_cache", None, c_void_p, c_int64)
    _aot = _sig("forge_aot_precompile", c_int64, c_char_p, c_char_p, POINTER(c_int32), c_int64)

def abi_version() -> int: return _ABI
def version() -> str: return _ver().decode()

def _need(abi: int, what: str) -> None:
    if _ABI < abi: raise ForgeError(f"{what} needs libforge ABI >= {abi} (this library is ABI {_ABI})")

def _check(ok: Any) -> None:
    if not ok: raise ForgeError(_err().decode(errors="replace"))

def _ids(seq: Iterable[int]) -> Tuple[Any, int]:
    seq = list(seq); return (c_int32 * max(len(seq), 1))(*seq), len(seq)

def _floats(buf: Any) -> Any:
    try:
        import numpy as np
        return np.ctypeslib.as_array(buf).copy()
    except ImportError:
        return list(buf)

def _text(call: Callable[[Any, int], int]) -> str:
    """A C function that writes text and returns the bytes needed: size the buffer, then read."""
    need = call(None, 0); _check(need >= 0)
    buf = ctypes.create_string_buffer(need + 1); call(buf, need + 1)
    return buf.raw[:need].decode(errors="replace")

def _json_arg(v: Any) -> Optional[bytes]:
    if v is None: return None
    return (v if isinstance(v, str) else json.dumps(v)).encode()

def json_schema_to_grammar(schema: Union[str, Dict[str, Any]]) -> str:
    """GBNF text for a JSON Schema (the subset in docs/inference.md); output is compact JSON."""
    _need(4, "json_schema_to_grammar")
    s = _json_arg(schema)
    return _text(lambda b, n: _schema(s, b, n))

def grammar_check(grammar: str) -> None:
    """Raises ForgeError with the parse error when `grammar` is not valid GBNF."""
    _need(4, "grammar_check")
    _check(_gcheck(grammar.encode()) == 0)

def set_deterministic(mode: Optional[bool]) -> bool:
    """Process-wide determinism mode: True / False, or None to follow $FORGE_DETERMINISTIC. Returns the mode in effect."""
    _need(4, "set_deterministic")
    return _set_det(-1 if mode is None else int(bool(mode))) == 1

def aot_precompile(path: str, out: Optional[str] = None, seq_lengths: Optional[Sequence[int]] = None) -> int:
    """Precompile the model's CPU plans into a sidecar (`<path>.fplan` unless `out`); returns the number of plans."""
    _need(4, "aot_precompile")
    arr, n = _ids(seq_lengths or [])
    k = _aot(os.fsencode(path), os.fsencode(out) if out else None, arr if n else None, n); _check(k >= 0)
    return int(k)

def _sampler(temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, typical_p: float = 1.0,
             mirostat_tau: float = 0.0, mirostat_eta: float = 0.1, repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
             frequency_penalty: float = 0.0, repetition_window: int = 64, seed: int = 0,
             logit_bias: Optional[Dict[int, float]] = None) -> Tuple[_SamplerParams, Any]:
    """The C sampler struct, and the objects it points to (keep them alive while it is in use)."""
    lb = None
    if logit_bias:
        lb = (_LogitBias * len(logit_bias))(*[_LogitBias(int(t), float(b)) for t, b in logit_bias.items()])
    s = _SamplerParams(ctypes.sizeof(_SamplerParams), temperature, top_k, top_p, min_p, typical_p, mirostat_tau, mirostat_eta,
                       repetition_penalty, presence_penalty, frequency_penalty, repetition_window, seed, lb, len(logit_bias) if logit_bias else 0)
    return s, lb

class Weights:
    """One opened model file: weights, tokenizer, compiled plans, LoRA adapters, device. Sessions (`session()`) and
    batches (`Batch`) share it; it may be closed while they are still in use."""
    def __init__(self, path: str, verify: bool = False, device: Optional[str] = None, deterministic: Optional[bool] = None, aot: bool = True):
        _need(4, "Weights")
        p = _OpenParams(ctypes.sizeof(_OpenParams), int(verify), device.encode() if device else None,
                        0 if deterministic is None else (1 if deterministic else 2), 0 if aot else 1)
        self._h: Optional[int] = _w_open(os.fsencode(path), ctypes.byref(p))
        _check(self._h)
        self.path = path

    @classmethod
    def _wrap(cls, handle: int, path: str = "") -> "Weights":
        w = cls.__new__(cls); w._h = handle; w.path = path; return w

    def close(self) -> None:
        if getattr(self, "_h", None): _w_close(self._h); self._h = None
    __del__ = close
    def __enter__(self) -> "Weights": return self
    def __exit__(self, *a: Any) -> None: self.close()

    @property
    def device(self) -> str:
        return _w_dev(self._h).decode()
    def set_device(self, device: str) -> str:
        """Select the execution device ("cpu", "auto", "vulkan", "cuda", ...); returns the one in use (the CPU when
        the requested device is not available). Not while sessions on these weights are running."""
        r = _w_set_dev(self._h, device.encode()); _check(r)
        return r.decode()

    def session(self, context: int = 0, kv_codec: Optional[str] = None, kv_offload: bool = False) -> "Model":
        """A new sequence (KV state) on these weights."""
        return Model(self, context=context, kv_codec=kv_codec, kv_offload=kv_offload)

    def load_adapter(self, name: str, path: str) -> None:
        """Load a LoRA adapter (.fpk or .safetensors with <weight>.lora_a / .lora_b). Not while sessions are running."""
        _check(_ad_load(self._h, name.encode(), os.fsencode(path)) == 0)
    def has_adapter(self, name: str) -> bool:
        return _ad_has(self._h, name.encode()) == 1

class Model:
    """One sequence: KV state and position. `Model(path)` opens its own weights; `Model(weights)` or
    `weights.session()` shares them. Not thread-safe: one thread at a time per Model."""
    def __init__(self, source: Union[str, "os.PathLike[str]", Weights], context: int = 0, verify: bool = False, device: Optional[str] = None,
                 kv_codec: Optional[str] = None, kv_offload: bool = False):
        self._h: Optional[int] = None
        self._weights: Optional[Weights] = None
        if isinstance(source, Weights):
            self._weights = source
        elif _ABI >= 4:
            self._weights = Weights(os.fspath(source), verify=verify, device=device)
            self._own_weights = True
        else:
            if device: os.environ["FORGE_DEVICE"] = device
            self._h = _open(os.fsencode(source), context, int(verify))
            _check(self._h)
        if self._weights is not None:
            sp = _SessionParams(ctypes.sizeof(_SessionParams), int(kv_offload), context, kv_codec.encode() if kv_codec else None)
            self._h = _s_open(self._weights._h, ctypes.byref(sp))
            _check(self._h)
        self.vocab_size = int(_vocab(self._h))
        self.last_stop_reason: Optional[str] = None

    def close(self) -> None:
        if getattr(self, "_h", None): _close(self._h); self._h = None
        if getattr(self, "_own_weights", False) and self._weights is not None: self._weights.close(); self._own_weights = False
    __del__ = close
    def __enter__(self) -> "Model": return self
    def __exit__(self, *a: Any) -> None: self.close()

    def _ok(self, ok: Any) -> None:
        """Raise this handle's error (the thread-local one on older libraries)."""
        if ok: return
        msg = _m_err(self._h) if _ABI >= 4 and self._h else None
        raise ForgeError((msg or _err()).decode(errors="replace"))

    # ---- model facts
    @property
    def weights(self) -> Weights:
        """The shared weights of this session (ABI 4)."""
        _need(4, "Model.weights")
        assert self._weights is not None
        return self._weights
    @property
    def position(self) -> int: return int(_pos(self._h))
    @property
    def context_size(self) -> int:
        _need(4, "context_size"); return int(_ctx(self._h))
    @property
    def kind(self) -> str:
        """The session kind: "generative_sequence", "embedding", ..."""
        _need(4, "kind"); return _kind(self._h).decode()
    @property
    def manifest(self) -> Dict[str, Any]:
        _need(4, "manifest"); return json.loads(_text(lambda b, n: _manifest(self._h, b, n)))
    def reset(self) -> None: _reset(self._h)
    def rewind(self, position: int) -> None: self._ok(_rew(self._h, position) == 0)

    # ---- tokenizer
    def tokenize(self, text: str, add_bos: Optional[bool] = None) -> List[int]:
        """Token ids. add_bos None: the model's default (no BOS for Qwen-family models, [CLS] .. [SEP] for BERT)."""
        b = text.encode(); ab = -1 if add_bos is None else int(bool(add_bos))
        n = _tok(self._h, b, ab, None, 0); self._ok(n >= 0)
        out = (c_int32 * max(n, 1))(); _tok(self._h, b, ab, out, n)
        return list(out[:n])

    def detokenize(self, ids: Iterable[int]) -> str:
        arr, n = _ids(ids)
        need = _detok(self._h, arr, n, None, 0); self._ok(need >= 0)
        buf = ctypes.create_string_buffer(need + 1); _detok(self._h, arr, n, buf, need + 1)
        return buf.raw[:need].decode(errors="replace")

    @property
    def bos_token(self) -> int:
        _need(4, "bos_token"); return int(_t_bos(self._h))
    @property
    def eos_token(self) -> int:
        _need(4, "eos_token"); return int(_t_eos(self._h))
    def token_bytes(self, token: int) -> bytes:
        """The bytes one token adds to a text (possibly part of a UTF-8 character)."""
        _need(4, "token_bytes")
        need = _t_text(self._h, token, None, 0); self._ok(need >= 0)
        buf = ctypes.create_string_buffer(need + 1); _t_text(self._h, token, buf, need + 1)
        return buf.raw[:need]
    def token_id(self, token: str) -> int:
        """The id of a vocabulary entry given as stored ("<|im_end|>"), or -1."""
        _need(4, "token_id"); return int(_t_id(self._h, token.encode()))
    def is_special(self, token: int) -> bool:
        _need(4, "is_special"); return _t_special(self._h, token) == 1
    def stop_tokens(self, chat: bool = True) -> List[int]:
        """Ids that end an assistant turn: EOS, the manifest's eos_token_ids and (chat) the template's end-of-turn markers."""
        _need(4, "stop_tokens")
        n = _stops(self._h, int(chat), None, 0); self._ok(n >= 0)
        out = (c_int32 * max(n, 1))(); _stops(self._h, int(chat), out, n)
        return list(out[:n])
    @property
    def chat_template(self) -> str:
        """The model's Jinja chat template, "" when the file has none."""
        _need(4, "chat_template"); return _text(lambda b, n: _tmpl(self._h, b, n))
    def apply_chat_template(self, messages: Union[str, List[Dict[str, Any]]], tools: Optional[Union[str, List[Dict[str, Any]]]] = None,
                            add_generation_prompt: bool = True, **template_vars: Any) -> str:
        """Render the model's chat template. Tokenize the result with add_bos=False (the template writes BOS)."""
        _need(4, "apply_chat_template")
        m, t, x = _json_arg(messages), _json_arg(tools), _json_arg(template_vars or None)
        need = _tmpl_apply(self._h, m, t, x, int(add_generation_prompt), None, 0); self._ok(need >= 0)
        buf = ctypes.create_string_buffer(need + 1); _tmpl_apply(self._h, m, t, x, int(add_generation_prompt), buf, need + 1)
        return buf.raw[:need].decode(errors="replace")

    # ---- raw forward passes
    def forward(self, ids: Iterable[int]) -> Any:
        """Continue from the current position; returns last-position logits."""
        arr, n = _ids(ids); out = (c_float * self.vocab_size)()
        self._ok(_fwd(self._h, arr, n, out, self.vocab_size) == 0)
        return _floats(out)

    def forward_all(self, ids: Iterable[int]) -> Any:
        """Logits for every position, shape [len(ids), vocab]."""
        arr, n = _ids(ids); out = (c_float * (n * self.vocab_size))()
        self._ok(_fwd_all(self._h, arr, n, out, n * self.vocab_size) == 0)
        r = _floats(out)
        return r.reshape(n, self.vocab_size) if hasattr(r, "reshape") else [r[i * self.vocab_size:(i + 1) * self.vocab_size] for i in range(n)]

    # ---- generation
    def generate(self, prompt: Prompt, max_tokens: int = 32, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                 repetition_penalty: float = 1.0, seed: int = 0, grammar: Optional[str] = None, draft: Optional["Model"] = None, spec_k: int = 4,
                 on_token: Optional[TokenCallback] = None, typical_p: float = 1.0, mirostat_tau: float = 0.0, mirostat_eta: float = 0.1,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0, repetition_window: int = 64,
                 logit_bias: Optional[Dict[int, float]] = None, json_schema: Optional[Union[str, Dict[str, Any]]] = None,
                 stop_tokens: Optional[Sequence[int]] = None, ignore_eos: bool = False, reset: bool = True,
                 _on_bytes: Optional[Callable[[int, bytes], Any]] = None) -> List[int]:
        """Generate from a string or token ids; returns the new ids. The state is reset first unless reset=False.
        on_token(id, text) -> True to stop early (it is called while generation runs). grammar is GBNF text,
        json_schema a schema (dict or text), draft another Model for speculative decoding. With libforge ABI >= 4 the
        token that ends generation (EOS, a stop token) is not in the result, and `last_stop_reason` says why it ended
        ("length", "eos", "callback", "grammar")."""
        if isinstance(prompt, str): prompt = self.tokenize(prompt)
        arr, n = _ids(prompt); out = (c_int32 * max(max_tokens, 1))()
        err: List[BaseException] = []
        def cb(t: int, txt: Optional[bytes], _u: Any) -> int:
            try:
                stop = bool(_on_bytes(int(t), txt or b"")) if _on_bytes is not None else False   # the token's raw bytes (stream)
                if on_token is not None and on_token(int(t), txt.decode(errors="replace") if txt else ""): stop = True
                return 1 if stop else 0
            except BaseException as e: err.append(e); return 1
        ccb = _CB(cb) if on_token is not None or _on_bytes is not None else _CB()   # no Python callback: a NULL pointer, no call per token
        if _ABI >= 4:
            s, keep = _sampler(temperature, top_k, top_p, min_p, typical_p, mirostat_tau, mirostat_eta, repetition_penalty,
                               presence_penalty, frequency_penalty, repetition_window, seed, logit_bias)
            st, ns = _ids(stop_tokens or [])
            p = _GenParams(ctypes.sizeof(_GenParams), spec_k, ctypes.pointer(s), grammar.encode() if grammar else None, _json_arg(json_schema),
                           draft._h if draft else None, st if ns else None, ns, int(ignore_eos), int(reset), ccb, None)
            why = c_int32(0)
            k = _gen4(self._h, arr, n, max_tokens, ctypes.byref(p), out, max_tokens, ctypes.byref(why))
            del keep
            if err: raise err[0]
            self._ok(k >= 0)
            self.last_stop_reason = _STOP_REASONS.get(why.value, "length")
            return list(out[:k])
        # older libraries: the ABI 2 / 3 entry points (no penalties beyond repetition, no schema, stop tokens or bias)
        if presence_penalty or frequency_penalty or logit_bias or json_schema or stop_tokens or ignore_eos:
            _need(4, "presence / frequency penalties, logit_bias, json_schema, stop_tokens and ignore_eos")
        if reset: self.reset()
        s2 = _Sampling(temperature, top_k, top_p, min_p, repetition_penalty, seed)
        o = _GenOpts(grammar.encode() if grammar else None, draft._h if draft else None, spec_k)
        if typical_p != 1.0 or mirostat_tau:
            if _gen2 is None: raise ForgeError("typical_p / mirostat need libforge ABI >= 3")
            x = _SamplingExt(ctypes.sizeof(_SamplingExt), typical_p, mirostat_tau, mirostat_eta)
            k = _gen2(self._h, arr, n, max_tokens, ctypes.byref(s2), ctypes.byref(o), ctypes.byref(x), out, max_tokens, ccb, None)
        else:
            k = _gen(self._h, arr, n, max_tokens, ctypes.byref(s2), ctypes.byref(o), out, max_tokens, ccb, None)
        if err: raise err[0]
        self._ok(k >= 0)
        return list(out[:k])

    def complete(self, text: str, **kw: Any) -> str:
        """Generated text for a prompt string (see generate for the keyword arguments)."""
        return self.detokenize(self.generate(self.tokenize(text), **kw))

    def stream(self, prompt: Prompt, **kw: Any) -> Iterator[str]:
        """Yields text pieces while generation runs (the C token callback feeds a queue from a worker thread).
        Bytes of a UTF-8 character that is split over tokens are held until it is complete. Closing the generator
        early (break, .close()) stops the generation at the next token. Do not use the Model for anything else
        until the generator is finished or closed."""
        q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        stop = threading.Event()
        def on_bytes(t: int, raw: bytes) -> bool:
            q.put(("tok", raw))
            return stop.is_set()
        def work() -> None:
            try: self.generate(prompt, _on_bytes=on_bytes, **kw); q.put(("end", None))
            except BaseException as e: q.put(("err", e))
        th = threading.Thread(target=work, name="forge-stream", daemon=True)
        th.start()
        pending = b""
        try:
            while True:
                kind, val = q.get()
                if kind == "err": raise val
                if kind == "end": break
                pending += val
                cut = len(pending)   # keep back an incomplete trailing UTF-8 sequence
                for back in range(1, min(4, len(pending)) + 1):
                    c = pending[-back]
                    if c & 0xC0 == 0x80: continue
                    need = 1 if c < 0x80 else 2 if c >> 5 == 0b110 else 3 if c >> 4 == 0b1110 else 4 if c >> 3 == 0b11110 else 1
                    if need > back: cut = len(pending) - back
                    break
                if cut:
                    yield pending[:cut].decode(errors="replace"); pending = pending[cut:]
            if pending: yield pending.decode(errors="replace")
        finally:
            stop.set(); th.join()

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kw: Any) -> str:
        """One assistant turn: the messages through the model's chat template, generation up to an end-of-turn token."""
        ids = self.tokenize(self.apply_chat_template(messages, tools=tools), add_bos=False)
        kw.setdefault("stop_tokens", self.stop_tokens(chat=True))
        return self.detokenize(self.generate(ids, **kw))

    # ---- LoRA
    @property
    def adapter(self) -> str:
        """The selected LoRA adapter ("" = base weights). Assigning clears the KV state when the adapter changes."""
        _need(4, "adapter"); return _ad_get(self._h).decode()
    @adapter.setter
    def adapter(self, name: Optional[str]) -> None:
        _need(4, "adapter"); self._ok(_ad_set(self._h, name.encode() if name else None) == 0)

    # ---- KV state
    def save_state(self, n_tokens: int = -1) -> bytes:
        """The KV state of the first n_tokens tokens (all by default), for load_state on a session of the same model."""
        _need(4, "save_state")
        need = _st_save(self._h, n_tokens, None, 0); self._ok(need >= 0)
        buf = ctypes.create_string_buffer(need)
        self._ok(_st_save(self._h, n_tokens, buf, need) == need)
        return buf.raw
    def load_state(self, blob: bytes) -> None:
        _need(4, "load_state"); self._ok(_st_load(self._h, blob, len(blob)) == 0)
    def copy_state_from(self, other: "Model", n_tokens: Optional[int] = None) -> None:
        """Take the first n_tokens tokens (all by default) of another session on the same weights (prefix sharing)."""
        _need(4, "copy_state_from")
        self._ok(_st_copy(self._h, other._h, other.position if n_tokens is None else n_tokens) == 0)
    @property
    def state_bytes(self) -> int:
        _need(4, "state_bytes"); return int(_st_bytes(self._h))

    # ---- embeddings
    @property
    def embed_dim(self) -> int:
        _need(4, "embed_dim")
        d = _e_dim(self._h); self._ok(d >= 0)
        return int(d)
    def embed(self, input: Union[str, Sequence[int], Sequence[str]], pair: Optional[str] = None) -> Any:
        """Embedding models: one vector for a string or a list of token ids ([dim]), or one row per string for a list of
        strings ([n, dim]). `pair` is a second sequence for a single string (query / passage). numpy arrays when numpy
        is installed, else lists."""
        _need(4, "embed")
        dim = self.embed_dim
        if isinstance(input, str):
            out = (c_float * dim)()
            self._ok(_e_text(self._h, input.encode(), pair.encode() if pair else None, out, dim, None) == dim)
            return _floats(out)
        items = list(input)
        if items and isinstance(items[0], str):
            arr = (c_char_p * len(items))(*[s.encode() for s in items])
            out = (c_float * (dim * len(items)))()
            self._ok(_e_texts(self._h, arr, len(items), out, dim * len(items)) == dim)
            r = _floats(out)
            return r.reshape(len(items), dim) if hasattr(r, "reshape") else [r[i * dim:(i + 1) * dim] for i in range(len(items))]
        ids, n = _ids(items); out = (c_float * dim)()
        self._ok(_e_tokens(self._h, ids, n, None, out, dim) == dim)
        return _floats(out)

class Batch:
    """Continuous batching over shared weights: submit requests, then step() / run(). Callbacks run on the thread that
    calls step() / run(). results[id] = (tokens, reason) for every finished request."""
    def __init__(self, weights: Weights, max_active: int = 8, context: int = 0, prefill_chunk: int = 256, prefix_cache: int = 4,
                 kv_codec: Optional[str] = None):
        _need(4, "Batch")
        p = _BatchParams(ctypes.sizeof(_BatchParams), max_active, context, prefill_chunk, prefix_cache if prefix_cache > 0 else -1, 0,
                         kv_codec.encode() if kv_codec else None)
        self._h: Optional[int] = _b_open(weights._h, ctypes.byref(p))
        if not self._h: raise ForgeError(_err().decode(errors="replace"))
        self.weights = weights
        self.results: Dict[int, Tuple[List[int], str]] = {}
        self._live: Dict[int, Any] = {}      # per request: the C callbacks and buffers, kept until it is done
        self._errors: List[BaseException] = []
        self._next_key = 0

    def close(self) -> None:
        if getattr(self, "_h", None): _b_close(self._h); self._h = None; self._live.clear()
    __del__ = close
    def __enter__(self) -> "Batch": return self
    def __exit__(self, *a: Any) -> None: self.close()

    def _ok(self, ok: Any) -> None:
        if not ok: raise ForgeError(_b_err(self._h).decode(errors="replace"))

    def submit(self, prompt: Sequence[int], max_tokens: int = 32, grammar: Optional[str] = None, json_schema: Optional[Union[str, Dict[str, Any]]] = None,
               stop_tokens: Optional[Sequence[int]] = None, adapter: Optional[str] = None, priority: int = 0,
               on_token: Optional[TokenCallback] = None, on_done: Optional[Callable[[int, List[int], str], Any]] = None, **sampling: Any) -> int:
        """Queue a request (token ids); returns its id. sampling: the sampler keyword arguments of Model.generate."""
        s, keep = _sampler(**sampling)
        arr, n = _ids(prompt); st, ns = _ids(stop_tokens or [])
        self._next_key += 1; key = self._next_key
        def tok_cb(t: int, txt: Optional[bytes], _u: Any) -> int:
            try: return 1 if on_token is not None and on_token(int(t), txt.decode(errors="replace") if txt else "") else 0
            except BaseException as e: self._errors.append(e); return 1
        def done_cb(rid: int, toks: Any, k: int, reason: Optional[bytes], _u: Any) -> None:
            ids = [toks[i] for i in range(k)]; why = (reason or b"").decode(errors="replace")
            self.results[int(rid)] = (ids, why)
            self._live.pop(key, None)
            if on_done is not None:
                try: on_done(int(rid), ids, why)
                except BaseException as e: self._errors.append(e)
        ctok, cdone = (_CB(tok_cb) if on_token is not None else _CB()), _DONE(done_cb)
        r = _BatchRequest(ctypes.sizeof(_BatchRequest), max_tokens, arr, n, ctypes.pointer(s), grammar.encode() if grammar else None, _json_arg(json_schema),
                          st if ns else None, ns, adapter.encode() if adapter else None, priority, 0, ctok, cdone, None)
        self._live[key] = (ctok, cdone)   # the C side copies everything else during submit
        rid = _b_submit(self._h, ctypes.byref(r))
        del keep
        if not rid: self._live.pop(key, None); self._ok(False)
        return int(rid)

    def _raise_pending(self) -> None:
        if self._errors: e = self._errors[0]; self._errors.clear(); raise e
    def step(self) -> bool:
        """One scheduling step; False when nothing is waiting or running."""
        r = _b_step(self._h); self._raise_pending(); self._ok(r >= 0)
        return r == 1
    def run(self) -> None:
        """Step until idle."""
        r = _b_run(self._h); self._raise_pending(); self._ok(r == 0)
    def cancel(self, request_id: int) -> bool:
        return _b_cancel(self._h, request_id) == 1
    def stats(self) -> Dict[str, int]:
        s = _BatchStats(ctypes.sizeof(_BatchStats)); self._ok(_b_stats(self._h, ctypes.byref(s)) == 0)
        return {n: int(getattr(s, n)) for n, _ in _BatchStats._fields_[2:]}
    def evict_prefix_cache(self, keep: int = 0) -> None:
        _b_evict(self._h, keep)

    def generate_many(self, prompts: Sequence[Sequence[int]], **kw: Any) -> List[List[int]]:
        """Run several prompts (token ids) together; returns their new tokens in order. A failed request raises."""
        ids = [self.submit(p, **kw) for p in prompts]
        self.run()
        out = []
        for i in ids:
            toks, why = self.results.pop(i)
            if why.startswith("error: "): raise ForgeError(why[7:])
            out.append(toks)
        return out

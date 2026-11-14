#pragma once
// Umbrella header for inference: sampler, model/session, speculative decoding, GBNF grammar, the batch scheduler, chat templates and tool calls.
// Guide: docs/inference.md. Stable C ABI: forge.h (docs/capi.md).
#include <forge/inference/sampler.hpp>
#include <forge/inference/session.hpp>
#include <forge/inference/speculative.hpp>
#include <forge/inference/grammar.hpp>
#include <forge/inference/batch.hpp>
#include <forge/inference/chat_template.hpp>
#include <forge/inference/chat_tools.hpp>
#include <forge/inference/remote.hpp>          // remote AI servers / commands with the same chat shape (docs/inference.md)
#include <forge/inference/file_sessions.hpp>   // session kinds chosen by the file: diffusion_loop, streaming

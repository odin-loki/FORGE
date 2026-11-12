#pragma once
// Umbrella header for inference: sampler, model/session, speculative decoding, GBNF grammar, the batch scheduler and chat templates.
// Guide: docs/inference.md. Stable C ABI: forge.h (docs/capi.md).
#include <forge/inference/sampler.hpp>
#include <forge/inference/session.hpp>
#include <forge/inference/speculative.hpp>
#include <forge/inference/grammar.hpp>
#include <forge/inference/batch.hpp>
#include <forge/inference/chat_template.hpp>

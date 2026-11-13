#pragma once
// The .fpk "calib" section (FORGE_PLAN D.2 "[CALIB] optional calibration data"): per-tensor statistics of the
// full-precision weights, recorded when a model is imported (`forge import --calib`) and read when it is quantised
// again (`forge requant`, import/requant.hpp), where the original values are no longer at hand.
//
//   {"version":1,"source":"weights","bins":64,"tensors":{
//      "layer.0.wq":{"n":1024,"min":-0.41,"max":0.39,"mean":0.001,"rms":0.12,"absmax":0.41,"hist":[...64 counts...]}}}
//
// hist counts |x| in `bins` equal bins over [0, absmax] (the last bin includes absmax), so quantiles of the magnitude
// can be read back without the tensor. What is recorded is the distribution of the weights; activation statistics
// (which need calibration inputs run through the model) are not collected.
#include <forge/core/error.hpp>
#include <forge/json/json.hpp>
#include <cmath>
#include <map>
#include <span>

namespace forge::modelfile {

inline constexpr char calib_section[] = "calib";
inline constexpr std::size_t calib_bins = 64;

struct tensor_stats {
    std::uint64_t n = 0;
    double min = 0, max = 0, mean = 0, rms = 0, absmax = 0;
    std::vector<std::uint64_t> hist;   // calib_bins counts of |x| over [0, absmax]
    // absmax / rms: about sqrt(2 ln n) (4 to 6) for Gaussian weights, far larger when a few values dominate the range
    double crest() const { return rms > 0 ? absmax / rms : 0; }
    // the magnitude below which a fraction q of the values lie (bin upper edge: an upper bound)
    double quantile(double q) const {
        if (hist.empty() || !n) return absmax;
        std::uint64_t want = std::uint64_t(std::ceil(q * double(n))), acc = 0;
        for (std::size_t i = 0; i < hist.size(); ++i) { acc += hist[i]; if (acc >= want) return absmax * double(i + 1) / double(hist.size()); }
        return absmax;
    }
    static tensor_stats of(std::span<const float> x) {
        tensor_stats s; s.n = x.size(); s.hist.assign(calib_bins, 0);
        if (x.empty()) return s;
        double sum = 0, sq = 0; float lo = x[0], hi = x[0];
        for (float v : x) { sum += v; sq += double(v) * v; lo = std::min(lo, v); hi = std::max(hi, v); }
        s.min = lo; s.max = hi; s.mean = sum / double(x.size()); s.rms = std::sqrt(sq / double(x.size())); s.absmax = std::max(std::fabs(double(lo)), std::fabs(double(hi)));
        if (s.absmax > 0 && std::isfinite(s.absmax)) {
            const double k = double(calib_bins) / s.absmax;
            for (float v : x) { auto b = std::size_t(std::fabs(double(v)) * k); ++s.hist[std::min(b, calib_bins - 1)]; }
        } else s.hist[0] = x.size();
        return s;
    }
};

struct calib_table {
    std::string source = "weights";
    std::map<std::string, tensor_stats> tensors;
    const tensor_stats* find(const std::string& n) const { auto it = tensors.find(n); return it == tensors.end() ? nullptr : &it->second; }
    bool empty() const { return tensors.empty(); }

    std::string dump() const {
        json::object ts;
        for (auto& [name, s] : tensors) {
            json::array h; for (auto c : s.hist) h.push_back(double(c));
            ts.emplace_back(name, json::object{{"n", double(s.n)}, {"min", s.min}, {"max", s.max}, {"mean", s.mean}, {"rms", s.rms}, {"absmax", s.absmax}, {"hist", h}});
        }
        return json::dump(json::value(json::object{{"version", 1.0}, {"source", source}, {"bins", double(calib_bins)}, {"tensors", ts}}));
    }
    // Parses a "calib" section (untrusted): bounded counts, finite numbers, a histogram that sums to n.
    static calib_table parse(std::string_view text) {
        auto bad = [](const std::string& why) { raise(errc::format_error, "calib section: {}", why); };
        json::value v = json::parse(text);
        if (!v.is_obj()) bad("not a JSON object");
        if (v.num_or("version", 0) != 1) raise(errc::unsupported, "calib section: version {} is not supported (this build reads version 1)", v.num_or("version", 0));
        calib_table t; t.source = v.str_or("source", "weights");
        if (t.source.size() > 64) bad("bad source");
        auto* ts = v.find("tensors");
        if (!ts || !ts->is_obj() || ts->obj().size() > (std::size_t(1) << 20)) bad("no tensor table");
        for (auto& [name, e] : ts->obj()) {
            if (name.empty() || name.size() > 256 || !e.is_obj()) bad("bad entry");
            tensor_stats s;
            double n = e.num_or("n", -1);
            if (!(n >= 0 && n <= 9e15) || n != std::floor(n)) bad("bad count for " + name);
            s.n = std::uint64_t(n);
            s.min = e.num_or("min", 0); s.max = e.num_or("max", 0); s.mean = e.num_or("mean", 0); s.rms = e.num_or("rms", 0); s.absmax = e.num_or("absmax", 0);
            for (double x : {s.min, s.max, s.mean, s.rms, s.absmax}) if (!std::isfinite(x)) bad("non-finite statistic for " + name);
            if (s.rms < 0 || s.absmax < 0 || s.min > s.max) bad("inconsistent statistics for " + name);
            if (auto* h = e.find("hist"); h && h->is_arr()) {
                if (h->arr().size() > 4096) bad("histogram too long for " + name);
                std::uint64_t sum = 0;
                for (auto& c : h->arr()) { if (!c.is_num() || !(c.num() >= 0 && c.num() <= 9e15) || c.num() != std::floor(c.num())) bad("bad histogram for " + name); s.hist.push_back(std::uint64_t(c.num())); sum += s.hist.back(); if (sum > std::uint64_t(9e15)) bad("bad histogram for " + name); }
                if (sum != s.n) bad("histogram of " + name + " does not sum to its count");
                if (s.hist.size() > 0 && s.n > std::uint64_t(9e15)) bad("bad count for " + name);
            }
            if (!t.tensors.emplace(name, std::move(s)).second) bad(name + " is listed twice");
        }
        return t;
    }
};

} // namespace forge::modelfile

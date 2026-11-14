// ForgeScript: the full compression experiment plan for three ~1B-parameter models, run with
//
//   forge script tools/compress_plan.cpp [PHASES]        PHASES: any of base,chains,sweeps (default: all)
//
// Environment: FORGE_BIN (forge), COMPRESS_MODELS (dir of the imported .f32.fpk files), COMPRESS_DATA (output of
// tools/compress_data.py), COMPRESS_RUNS (scratch for model files), COMPRESS_RESULTS (results.jsonl + REPORT.md).
//
// Every job is `forge compress` (+ evaluation) or `forge eval`; its record is appended to results.jsonl and the
// report is regenerated after each job, so the plan can be stopped and resumed (finished jobs are skipped).
//
// Supervision (long unattended runs; tools/compress_plan_supervise.sh / .ps1 restart the plan until it is done):
//   - one plan at a time: an OS lock on COMPRESS_RUNS/plan.lock (a second instance exits with status 75)
//   - COMPRESS_RUNS/plan.heartbeat: {pid, time, job} rewritten at least every 30 s while the plan runs;
//     COMPRESS_RUNS/plan.done is written when every phase finished
//   - jobs run with --checkpoint COMPRESS_RUNS/ckpt/<job> (forge compress resumes mid-job after a crash or a
//     restart of the plan) and inherit FORGE_MEM_LIMIT (COMPRESS_MEM_LIMIT, else FORGE_MEM_LIMIT) when set
//   - a failed job is retried COMPRESS_RETRIES times (default 3) with exponential backoff (COMPRESS_BACKOFF s,
//     default 30); a job killed for memory (exit 137 / 9, std::bad_alloc, "out of memory") is retried at once
//     with --nsamples and --probes halved, down to COMPRESS_NSAMPLES_FLOOR (4) and COMPRESS_PROBES_FLOOR (2)
//
// Budget for chained techniques: perplexity (WikiText-2) at most +5% and mean zero-shot accuracy at least 95% of
// the uncompressed model. Chains are searched greedily stage by stage on the cheap perplexity check, each stage
// limited to a share of the budget, then the final model gets the full evaluation; a chain that fails it falls
// back to its previous accepted model.

std::string envor(std::string k, std::string d) { auto v = os.env(k); if (v == nullptr) return d; return v; }

std::string FORGE = envor("FORGE_BIN", "/home/user/FORGE/build/release/forge");
std::string MODELS = envor("COMPRESS_MODELS", "/home/user/models/fpk");
std::string DATA = envor("COMPRESS_DATA", "/home/user/models/data");
std::string RUNS = envor("COMPRESS_RUNS", "/home/user/runs");
std::string RES = envor("COMPRESS_RESULTS", "/home/user/AI-Optimisation/results");
std::string CALIB = DATA + "/c4_calib.jsonl";
std::string TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq";
std::string NSAMPLES = envor("COMPRESS_NSAMPLES", "32");
std::string SEQLEN = envor("COMPRESS_SEQLEN", "512");
std::string PPL_WINDOWS = envor("COMPRESS_PPL_WINDOWS", "40");
std::string LIMIT = envor("COMPRESS_LIMIT", "200");
std::string PPL_WINDOW = envor("COMPRESS_PPL_WINDOW", "512");
double BUDGET = 0.05;
int RETRIES = std::stoi(envor("COMPRESS_RETRIES", "3"));
int BACKOFF_S = std::stoi(envor("COMPRESS_BACKOFF", "30"));
int NSAMPLES_FLOOR = std::stoi(envor("COMPRESS_NSAMPLES_FLOOR", "4"));
std::string PROBES = envor("COMPRESS_PROBES", "8");
int PROBES_FLOOR = std::stoi(envor("COMPRESS_PROBES_FLOOR", "2"));
std::string CKPT_MINUTES = envor("COMPRESS_CHECKPOINT_MINUTES", "10");
std::string MEM_LIMIT = envor("COMPRESS_MEM_LIMIT", envor("FORGE_MEM_LIMIT", ""));

os.setenv("FORGE_CPU_INT8", "0");   // exact quantised matmuls in every evaluation (children inherit it)
if (MEM_LIMIT != "") os.setenv("FORGE_MEM_LIMIT", MEM_LIMIT);   // the memory budget of every job
fs.mkdir(RUNS + "/models");
fs.mkdir(RUNS + "/reports");
fs.mkdir(RUNS + "/logs");
fs.mkdir(RUNS + "/ckpt");
fs.mkdir(RES);

// ------------------------------------------------------------------------------------------------ supervision
// One plan at a time: the lock is the OS's, so it goes away with the process (no stale lock after a crash).
auto plan_lock = fs.lock(RUNS + "/plan.lock", false);
if (plan_lock == nullptr) {
    std::println("plan: another instance holds {} (see {}); exiting", RUNS + "/plan.lock", RUNS + "/plan.heartbeat");
    std::exit(75);
}
if (fs.exists(RUNS + "/plan.done")) fs.remove(RUNS + "/plan.done");
std::string current_job = "";
void heartbeat(std::string state) {
    auto h = json.parse("{}");
    h["pid"] = os.pid(); h["time"] = time.now(); h["iso"] = time.iso(time.now()); h["job"] = current_job; h["state"] = state;
    fs.write_atomic(RUNS + "/plan.heartbeat", json.dump(h) + "\n");
}
heartbeat("starting");

// Runs argv, streaming its output (stderr merged) to `log` and keeping the heartbeat fresh. Returns
// {code, tail (the last 64 KB of output), seconds}.
auto run_job(std::vector<std::string> argv, std::string log) {
    std::string cmdline = "";
    for (auto a : argv) cmdline += " " + a;
    fs.append(log, "\n[" + time.iso(time.now()) + "] $" + cmdline + "\n");   // every attempt, appended
    double t0 = time.ms();
    auto p = forge::proc::start(argv);
    std::string tail = "";
    int code = -1;
    heartbeat("running");
    double last_hb = time.ms();
    while (code == -1) {
        std::string s = p.read(30000);
        if (s != "") {
            fs.append(log, s);
            tail += s;
            if (len(tail) > 65536) tail = tail.substr(len(tail) - 65536);
        }
        if (time.ms() - last_hb > 30000) { heartbeat("running"); last_hb = time.ms(); }
        if (s == "") {
            code = p.wait(1000);
            // -1: still running
        }
    }
    for (int i = 0; i < 100; ++i) { std::string s = p.read(100); if (s == "") break; fs.append(log, s); tail += s; }
    auto r = json.parse("{}");
    r["code"] = code; r["tail"] = tail; r["seconds"] = (time.ms() - t0) / 1000.0;
    return r;
}

bool is_oom(auto r) {
    int c = r["code"];
    std::string t = r["tail"];
    return c == 137 || c == 9 || t.contains("bad_alloc") || t.contains("out of memory") || t.contains("Out of memory") || t.contains("Cannot allocate memory") || t.contains("OutOfMemory");
}
bool is_busy(auto r) { std::string t = r["tail"]; return t.contains("is in use by another forge compress"); }   // an orphaned job of a killed plan
void backoff(int attempt) {
    int s = BACKOFF_S;
    for (int i = 0; i < attempt; ++i) s = s * 2;
    std::println("  retrying in {}s", s);
    for (int w = 0; w < s; w += 30) { heartbeat("backoff"); sleep(1000 * std::min(30, s - w)); }
}

// ------------------------------------------------------------------------------------------------ records
auto results = json.parse("{}");
std::vector<std::string> order;
if (fs.exists(RES + "/results.jsonl")) {
    for (auto line : fs.read(RES + "/results.jsonl").split("\n")) {
        if (len(line) < 2) continue;
        auto r = json.parse(line);
        if (!results.contains(r["id"])) order.push_back(r["id"]);
        results[r["id"]] = r;
    }
}
std::println("plan: {} finished jobs on record", len(order));

auto models = std::vector<std::map<std::string, std::string>>{
    {{"name", "qwen3-0.6b"}, {"kind", "dense"}, {"title", "Qwen3-0.6B (dense)"}},
    {{"name", "tinyllama-sparse24"}, {"kind", "sparse"}, {"title", "TinyLlama-1.1B-Chat pruned 2:4 (sparse)"}},
    {{"name", "tinymixtral"}, {"kind", "moe"}, {"title", "TinyMixtral 4x top-2 (MoE)"}}};

std::string safe(std::string id) { std::string o = ""; for (auto c : id) { if (c == '/') o += "__"; else o += c; } return o; }

std::string base_fpk(std::string m) { return MODELS + "/" + m + ".f32.fpk"; }

void record(auto r) {
    std::string id = r["id"];
    if (!results.contains(id)) order.push_back(id);
    results[id] = r;
    fs.append(RES + "/results.jsonl", json.dump(r) + "\n");
}

auto summarise(std::string id, std::string model, std::string parent, std::string recipe, std::string codec, std::string mode, auto rep, double wall) {
    auto r = json.parse("{}");
    r["id"] = id; r["model"] = model; r["parent"] = parent; r["recipe"] = recipe; r["codec"] = codec; r["mode"] = mode;
    r["wall_seconds"] = wall;
    if (rep.contains("after")) { r["counts"] = rep["after"]; r["before"] = rep["before"]; }
    if (rep.contains("counts")) r["counts"] = rep["counts"];
    r["file_bytes"] = rep["file_bytes"];
    if (rep.contains("compress_seconds")) r["compress_seconds"] = rep["compress_seconds"];
    if (rep.contains("eval")) r["eval"] = rep["eval"];
    if (rep.contains("stages")) {
        auto st = json.parse("[]");
        for (auto s : rep["stages"]) {
            auto o = json.parse("{}");
            o["stage"] = s["stage"]; o["seconds"] = s["seconds"];
            if (s.contains("mean_rel_err")) o["mean_rel_err"] = s["mean_rel_err"];
            if (s.contains("removed")) o["removed"] = s["removed"];
            if (s.contains("nm_fraction")) o["nm_fraction"] = s["nm_fraction"];
            if (s.contains("layers")) o["layers"] = s["layers"];
            st.push_back(o);
        }
        r["stages"] = st;
    }
    return r;
}

std::vector<std::string> eval_args(std::string mode) {
    auto a = std::vector<std::string>{"--data", DATA, "--ppl-windows", PPL_WINDOWS, "--ppl-window", PPL_WINDOW, "--mix", "1"};
    if (mode == "full") { a.push_back("--tasks"); a.push_back(TASKS); a.push_back("--limit"); a.push_back(LIMIT); }
    return a;
}

// forge compress INPUT -> RUNS/models/ID.fpk, evaluated; the model file is kept only when asked
auto compress_job(std::string id, std::string model, std::string parent, std::string input, std::string recipe, std::string codec, std::string mode, bool keep, std::string calib = "") {
    std::string out = RUNS + "/models/" + safe(id) + ".fpk";
    if (results.contains(id) && (!keep || fs.exists(out))) return results[id];
    std::string rep = RUNS + "/reports/" + safe(id) + ".json";
    std::string ckpt = RUNS + "/ckpt/" + safe(id);
    std::string logf = RUNS + "/logs/" + safe(id) + ".log";
    int ns = std::stoi(NSAMPLES), pr = std::stoi(PROBES), attempt = 0, ooms = 0, waits = 0;
    double t0 = time.ms();
    current_job = id;
    auto p = json.parse("{}");
    bool again = true;
    while (again) {
        again = false;
        auto argv = std::vector<std::string>{FORGE, "compress", input, "-o", out, "--recipe", recipe, "--calib-data", calib == "" ? CALIB : calib,
                                             "--nsamples", std::to_string(ns), "--probes", std::to_string(pr), "--seqlen", SEQLEN, "--codec", codec, "--report", rep,
                                             "--checkpoint", ckpt, "--checkpoint-minutes", CKPT_MINUTES};
        for (auto x : eval_args(mode)) argv.push_back(x);
        std::println("[{}] {} :: {} ({}, {}, nsamples {}){}", time.iso(time.now()), id, recipe, codec, mode, ns, attempt + ooms > 0 ? std::format(" retry {}", attempt + ooms) : "");
        if (fs.exists(rep)) fs.remove(rep);
        p = run_job(argv, logf);
        if (p["code"] == 0 && fs.exists(rep)) {
            again = false;   // done
        } else if (is_busy(p) && waits < 60) {   // the checkpoint is held by an orphaned job of a killed plan
            ++waits; again = true;
            std::println("  checkpoint in use by another process; waiting");
            for (int w = 0; w < 2; ++w) { heartbeat("waiting"); sleep(30000); }
        } else if (is_oom(p) && (ns > NSAMPLES_FLOOR || pr > PROBES_FLOOR)) {   // memory: retry at once with less calibration data
            ++ooms; again = true;
            ns = std::max(NSAMPLES_FLOOR, ns / 2); pr = std::max(PROBES_FLOOR, pr / 2);
            std::println("  out of memory (exit {}): retrying with --nsamples {} --probes {}", p["code"], ns, pr);
        } else if (attempt < RETRIES) {   // anything else: the same command again later (it resumes from its checkpoint)
            ++attempt; again = true;
            std::println("  failed (exit {}), attempt {}/{}: see {}", p["code"], attempt, RETRIES, logf);
            backoff(attempt - 1);
        }
    }
    current_job = "";
    double wall = (time.ms() - t0) / 1000.0;
    if (p["code"] != 0 || !fs.exists(rep)) {
        std::println("  FAILED (exit {}): see {}", p["code"], logf);
        auto f = json.parse("{}");
        f["id"] = id; f["model"] = model; f["parent"] = parent; f["recipe"] = recipe; f["codec"] = codec; f["mode"] = mode; f["failed"] = true;
        f["exit"] = p["code"]; f["oom"] = is_oom(p); f["nsamples"] = ns; f["attempts"] = attempt + ooms;
        record(f);
        if (fs.exists(out)) fs.remove(out);
        if (fs.exists(ckpt)) fs.remove(ckpt);
        return f;
    }
    auto r = summarise(id, model, parent, recipe, codec, mode, json.parse(fs.read(rep)), wall);
    r["nsamples"] = ns; r["probes"] = pr;
    if (attempt + ooms > 0) { r["retries"] = attempt + ooms; r["oom_retries"] = ooms; }
    if (MEM_LIMIT != "") r["mem_limit"] = MEM_LIMIT;
    record(r);
    if (!keep && fs.exists(out)) fs.remove(out);
    std::println("  ppl {:.3f}  mean_acc {}  params {}  nonzero {}  bytes {}  ({:.0f}s)", double(r["eval"]["ppl"]),
                 r["eval"].contains("mean_acc") ? r["eval"]["mean_acc"] : 0, r["counts"]["params"], r["counts"]["nonzero"], r["file_bytes"], wall);
    return r;
}

auto eval_job(std::string id, std::string model, std::string input, std::string mode) {
    if (results.contains(id)) return results[id];
    std::string rep = RUNS + "/reports/" + safe(id) + ".json";
    auto argv = std::vector<std::string>{FORGE, "eval", input, "--report", rep};
    for (auto x : eval_args(mode)) argv.push_back(x);
    std::println("[{}] {} :: eval {} ({})", time.iso(time.now()), id, input, mode);
    double t0 = time.ms();
    current_job = id;
    auto p = json.parse("{}");
    int attempt = 0;
    bool again = true;
    while (again) {
        p = run_job(argv, RUNS + "/logs/" + safe(id) + ".log");
        again = p["code"] != 0 && attempt < RETRIES;
        if (again) {
            std::println("  failed (exit {}{}), attempt {}/{}", p["code"], is_oom(p) ? ", out of memory" : "", attempt + 1, RETRIES);
            backoff(attempt);
            ++attempt;
        }
    }
    current_job = "";
    if (p["code"] != 0) { std::println("  FAILED (exit {})", p["code"]); auto f = json.parse("{}"); f["id"] = id; f["model"] = model; f["failed"] = true; return f; }
    auto r = summarise(id, model, "", "", "", mode, json.parse(fs.read(rep)), (time.ms() - t0) / 1000.0);
    record(r);
    std::println("  ppl {:.3f}  mean_acc {}", double(r["eval"]["ppl"]), r["eval"].contains("mean_acc") ? r["eval"]["mean_acc"] : 0);
    return r;
}

// COMPRESS_GIT=1: commit and push the results directory after every job (the run may outlive the machine).
std::string GIT_MSG_TAIL = envor("COMPRESS_GIT_TRAILER", "");
void persist(std::string msg) {
    if (envor("COMPRESS_GIT", "0") != "1") return;
    forge::run(std::vector<std::string>{"git", "-C", RES, "add", "-A", "."}, {{"merge", true}});
    auto c = forge::run(std::vector<std::string>{"git", "-C", RES, "commit", "-q", "-m", msg + GIT_MSG_TAIL}, {{"merge", true}});
    if (c.code != 0) return;   // nothing new
    for (int i = 0; i < 4; ++i) {
        auto p = forge::run(std::vector<std::string>{"git", "-C", RES, "push", "-q", "-u", "origin", "HEAD"}, {{"merge", true}, {"timeout", 120000}});
        if (p.code == 0) return;
        sleep(2000 * (i + 1));
    }
    std::println("  (push failed; will retry after the next job)");
}

bool ok(auto r) { return !r.contains("failed") && r.contains("eval"); }

// ------------------------------------------------------------------------------------------------ report
std::string pct(double x) { return std::format("{:+.2f}%", 100.0 * x); }

void write_report() {
    std::string o = "# Compression results (FORGE `forge compress`)\n\n";
    o += "Generated by `tools/compress_plan.cpp` (ForgeScript) from `results.jsonl`. Perplexity: WikiText-2 test, " + PPL_WINDOWS + " windows x 512 tokens; ";
    o += "zero-shot: " + TASKS + " (first " + LIMIT + " of each, fixed order). Calibration: " + NSAMPLES + " x " + SEQLEN + " C4 tokens. ";
    o += "Budget for chains: perplexity <= +5% and mean accuracy >= 95% of the uncompressed model.\n\n";
    o += "Size columns: `params` counts every stored value (tied embeddings once), `nonzero` the non-zero ones, `bytes` the .fpk file.\n\n";
    for (auto& m : models) {
        std::string name = m["name"];
        std::string bid = name + "/base";
        if (!results.contains(bid)) continue;
        auto b = results[bid];
        double bppl = b["eval"]["ppl"];
        double bacc = b["eval"]["mean_acc"];
        double bparams = b["counts"]["params"];
        double bnz = b["counts"]["nonzero"];
        double bbytes = b["file_bytes"];
        o += "## " + std::string(m["title"]) + "\n\n";
        if (b["eval"].contains("ppl_mix")) o += std::format("Baseline with context mixing: ppl {:.3f}. ", double(b["eval"]["ppl_mix"]));
        o += std::format("Baseline: ppl **{:.3f}**, mean acc **{:.4f}**, params {:.1f} M ({:.1f} M non-zero), file {:.0f} MB (f32).", bppl, bacc, bparams / 1e6, bnz / 1e6, bbytes / 1e6);
        std::string accs = "";
        for (auto& [k, v] : b["eval"]["acc"]) accs += std::format(" {} {:.3f};", k, double(v));
        o += " Per task:" + accs + "\n\n";
        // chain
        std::string cid = name + "/chain/final";
        if (results.contains(cid)) {
            auto c = results[cid];
            double cppl = c["eval"]["ppl"];
            double cacc = c["eval"]["mean_acc"];
            o += "### Chain within the 5% budget\n\n";
            o += "Stages: `" + std::string(c["recipe"]) + "`\n\n";
            o += "| | ppl | mean acc | params | non-zero | file | \n|---|---|---|---|---|---|\n";
            o += std::format("| baseline | {:.3f} | {:.4f} | {:.1f} M | {:.1f} M | {:.0f} MB |\n", bppl, bacc, bparams / 1e6, bnz / 1e6, bbytes / 1e6);
            o += std::format("| chained | {:.3f} ({}) | {:.4f} ({}) | {:.1f} M ({}) | {:.1f} M ({}) | {:.0f} MB ({}) |\n\n", cppl, pct(cppl / bppl - 1), cacc, pct(cacc / bacc - 1),
                             double(c["counts"]["params"]) / 1e6, pct(double(c["counts"]["params"]) / bparams - 1), double(c["counts"]["nonzero"]) / 1e6,
                             pct(double(c["counts"]["nonzero"]) / bnz - 1), double(c["file_bytes"]) / 1e6, pct(double(c["file_bytes"]) / bbytes - 1));
            std::string pass = (cppl <= bppl * (1 + BUDGET) && cacc >= bacc * (1 - BUDGET)) ? "**PASS**" : "**FAIL**";
            o += "Budget check: " + pass + std::format(" (ppl {} vs +5.00% allowed, accuracy {} vs -5.00% allowed).", pct(cppl / bppl - 1), pct(cacc / bacc - 1));
            std::string acc2 = "";
            for (auto& [k, v] : c["eval"]["acc"]) acc2 += std::format(" {} {:.3f};", k, double(v));
            o += " Per task:" + acc2 + "\n\n";
        }
        o += "### Search log and single techniques\n\n`ppl+mix`: the same model's perplexity with FORGE's online context mixer (cache, match and temperature models mixed with the network, weights learned online).\n\n";
        o += "| job | recipe | codec | ppl | Δppl | ppl+mix | mean acc | params | non-zero | file MB |\n|---|---|---|---|---|---|---|---|---|---|\n";
        for (auto id : order) {
            auto r = results[id];
            if (r["model"] != name || id == bid) continue;
            if (!ok(r)) { o += "| " + id + " | `" + std::string(r["recipe"]) + "` | " + std::string(r["codec"]) + " | failed | | | | | | |\n"; continue; }
            double p = r["eval"]["ppl"];
            std::string acc = r["eval"].contains("mean_acc") && double(r["eval"]["mean_acc"]) > 0 ? std::format("{:.4f}", double(r["eval"]["mean_acc"])) : "";
            std::string mixs = r["eval"].contains("ppl_mix") && double(r["eval"]["ppl_mix"]) > 0 ? std::format("{:.3f}", double(r["eval"]["ppl_mix"])) : "";
            o += std::format("| {} | `{}` | {} | {:.3f} | {} | {} | {} | {:.1f} M | {:.1f} M | {:.0f} |\n", id, std::string(r["recipe"]), std::string(r["codec"]), p, pct(p / bppl - 1), mixs, acc,
                             double(r["counts"]["params"]) / 1e6, double(r["counts"]["nonzero"]) / 1e6, double(r["file_bytes"]) / 1e6);
        }
        o += "\n";
    }
    fs.write(RES + "/REPORT.md", o);
    persist(std::format("results: {} jobs", len(order)));
}

// ------------------------------------------------------------------------------------------------ phases
std::string phases = len(args) > 0 ? args[len(args) - 1] : "base,chains,sweeps";

if (phases.contains("base")) {
    for (auto& m : models) eval_job(std::string(m["name"]) + "/base", m["name"], base_fpk(m["name"]), "full");
    write_report();
}

// Greedy budgeted chain: stage k may use up to caps[k] of the perplexity budget (cumulative); within a stage the
// options are tried from the most to the least aggressive and the first that fits is kept.
void run_chain(std::string name, std::vector<std::vector<std::string>> stages, std::vector<double> caps, std::vector<std::string> codecs) {
    std::string bid = name + "/base";
    auto b = results[bid];
    double bppl = b["eval"]["ppl"];
    double bacc = b["eval"]["mean_acc"];
    std::string cur = base_fpk(name), cur_id = "", chain = "", prev = "", prev_chain = "";
    for (int k = 0; k < len(stages); ++k) {
        double cap = bppl * (1.0 + BUDGET * caps[k]);
        bool accepted = false;
        for (int j = 0; j < len(stages[k]); ++j) {
            std::string opt = stages[k][j];
            std::string id = name + "/chain/s" + std::to_string(k + 1) + "_" + std::to_string(j + 1);
            auto r = compress_job(id, name, cur_id, cur, opt, codecs[k], "ppl", true);
            std::string out = RUNS + "/models/" + safe(id) + ".fpk";
            if (ok(r) && double(r["eval"]["ppl"]) <= cap) {
                std::println("  chain {}: stage {} keeps {} (ppl {:.3f} <= {:.3f})", name, k + 1, opt, double(r["eval"]["ppl"]), cap);
                if (prev != "" && fs.exists(prev) && prev != base_fpk(name)) fs.remove(prev);
                prev = cur; prev_chain = chain;
                cur = out; cur_id = id;
                chain = chain == "" ? opt : chain + " + " + opt;
                accepted = true;
                break;
            }
            if (fs.exists(out)) fs.remove(out);
        }
        if (!accepted) std::println("  chain {}: stage {} skipped (nothing fits)", name, k + 1);
        write_report();
    }
    // full evaluation of the final model; one step back if the accuracy check fails
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string fid = name + "/chain/final" + (attempt ? "_fallback" : "");
        auto f = eval_job(fid, name, cur, "full");
        if (!ok(f)) break;
        f["recipe"] = chain; f["parent"] = cur_id;
        bool pass = double(f["eval"]["ppl"]) <= bppl * (1 + BUDGET) && double(f["eval"]["mean_acc"]) >= bacc * (1 - BUDGET);
        if (pass || prev == "" || !fs.exists(prev)) { f["id"] = name + "/chain/final"; record(f); break; }
        std::println("  chain {}: final model fails the accuracy check, falling back one stage", name);
        cur = prev; chain = prev_chain; prev = "";
    }
    write_report();
}

if (phases.contains("chains")) {
    // dense: drop sublayers -> narrow FFNs -> 2:4 on the cheapest projections -> GPTQ int8 (2:4 tensors in s24_q8)
    run_chain("qwen3-0.6b", {
        {"depth(n=4,score=probe,shortlist=6,probe=4)", "depth(n=2,score=probe,shortlist=6,probe=4)", "depth(n=1,score=probe,shortlist=6,probe=4)"},
        {"width(s=0.2,score=obs,alloc=eba)", "width(s=0.1,score=obs,alloc=eba)", "width(s=0.05,score=obs,alloc=eba)"},
        {"prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.3,refit=mcr,iters=15)", "prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.15,refit=mcr,iters=15)", "prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.08,refit=mcr,iters=15)"},
        {"optimise(kl=0.02)", "optimise(kl=0.008)", "optimise(kl=0.003)"}},
        {0.25, 0.5, 0.75, 0.92}, {"f32", "f32", "f32", "f32"});
    // sparse (2:4 already): drop sublayers -> narrow FFNs keeping the 2:4 masks -> GPTQ into the 2:4 codecs
    run_chain("tinyllama-sparse24", {
        {"depth(n=4,score=probe,shortlist=6,probe=4)", "depth(n=2,score=probe,shortlist=6,probe=4)", "depth(n=1,score=probe,shortlist=6,probe=4)"},
        {"width(s=0.15,score=obs)", "width(s=0.08,score=obs)", "width(s=0.04,score=obs)"},
        {"optimise(kl=0.02)", "optimise(kl=0.008)", "optimise(kl=0.003)"}},
        {0.3, 0.6, 0.92}, {"f32", "f32", "f32"});
    // MoE: smaller MoE or dense conversion -> drop sublayers -> GPTQ
    run_chain("tinymixtral", {
        {"moe2dense(width=0.5)", "merge(drop=1,method=ls)", "experts(drop=1,score=reap)", "moe2dense(width=0.75)"},
        {"depth(n=2,score=probe,shortlist=6,probe=4)", "depth(n=1,score=probe,shortlist=6,probe=4)"},
        {"optimise(kl=0.02)", "optimise(kl=0.008)", "optimise(kl=0.003)"}},
        {0.5, 0.7, 0.92}, {"f32", "f32", "f32"});
}

if (phases.contains("sweeps")) {
    auto sweep = std::vector<std::vector<std::string>>{
        // data-free optimisation: function-preserving transforms, mixed precision, distribution calibration
        {"qwen3-0.6b", "optimise", "optimise", "f32", "self"},
        {"qwen3-0.6b", "optimise_c4", "optimise", "f32"},
        {"qwen3-0.6b", "optimise_kl02", "optimise(kl=0.02)", "f32", "self"},
        {"qwen3-0.6b", "rtn_q4_plain", "quant(method=rtn,codec=q4_0)", "f32"},
        {"qwen3-0.6b", "rtn_q4_rotated", "rotate+quant(method=rtn,codec=q4_0)", "f32"},
        {"qwen3-0.6b", "rtn_q4_equalized", "equalize+quant(method=rtn,codec=q4_0)", "f32"},
        {"qwen3-0.6b", "gptq_q4_rotated", "rotate+quant(method=gptq,codec=q4_0)", "f32"},
        {"qwen3-0.6b", "gptq_q4_calibrated", "quant(method=gptq,codec=q4_0)+calibrate", "f32"},
        {"qwen3-0.6b", "mixed_unrotated", "quant(method=mixed,kl=0.005)", "f32", "self"},
        {"tinyllama-sparse24", "optimise", "optimise", "f32", "self"},
        {"tinyllama-sparse24", "equalize_gptq_s24q4", "equalize+quant(method=gptq,codec=s24_q4)", "f32"},
        {"tinymixtral", "optimise", "optimise", "f32", "self"},
        {"tinymixtral", "rtn_q4_rotated", "rotate+quant(method=rtn,codec=q4_0)", "f32"},
        // MoE model
        {"tinymixtral", "experts_freq", "experts(drop=1,score=freq)", "f32"},
        {"tinymixtral", "experts_gate", "experts(drop=1,score=gate)", "f32"},
        {"tinymixtral", "experts_reap", "experts(drop=1,score=reap)", "f32"},
        {"tinymixtral", "merge_avg", "merge(drop=1,method=avg)", "f32"},
        {"tinymixtral", "merge_ls", "merge(drop=1,method=ls)", "f32"},
        {"tinymixtral", "moe2dense_50", "moe2dense(width=0.5)", "f32"},
        {"tinymixtral", "moe2dense_25", "moe2dense(width=0.25)", "f32"},
        {"tinymixtral", "sparsegpt50", "prune(method=sparsegpt,s=0.5)", "f32"},
        {"tinymixtral", "rtn_q4", "quant(method=rtn,codec=q4_0)", "f32"},
        {"tinymixtral", "gptq_q4", "quant(method=gptq,codec=q4_0)", "f32"},
        {"tinymixtral", "depth2_probe", "depth(n=2,score=probe,shortlist=6,probe=4)", "f32"},
        // dense model
        {"qwen3-0.6b", "magnitude50", "prune(method=magnitude,s=0.5)", "f32"},
        {"qwen3-0.6b", "wanda50", "prune(method=wanda,s=0.5)", "f32"},
        {"qwen3-0.6b", "ria50", "prune(method=ria,s=0.5)", "f32"},
        {"qwen3-0.6b", "sparsegpt50", "prune(method=sparsegpt,s=0.5)", "f32"},
        {"qwen3-0.6b", "wanda50_mcr", "prune(method=wanda,s=0.5,refit=mcr,iters=15)", "f32"},
        {"qwen3-0.6b", "sparsegpt50_mcr", "prune(method=sparsegpt,s=0.5,refit=mcr,iters=15)", "f32"},
        {"qwen3-0.6b", "wanda50_owl", "prune(method=wanda,s=0.5,alloc=owl)", "f32"},
        {"qwen3-0.6b", "wanda50_eba", "prune(method=wanda,s=0.5,alloc=eba)", "f32"},
        {"qwen3-0.6b", "sparsegpt24", "prune(method=sparsegpt,nm=2:4)", "f32"},
        {"qwen3-0.6b", "sparsegpt24_mcr", "prune(method=sparsegpt,nm=2:4,refit=mcr,iters=15)", "f32"},
        {"qwen3-0.6b", "width20_mag", "width(s=0.2,score=mag)", "f32"},
        {"qwen3-0.6b", "width20_act", "width(s=0.2,score=act)", "f32"},
        {"qwen3-0.6b", "width20_obs", "width(s=0.2,score=obs)", "f32"},
        {"qwen3-0.6b", "width20_obs_eba", "width(s=0.2,score=obs,alloc=eba)", "f32"},
        {"qwen3-0.6b", "depth4_bi", "depth(n=4,score=bi)", "f32"},
        {"qwen3-0.6b", "depth4_probe", "depth(n=4,score=probe,shortlist=6,probe=4)", "f32"},
        {"qwen3-0.6b", "depth4_probe_linear", "depth(n=4,score=probe,shortlist=6,probe=4,replace=linear,rank=256)", "f32"},
        {"qwen3-0.6b", "rtn_q4", "quant(method=rtn,codec=q4_0)", "f32"},
        {"qwen3-0.6b", "gptq_q4", "quant(method=gptq,codec=q4_0)", "f32"},
        {"qwen3-0.6b", "gptq_q4k", "quant(method=gptq,codec=q4_k)", "f32"},
        {"qwen3-0.6b", "gptq_q8", "quant(method=gptq,codec=q8_0,head=rtn)", "f32"},
        // sparse model
        {"tinyllama-sparse24", "store_s24_f16", "none", "s24_f16"},
        {"tinyllama-sparse24", "rtn_s24q8", "quant(method=rtn,codec=s24_q8)", "f32"},
        {"tinyllama-sparse24", "gptq_s24q8", "quant(method=gptq,codec=s24_q8)", "f32"},
        {"tinyllama-sparse24", "rtn_s24q4", "quant(method=rtn,codec=s24_q4)", "f32"},
        {"tinyllama-sparse24", "gptq_s24q4", "quant(method=gptq,codec=s24_q4)", "f32"},
        {"tinyllama-sparse24", "width15_act", "width(s=0.15,score=act)", "f32"},
        {"tinyllama-sparse24", "width15_obs", "width(s=0.15,score=obs)", "f32"},
        {"tinyllama-sparse24", "depth4_bi", "depth(n=4,score=bi)", "f32"},
        {"tinyllama-sparse24", "depth4_probe", "depth(n=4,score=probe,shortlist=6,probe=4)", "f32"},
        {"tinyllama-sparse24", "sparsegpt60", "prune(method=sparsegpt,s=0.6)", "f32"}};
    for (auto s : sweep) {
        std::string name = s[0];
        compress_job(name + "/" + s[1], name, "", base_fpk(name), s[2], s[3], "ppl", false, len(s) > 4 ? s[4] : "");
        write_report();
    }
}

write_report();
heartbeat("done");
fs.write_atomic(RUNS + "/plan.done", time.iso(time.now()) + "\n");   // the supervisor stops on this
std::println("plan: done ({} jobs on record); report in {}", len(order), RES + "/REPORT.md");

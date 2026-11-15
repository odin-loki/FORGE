// ForgeScript: the fast, metric-guided version of the compression plan (tools/compress_plan.cpp takes days at full
// fidelity; this one takes hours). Run with
//
//   forge script tools/compress_fast.cpp [PHASES]        PHASES: any of base,sweeps,chains (default: all, in order)
//
// Metrics (cheap, paired against the uncompressed model, so their noise is far below that of absolute accuracies):
//   - KL(original || compressed) per token over the full vocabulary, and top-1 agreement, on WikiText-2 *train*
//     windows (held out from the test split that perplexity is reported on). KL tracks answer flips closely
//     (Dutta et al. 2024, "Accuracy is Not All You Need") and is EvoPress's search fitness (Sieberling et al. 2024).
//   - dnll: the compressed model's cross-entropy minus the original's on the same train tokens; exp(dnll) - 1 is the
//     perplexity change, so the 5% budget is dnll <= ln 1.05.
//   - Anchor items (tinyBenchmarks / metabench idea): the first ANCHORS items of each zero-shot task (the task files
//     are shuffled with a fixed seed, so that is a random sample); flips and the paired accuracy change are counted
//     against the baseline's answers on the same items.
// Fidelity levels (successive halving: many candidates get the cheap level, few the expensive one):
//   S  screen  KL on KL_S windows                                   ~30 s on the GPU
//   P  sweep   KL on KL_P windows + perplexity on PPL_P test windows ~1-2 min
//   M  check   P + ANCHORS items of each task (flips)               ~5 min
//   F  final   whole test split + FINAL items of each task          ~20-30 min (baselines and chain winners only)
// Devices: compression runs on the CPU (CPU_LANES jobs at once, CPU_THREADS threads each), evaluation on the GPU
// (GPU_LANES) and, when EVAL_CPU_LANES > 0, also on the CPU; a task waits only for its own inputs, so the CPU
// compresses the next candidates while the GPU evaluates the previous ones.
// Search:
//   sweeps  every single technique of compress_plan.cpp at level P; the per-model Pareto front over (KL, estimated
//           size) is promoted to level M; dominated model files are deleted as soon as they are dominated. Tables:
//           technique, KL, top-1, dnll, dppl, size, efficiency (size saved per unit KL), Pareto flag, flips, dacc;
//           and the proxy check: Spearman correlation of KL with dppl and with flips over the M points.
//   chains  stage-by-stage beam search (BEAM states): each state is expanded with every option of the stage and
//           "skip"; an option whose single-technique KL in the sweep table already exceeds the stage's budget is not
//           tried (table pruning); candidates are screened at level S, those within the stage's cumulative dnll cap
//           are ranked by estimated size (then KL) and the best BEAM survive. The leaves get level M; the smallest
//           that keeps dnll <= ln 1.05 and anchor accuracy >= 95% (paired) gets level F and the PASS/FAIL verdict.

std::string envor(std::string k, std::string d) { auto v = os.env(k); if (v == nullptr) return d; return v; }

std::string FORGE = envor("FORGE_BIN", "forge");
std::string MODELS = envor("COMPRESS_MODELS", "/home/user/models/fpk");
std::string DATA = envor("COMPRESS_DATA", "/home/user/models/data");
std::string RUNS = envor("COMPRESS_RUNS", "/home/user/runs_fast");
std::string RES = envor("COMPRESS_RESULTS", "/home/user/AI-Optimisation/results");
std::string CALIB = DATA + "/c4_calib.jsonl";
std::string TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq";
std::string NSAMPLES = envor("COMPRESS_NSAMPLES", "32");
std::string SEQLEN = envor("COMPRESS_SEQLEN", "512");
std::string PROBES = envor("COMPRESS_PROBES", "8");
std::string KL_S = envor("FAST_KL_S", "8");
std::string KL_P = envor("FAST_KL_P", "16");
std::string PPL_P = envor("FAST_PPL_P", "40");
std::string ANCHORS = envor("FAST_ANCHORS", "100");
std::string FINAL = envor("FAST_FINAL", "500");
std::string PPL_F = envor("FAST_PPL_F", "0");
int CPU_LANES = std::stoi(envor("FAST_CPU_LANES", "2"));
std::string CPU_THREADS = envor("FAST_CPU_THREADS", "32");
int GPU_LANES = std::stoi(envor("FAST_GPU_LANES", "1"));
int EVAL_CPU_LANES = std::stoi(envor("FAST_EVAL_CPU_LANES", "0"));
int BEAM = std::stoi(envor("FAST_BEAM", "2"));
double BUDGET = 0.05;
double DNLL_BUDGET = std::log(1.0 + BUDGET);

fs.mkdir(RUNS); fs.mkdir(RUNS + "/models"); fs.mkdir(RUNS + "/reports"); fs.mkdir(RUNS + "/logs"); fs.mkdir(RES);
// one lock per model set: the chains of different models may run as separate processes at the same time
std::string LOCK_TAG = envor("FAST_MODELS", "") == "" ? "" : "_" + envor("FAST_MODELS", "").replace(",", "_");
auto plan_lock = fs.lock(RUNS + "/plan" + LOCK_TAG + ".lock", false);
if (plan_lock == nullptr) { std::println("fast plan: another instance holds {}; exiting", RUNS + "/plan.lock"); std::exit(75); }

// device environments: exact kernels everywhere (no int8 activations, f32 SIMT attention on the GPU)
std::vector<std::string> GPU_ENV = {"env", "FORGE_DEVICE=cuda", "FORGE_FA_TC=0", "FORGE_GPU_QMATMUL=0", "FORGE_CPU_INT8=0"};
// FAST_GPU_WRAP: a command that runs the GPU job while holding a slot shared with other plan processes (device memory)
if (envor("FAST_GPU_WRAP", "") != "") { auto g = std::vector<std::string>{envor("FAST_GPU_WRAP", "")}; for (auto x : GPU_ENV) g.push_back(x); GPU_ENV = g; }
std::vector<std::string> CPU_ENV = {"env", "FORGE_DEVICE=cpu", "FORGE_CPU_INT8=0", "FORGE_THREADS=" + CPU_THREADS};

std::string safe(std::string id) { std::string o = ""; for (auto c : id) { if (c == '/' || c == '@') o += "__"; else o += c; } return o; }
std::string base_fpk(std::string m) { return MODELS + "/" + m + ".f32.fpk"; }
std::string model_out(std::string id) { return RUNS + "/models/" + safe(id) + ".fpk"; }

// ------------------------------------------------------------------------------------------------ records
auto results = json.parse("{}");
std::vector<std::string> order;
std::string RESULTS = RES + "/results_fast.jsonl";
if (fs.exists(RESULTS)) {
    for (auto line : fs.read(RESULTS).split("\n")) {
        if (len(line) < 2) continue;
        auto r = json.parse(line);
        if (!results.contains(r["key"])) order.push_back(r["key"]);
        results[r["key"]] = r;
    }
}
std::println("fast plan: {} finished evaluations on record", len(order));
void record(auto r) {
    std::string k = r["key"];
    if (!results.contains(k)) order.push_back(k);
    results[k] = r;
    fs.append(RESULTS, json.dump(r) + "\n");
}

// ------------------------------------------------------------------------------------------------ scheduler
// A task: {"name", "lane" ("cpu" | "gpu" | "eval" = gpu or cpu), "argv" [...], "log", "after" [names], "retries"}.
// run_tasks starts every task whose inputs are done as soon as a lane is free, and returns {name: exit code}.
auto run_tasks(auto tasks) {
    auto done = json.parse("{}");
    auto started = json.parse("{}");
    auto procs = std::vector<auto>{};
    std::vector<std::string> pnames;
    std::vector<std::string> plane;
    std::vector<double> pt0;
    int busy_cpu = 0, busy_gpu = 0, busy_ecpu = 0;
    double t_start = time.ms(), last_hb = 0;
    int total = len(tasks);
    while (len(done) < total) {
        // start what can start
        for (auto& t : tasks) {
            std::string nm = t["name"];
            if (started.contains(nm)) continue;
            bool ready = true;
            if (t.contains("after")) for (auto a : t["after"]) if (!done.contains(std::string(a))) ready = false;
            if (!ready) continue;
            bool failed_dep = false;
            if (t.contains("after")) for (auto a : t["after"]) if (done.contains(std::string(a)) && int(done[std::string(a)]) != 0) failed_dep = true;
            if (failed_dep) { started[nm] = true; done[nm] = 99; std::println("  skip {} (an input failed)", nm); continue; }
            std::string lane = t["lane"], use = "";
            if (lane == "cpu" && busy_cpu < CPU_LANES) use = "cpu";
            if ((lane == "gpu" || lane == "eval") && busy_gpu < GPU_LANES) use = "gpu";
            if (use == "" && lane == "eval" && busy_ecpu < EVAL_CPU_LANES) use = "ecpu";
            if (use == "") continue;
            auto argv = std::vector<std::string>{};
            auto envv = use == "gpu" ? GPU_ENV : CPU_ENV;
            for (auto e : envv) argv.push_back(e);
            for (auto a : t["argv"]) argv.push_back(std::string(a));
            std::string cmd = "";
            for (auto a : argv) cmd += " '" + a + "'";
            std::string log = t["log"];
            fs.append(log, "\n[" + time.iso(time.now()) + "] (" + use + ")" + cmd + "\n");
            procs.push_back(forge::proc::start(std::vector<std::string>{"sh", "-c", "exec" + cmd + " >> '" + log + "' 2>&1"}));
            pnames.push_back(nm); plane.push_back(use); pt0.push_back(time.ms());
            started[nm] = true;
            if (use == "cpu") ++busy_cpu; else if (use == "gpu") ++busy_gpu; else ++busy_ecpu;
            std::println("[{}] start {} on {}", time.iso(time.now()), nm, use);
        }
        // poll the running ones
        bool any = false;
        for (int i = 0; i < len(procs); ++i) {
            if (pnames[i] == "") continue;
            any = true;
            int c = procs[i].wait(200);
            if (c == -1) continue;
            std::string nm = pnames[i];
            std::println("[{}] done  {} on {}: exit {} ({:.0f}s)", time.iso(time.now()), nm, plane[i], c, (time.ms() - pt0[i]) / 1000.0);
            if (plane[i] == "cpu") --busy_cpu; else if (plane[i] == "gpu") --busy_gpu; else --busy_ecpu;
            pnames[i] = "";
            // one retry for a failed task
            bool retried = false;
            if (c != 0) for (auto& t : tasks) if (std::string(t["name"]) == nm && !t.contains("retried")) { t["retried"] = true; started.erase(nm); retried = true; std::println("  retrying {}", nm); }
            if (!retried) done[nm] = c;
        }
        if (time.ms() - last_hb > 30000) {
            last_hb = time.ms();
            fs.write_atomic(RUNS + "/plan" + LOCK_TAG + ".heartbeat", std::format("{{\"pid\":{},\"iso\":\"{}\",\"done\":{},\"total\":{}}}\n", os.pid(), time.iso(time.now()), len(done), total));
        }
        if (!any) sleep(200);
    }
    std::println("  batch of {} tasks in {:.1f} min", total, (time.ms() - t_start) / 60000.0);
    return done;
}

// ------------------------------------------------------------------------------------------------ jobs and metrics
std::vector<std::string> level_args(std::string level, std::string model) {
    auto a = std::vector<std::string>{"--data", DATA, "--kl-ref", base_fpk(model)};
    if (level == "S") { a.push_back("--kl-windows"); a.push_back(KL_S); a.push_back("--ppl-windows"); a.push_back("-1"); }
    if (level == "P" || level == "M") { a.push_back("--kl-windows"); a.push_back(KL_P); a.push_back("--ppl-windows"); a.push_back(PPL_P); a.push_back("--mix"); a.push_back("1"); }
    if (level == "M") { a.push_back("--tasks"); a.push_back(TASKS); a.push_back("--limit"); a.push_back(ANCHORS); }
    if (level == "F") { a.push_back("--kl-windows"); a.push_back(KL_P); a.push_back("--ppl-windows"); a.push_back(PPL_F); a.push_back("--mix"); a.push_back("1");
                        a.push_back("--tasks"); a.push_back(TASKS); a.push_back("--limit"); a.push_back(FINAL); }
    return a;
}

// a candidate: {"id", "model", "parent" (input file), "recipe", "codec", "calib"}; tasks to compress it (CPU) and
// evaluate it at `level` (GPU); skipped when already on record
std::string ekey(std::string id, std::string level) { return id + "@" + level; }
void add_tasks(auto& tasks, auto c, std::string level, bool need_compress) {
    std::string id = c["id"], m = c["model"];
    std::string out = std::string(c["recipe"]) == "" ? std::string(c["parent"]) : model_out(id);
    if (results.contains(ekey(id, level))) return;
    if (fs.exists(RUNS + "/reports/" + safe(ekey(id, level)) + ".json")) return;   // evaluated by an interrupted run: collect() reads it
    if (need_compress && !fs.exists(out)) {
        std::string calib = c.contains("calib") && std::string(c["calib"]) != "" ? std::string(c["calib"]) : CALIB;
        auto t = json.parse("{}");
        t["name"] = "c:" + id; t["lane"] = "cpu"; t["log"] = RUNS + "/logs/" + safe(id) + ".log";
        auto argv = json.parse("[]");
        for (auto x : std::vector<std::string>{FORGE, "compress", std::string(c["parent"]), "-o", out, "--recipe", std::string(c["recipe"]), "--calib-data", calib,
                                               "--nsamples", NSAMPLES, "--probes", PROBES, "--seqlen", SEQLEN, "--codec", std::string(c["codec"]),
                                               "--report", RUNS + "/reports/" + safe(id) + ".compress.json", "--checkpoint", RUNS + "/ckpt_" + safe(id)}) argv.push_back(x);
        t["argv"] = argv;
        bool have = false; for (auto& u : tasks) if (std::string(u["name"]) == std::string(t["name"])) have = true;
        if (!have) tasks.push_back(t);
    }
    auto e = json.parse("{}");
    e["name"] = "e:" + ekey(id, level); e["lane"] = "eval"; e["log"] = RUNS + "/logs/" + safe(id) + ".log";
    auto argv = json.parse("[]");
    argv.push_back(FORGE); argv.push_back("eval"); argv.push_back(out); argv.push_back("--report"); argv.push_back(RUNS + "/reports/" + safe(ekey(id, level)) + ".json");
    for (auto x : level_args(level, m)) argv.push_back(x);
    e["argv"] = argv;
    auto after = json.parse("[]");
    for (auto& u : tasks) if (std::string(u["name"]) == "c:" + id) after.push_back("c:" + id);
    e["after"] = after;
    tasks.push_back(e);
}

// first number under `key` anywhere inside v (stage reports nest the allocation)
bool is_num(auto v) { std::string t = forge::typeof(v); return t == "float" || t == "int"; }
double find_num(auto v, std::string key, double dflt) {
    std::string t = forge::typeof(v);
    if (t == "map") {
        for (auto& [k, x] : v) { if (k == key && is_num(x)) return double(x); }
        for (auto& [k, x] : v) { double r = find_num(x, key, -1.0); if (r >= 0) return r; }
    } else if (t == "list") {
        for (auto x : v) { double r = find_num(x, key, -1.0); if (r >= 0) return r; }
    }
    return dflt;
}

// folds the compress and eval reports into one record keyed id@level
auto collect(auto c, std::string level) {
    std::string id = c["id"], m = c["model"];
    std::string k = ekey(id, level);
    if (results.contains(k)) return results[k];
    auto r = json.parse("{}");
    r["key"] = k; r["id"] = id; r["model"] = m; r["level"] = level; r["recipe"] = c["recipe"]; r["codec"] = c["codec"];
    if (c.contains("family")) r["family"] = c["family"];
    if (c.contains("chain")) r["chain"] = c["chain"];
    std::string erep = RUNS + "/reports/" + safe(k) + ".json";
    if (!fs.exists(erep)) { r["failed"] = true; record(r); return r; }
    auto ev = json.parse(fs.read(erep));
    r["eval"] = ev["eval"]; r["counts"] = ev["counts"]; r["file_bytes"] = ev["file_bytes"];
    // bits per body weight: the codec the stages quantised to (bits_per_weight / body_bits_per_weight), else f32
    double bpw = 32;
    std::string crep = RUNS + "/reports/" + safe(id) + ".compress.json";
    if (fs.exists(crep)) {
        auto cr = json.parse(fs.read(crep));
        r["compress_seconds"] = cr["compress_seconds"];
        double b1 = find_num(cr["stages"], "body_bits_per_weight", -1), b2 = find_num(cr["stages"], "bits_per_weight", -1);
        if (b1 > 0) bpw = b1; else if (b2 > 0) bpw = b2;
    } else if (c.contains("bpw")) bpw = double(c["bpw"]);
    if (c.contains("bpw_in") && bpw == 32) bpw = double(c["bpw_in"]);   // a chain stage that did not quantise keeps its input's bits
    r["bpw"] = bpw;
    double bnz = r["counts"]["body_nonzero"], nz = r["counts"]["nonzero"];
    r["est_mb"] = (bnz * bpw / 8.0 + (nz - bnz) * 2.0) / 1e6;   // body at its codec, embeddings / norms at f16
    if (ev["eval"].contains("kl")) r["dnll"] = double(ev["eval"]["kl"]["nll"]) - double(ev["eval"]["kl"]["nll_ref"]);
    record(r);
    return r;
}

// gold labels of the first FINAL items of each task (the evaluator reads the files in order)
auto labels = json.parse("{}");
std::string labels_of(std::string t) {
    if (labels.contains(t)) return labels[t];
    std::string s = "";
    int lim = std::stoi(FINAL);
    for (auto line : fs.read(DATA + "/tasks/" + t + ".jsonl").split("\n")) {
        if (len(line) < 2 || len(s) >= lim) continue;
        auto it = json.parse(line);
        s += std::to_string(int(it["label"]));
    }
    labels[t] = s;
    return s;
}

// flips, and the paired accuracy change: both models scored on the same items (the first min(n) of each task)
auto paired(auto r, auto b) {
    auto o = json.parse("{}");
    if (!r.contains("eval") || !r["eval"].contains("preds") || !b.contains("eval") || !b["eval"].contains("preds")) return o;
    double flips = 0, n = 0, dacc_sum = 0;
    int ntasks = 0;
    for (auto& [t, p] : r["eval"]["preds"]) {
        if (!b["eval"]["preds"].contains(t)) continue;
        std::string a = p, q = b["eval"]["preds"][t], g = labels_of(t);
        int m = std::min(std::min(len(a), len(q)), len(g));
        if (m == 0) continue;
        double f = 0, ca = 0, cb = 0;
        for (int i = 0; i < m; ++i) { if (a[i] != q[i]) f += 1; if (a[i] == g[i]) ca += 1; if (q[i] == g[i]) cb += 1; }
        flips += f; n += m;
        dacc_sum += (ca - cb) / m;
        ++ntasks;
    }
    if (n > 0) { o["flips"] = flips / n; o["items"] = n; }
    if (ntasks > 0) o["dacc"] = dacc_sum / ntasks;
    return o;
}

// ------------------------------------------------------------------------------------------------ models
auto models = std::vector<std::map<std::string, std::string>>{
    {{"name", "qwen3-0.6b"}, {"kind", "dense"}, {"title", "Qwen3-0.6B (dense)"}},
    {{"name", "tinyllama-sparse24"}, {"kind", "sparse"}, {"title", "TinyLlama-1.1B-Chat pruned 2:4 (sparse)"}},
    {{"name", "tinymixtral"}, {"kind", "moe"}, {"title", "TinyMixtral 4x top-2 (MoE)"}}};

std::string phases = len(args) > 0 ? args[len(args) - 1] : "base,sweeps,chains";
// FAST_MODELS=a,b: only these models (smoke tests); FAST_SWEEP_MAX=N: at most N sweep entries per model
std::string ONLY = envor("FAST_MODELS", "");
int SWEEP_MAX = std::stoi(envor("FAST_SWEEP_MAX", "1000"));
bool wanted(std::string m) { return ONLY == "" || ("," + ONLY + ",").contains("," + m + ","); }
{
    auto keep = std::vector<std::map<std::string, std::string>>{};
    for (auto& m : models) if (wanted(m["name"])) keep.push_back(m);
    models = keep;
}

// baselines: level F (their anchor answers are the reference for flips; the first ANCHORS of FINAL items) and a
// self-check at level S (KL of a model with itself must be 0)
auto base_c(std::string m) { auto c = json.parse("{}"); c["id"] = m + "/base"; c["model"] = m; c["parent"] = base_fpk(m); c["recipe"] = ""; c["codec"] = "f32"; c["bpw"] = 32; return c; }

// the single techniques of compress_plan.cpp: {model, name, recipe, codec[, calib]}
auto sweep = std::vector<std::vector<std::string>>{
    {"qwen3-0.6b", "wanda50_alpha", "prune(method=wanda,s=0.5,alloc=alpha)", "f32"},
    {"tinymixtral", "wanda50_alpha", "prune(method=wanda,s=0.5,alloc=alpha)", "f32"},
    {"qwen3-0.6b", "lowrank_mp", "lowrank(method=mp,scope=all)", "f32"},
    {"qwen3-0.6b", "lowrank_energy95", "lowrank(method=energy,keep=0.95,scope=all)", "f32"},
    {"qwen3-0.6b", "lowrank_whiten90", "lowrank(method=whiten,keep=0.9,scope=attn)", "f32"},
    {"qwen3-0.6b", "lowrank_mdl", "lowrank(method=mdl,scope=all)", "f32"},
    {"tinymixtral", "lowrank_whiten90", "lowrank(method=whiten,keep=0.9,scope=attn)", "f32"},
    {"qwen3-0.6b", "share2", "share(group=2,keep=0.95,scope=attn,data=1)", "f32"},
    {"qwen3-0.6b", "heads25_obs", "heads(s=0.25,score=obs)", "f32"},
    {"qwen3-0.6b", "heads25_mag", "heads(s=0.25,score=mag)", "f32"},
    {"tinyllama-sparse24", "heads25_obs", "heads(s=0.25,score=obs)", "f32"},
    {"qwen3-0.6b", "permute_sgpt24", "permute+prune(method=sparsegpt,nm=2:4)", "f32"},
    {"qwen3-0.6b", "actsparse40", "actsparse(p=0.4)", "f32"},
    {"qwen3-0.6b", "actsparse40_eba", "actsparse(p=0.4,alloc=eba)", "f32"},
    {"tinyllama-sparse24", "actsparse40", "actsparse(p=0.4)", "f32"},
    {"qwen3-0.6b", "optimise_jacobi", "optimise", "f32", "self"},
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
    {"tinymixtral", "experts_freq", "experts(drop=1,score=freq)", "f32"},
    {"tinymixtral", "experts_gate", "experts(drop=1,score=gate)", "f32"},
    {"tinymixtral", "experts_reap", "experts(drop=1,score=reap)", "f32"},
    {"tinymixtral", "merge_avg", "merge(drop=1,method=avg)", "f32"},
    {"tinymixtral", "merge_ls", "merge(drop=1,method=ls)", "f32"},
    {"tinymixtral", "moe2dense_50", "moe2dense(width=0.5)", "f32"},
    {"tinymixtral", "moe2dense_25", "moe2dense(width=0.25)", "f32"},
    {"tinymixtral", "moe2dense_75", "moe2dense(width=0.75)", "f32"},
    {"tinymixtral", "sparsegpt50", "prune(method=sparsegpt,s=0.5)", "f32"},
    {"tinymixtral", "rtn_q4", "quant(method=rtn,codec=q4_0)", "f32"},
    {"tinymixtral", "gptq_q4", "quant(method=gptq,codec=q4_0)", "f32"},
    {"tinymixtral", "depth2_probe", "depth(n=2,score=probe,shortlist=6,probe=4)", "f32"},
    {"tinymixtral", "depth1_probe", "depth(n=1,score=probe,shortlist=6,probe=4)", "f32"},
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

std::string family_of(std::string recipe) {
    for (auto f : std::vector<std::string>{"optimise", "moe2dense", "experts", "merge", "lowrank", "share", "heads", "actsparse", "width", "depth", "permute", "prune", "quant", "rotate", "equalize"})
        if (recipe.contains(f)) return f;
    return "store";
}

// the Pareto front of one model's records at `level` over (KL, est_mb); returns ids on the front
std::vector<std::string> pareto(std::string m, std::string level) {
    std::vector<std::string> ids;
    for (auto k : order) {
        auto r = results[k];
        if (std::string(r["model"]) != m || std::string(r["level"]) != level || r.contains("failed") || !r.contains("eval") || !r["eval"].contains("kl")) continue;
        if (std::string(r["id"]).contains("/chain/")) continue;
        bool dom = false;
        for (auto k2 : order) {
            auto q = results[k2];
            if (k2 == k || std::string(q["model"]) != m || std::string(q["level"]) != level || q.contains("failed") || !q.contains("eval") || !q["eval"].contains("kl")) continue;
            if (std::string(q["id"]).contains("/chain/")) continue;
            double a1 = r["eval"]["kl"]["kl"], a2 = r["est_mb"], b1 = q["eval"]["kl"]["kl"], b2 = q["est_mb"];
            if (b1 <= a1 && b2 <= a2 && (b1 < a1 || b2 < a2)) dom = true;
        }
        if (!dom) ids.push_back(r["id"]);
    }
    return ids;
}

// Spearman rank correlation of two equal-length lists
double spearman(std::vector<double> x, std::vector<double> y) {
    int n = len(x);
    if (n < 3) return 0;
    auto rank = [&](std::vector<double> v) {
        std::vector<double> rk;
        for (int i = 0; i < n; ++i) { double r = 0, eq = 0; for (int j = 0; j < n; ++j) { if (v[j] < v[i]) r += 1; if (v[j] == v[i]) eq += 1; } rk.push_back(r + (eq - 1) / 2.0); }
        return rk;
    };
    auto rx = rank(x), ry = rank(y);
    double mx = 0, my = 0; for (int i = 0; i < n; ++i) { mx += rx[i]; my += ry[i]; } mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0;
    for (int i = 0; i < n; ++i) { sxy += (rx[i] - mx) * (ry[i] - my); sxx += (rx[i] - mx) * (rx[i] - mx); syy += (ry[i] - my) * (ry[i] - my); }
    return sxx > 0 && syy > 0 ? sxy / std::sqrt(sxx * syy) : 0;
}

// ------------------------------------------------------------------------------------------------ report
std::string pct(double x) { return std::format("{:+.2f}%", 100.0 * x); }
std::string num(auto v, std::string fmt) {
    if (!is_num(v)) return "";
    double x = v;
    if (fmt == "{:.4f}") return std::format("{:.4f}", x);
    if (fmt == "{:.3f}") return std::format("{:.3f}", x);
    if (fmt == "{:+.4f}") return std::format("{:+.4f}", x);
    return std::format("{:.0f}", x);
}

void write_report() {
    std::string o = "# Fast metric-guided compression results (FORGE `tools/compress_fast.cpp`)\n\n";
    o += "Metrics: **KL** = KL(original || compressed) per token, full vocabulary, WikiText-2 *train* windows (" + KL_P + " x 512 at level P/M/F, " + KL_S + " at S); ";
    o += "**top-1** = argmax agreement with the original; **dnll** = cross-entropy change on those tokens (exp(dnll)-1 = perplexity change; 5% budget = 0.0488); ";
    o += "**dppl** = WikiText-2 *test* perplexity change (" + PPL_P + " windows at P/M, whole split at F); **flips** = share of zero-shot answers that differ from the baseline's on the same items (" + ANCHORS + " anchors per task at M, " + FINAL + " at F); ";
    o += "**dacc** = mean change of task accuracy. **est MB** = non-zero body weights at the bits per weight of their codec + other tensors at f16. Calibration: " + NSAMPLES + " x " + SEQLEN + " C4 tokens.\n\n";
    for (auto& m : models) {
        std::string name = m["name"];
        std::string bk = ekey(name + "/base", "F");
        if (!results.contains(bk)) continue;
        auto b = results[bk];
        double bppl = b["eval"]["ppl"], bmb = b["est_mb"];
        // the same PPL_P test windows as the P / M levels
        std::string bpk = ekey(name + "/base", "P");
        double bppl_p = results.contains(bpk) && results[bpk].contains("eval") ? double(results[bpk]["eval"]["ppl"]) : bppl;
        o += "## " + std::string(m["title"]) + "\n\n";
        o += std::format("Baseline (level F): ppl **{:.3f}** (with context mixing {:.3f}), mean acc **{:.4f}**, est {:.0f} MB; per task:", bppl, double(b["eval"]["ppl_mix"]), double(b["eval"]["mean_acc"]), bmb);
        for (auto& [k, v] : b["eval"]["acc"]) o += std::format(" {} {:.3f};", k, double(v));
        o += "\n\n";
        // chain verdict
        std::string ck = ekey(name + "/chain/final", "F");
        if (results.contains(ck) && results[ck].contains("eval")) {
            auto c = results[ck];
            double cppl = c["eval"]["ppl"], cacc = c["eval"]["mean_acc"], bacc = b["eval"]["mean_acc"];
            bool pass = cppl <= bppl * (1 + BUDGET) && cacc >= bacc * (1 - BUDGET);
            auto pr = paired(c, b);
            o += "### Chain within the 5% budget (beam search, level F)\n\n`" + std::string(c["chain"]) + "`\n\n";
            o += "| | ppl | mean acc | KL | top-1 | flips | params | non-zero | est MB |\n|---|---|---|---|---|---|---|---|---|\n";
            o += std::format("| baseline | {:.3f} | {:.4f} | 0 | 1 | 0 | {:.1f} M | {:.1f} M | {:.0f} |\n", bppl, bacc, double(b["counts"]["params"]) / 1e6, double(b["counts"]["nonzero"]) / 1e6, bmb);
            o += std::format("| chain | {:.3f} ({}) | {:.4f} ({}) | {:.4f} | {:.3f} | {} | {:.1f} M | {:.1f} M | {:.0f} ({}) |\n\n", cppl, pct(cppl / bppl - 1), cacc, pct(cacc / bacc - 1),
                             double(c["eval"]["kl"]["kl"]), double(c["eval"]["kl"]["top1"]), pr.contains("flips") ? std::format("{:.3f}", double(pr["flips"])) : "",
                             double(c["counts"]["params"]) / 1e6, double(c["counts"]["nonzero"]) / 1e6, double(c["est_mb"]), pct(double(c["est_mb"]) / bmb - 1));
            o += std::string("Budget check: ") + (pass ? "**PASS**" : "**FAIL**") + std::format(" (ppl {} vs +5% allowed, accuracy {} vs -5% allowed).\n\n", pct(cppl / bppl - 1), pct(cacc / bacc - 1));
        }
        // chain search log
        bool hdr = false;
        for (auto k : order) {
            auto r = results[k];
            if (std::string(r["model"]) != name || !std::string(r["id"]).contains("/chain/") || r.contains("failed") || !r.contains("eval")) continue;
            if (!hdr) { o += "### Chain search log\n\n| candidate | chain | level | KL | top-1 | dnll | dppl | flips | est MB | kept |\n|---|---|---|---|---|---|---|---|---|---|\n"; hdr = true; }
            auto pr = paired(r, b);
            o += "| " + std::string(r["id"]) + " | `" + (r.contains("chain") ? std::string(r["chain"]) : std::string(r["recipe"])) + "` | " + std::string(r["level"]) + " | "
               + num(r["eval"]["kl"]["kl"], "{:.4f}") + " | " + num(r["eval"]["kl"]["top1"], "{:.3f}") + " | " + num(r["dnll"], "{:+.4f}") + " | "
               + (double(r["eval"]["ppl"]) > 0 ? pct(double(r["eval"]["ppl"]) / (std::string(r["level"]) == "F" ? bppl : bppl_p) - 1) : "") + " | " + (pr.contains("flips") ? std::format("{:.3f}", double(pr["flips"])) : "") + " | "
               + num(r["est_mb"], "{:.0f}") + " | " + (r.contains("kept") ? std::string(r["kept"]) : "") + " |\n";
        }
        if (hdr) o += "\n";
        // sweep table
        auto front = pareto(name, "P");
        o += "### Single techniques (level P; Pareto front over KL and size promoted to M)\n\n";
        o += "| technique | recipe | KL | top-1 | dnll | dppl | ppl+mix | est MB | MB saved / KL | Pareto | flips (M) | dacc (M) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n";
        std::vector<double> xs, ys, fx, fy;
        for (auto k : order) {
            auto r = results[k];
            if (std::string(r["model"]) != name || std::string(r["level"]) != "P" || std::string(r["id"]).contains("/chain/") || std::string(r["id"]).contains("/base")) continue;
            if (r.contains("failed") || !r.contains("eval")) { o += "| " + std::string(r["id"]) + " | `" + std::string(r["recipe"]) + "` | failed | | | | | | | | | |\n"; continue; }
            double kl = r["eval"]["kl"]["kl"], mb = r["est_mb"], p = r["eval"]["ppl"];
            bool onf = false; for (auto f : front) if (f == std::string(r["id"])) onf = true;
            std::string mk = ekey(std::string(r["id"]), "M");
            std::string fl = "", da = "";
            if (results.contains(mk) && results[mk].contains("eval")) {
                auto pr = paired(results[mk], b);
                if (pr.contains("flips")) { fl = std::format("{:.3f}", double(pr["flips"])); da = pct(double(pr["dacc"])); fx.push_back(kl); fy.push_back(pr["flips"]); }
            }
            xs.push_back(kl); ys.push_back(p / bppl_p - 1);
            o += std::format("| {} | `{}` | {:.4f} | {:.3f} | {:+.4f} | {} | {:.3f} | {:.0f} | {} | {} | {} | {} |\n", std::string(r["id"]), std::string(r["recipe"]), kl, double(r["eval"]["kl"]["top1"]),
                             double(r["dnll"]), pct(p / bppl_p - 1), double(r["eval"]["ppl_mix"]), mb, kl > 0 ? std::format("{:.0f}", (bmb - mb) / kl) : "", onf ? "yes" : "", fl, da);
        }
        o += std::format("\nProxy check ({}): Spearman(KL, dppl) = **{:.3f}** over {} techniques; Spearman(KL, flips) = **{:.3f}** over {} anchor-evaluated techniques.\n\n",
                         name, spearman(xs, ys), len(xs), spearman(fx, fy), len(fx));
    }
    fs.write(RES + "/REPORT_FAST" + LOCK_TAG + ".md", o);
}

// ------------------------------------------------------------------------------------------------ phase: base + sweeps
// Baselines and sweeps share one task graph so the CPU compresses sweep models while the GPU evaluates the baselines.
if (phases.contains("base") || phases.contains("sweeps")) {
    auto tasks = json.parse("[]");
    auto cands = std::vector<auto>{};
    if (phases.contains("base")) for (auto& m : models) { auto c = base_c(m["name"]); add_tasks(tasks, c, "F", false); add_tasks(tasks, c, "P", false); add_tasks(tasks, c, "S", false); cands.push_back(c); }
    if (phases.contains("sweeps")) {
        auto per = json.parse("{}");
        for (auto s : sweep) {
            if (!wanted(s[0])) continue;
            per[s[0]] = per.contains(s[0]) ? int(per[s[0]]) + 1 : 1;
            if (int(per[s[0]]) > SWEEP_MAX) continue;
            auto c = json.parse("{}");
            c["id"] = s[0] + "/" + s[1]; c["model"] = s[0]; c["parent"] = base_fpk(s[0]); c["recipe"] = s[2]; c["codec"] = s[3]; c["family"] = family_of(s[2]);
            if (len(s) > 4) c["calib"] = s[4] == "self" ? "self" : s[4];
            add_tasks(tasks, c, "P", true);
            cands.push_back(c);
        }
    }
    std::println("fast plan: base + sweeps: {} tasks", len(tasks));
    run_tasks(tasks);
    for (auto c : cands) { collect(c, "P"); if (std::string(c["id"]).contains("/base")) { collect(c, "F"); collect(c, "S"); } }
    write_report();
    // promote each model's Pareto front to level M (anchor items, flips); delete the files of dominated models
    if (phases.contains("sweeps")) {
        auto t2 = json.parse("[]");
        auto promoted = std::vector<auto>{};
        for (auto& m : models) {
            auto front = pareto(m["name"], "P");
            std::println("  {}: Pareto front {} techniques", std::string(m["name"]), len(front));
            for (auto c : cands) {
                std::string id = c["id"];
                if (std::string(c["model"]) != std::string(m["name"]) || id.contains("/base")) continue;
                bool onf = false; for (auto f : front) if (f == id) onf = true;
                if (onf) { add_tasks(t2, c, "M", true); promoted.push_back(c); }
            }
        }
        for (auto c : cands) { bool keep = false; for (auto p : promoted) if (std::string(p["id"]) == std::string(c["id"])) keep = true; if (!keep && !std::string(c["id"]).contains("/base") && fs.exists(model_out(c["id"]))) fs.remove(model_out(c["id"])); }
        run_tasks(t2);
        for (auto c : promoted) { collect(c, "M"); if (fs.exists(model_out(c["id"]))) fs.remove(model_out(c["id"])); }
        write_report();
    }
}

// ------------------------------------------------------------------------------------------------ table pruning
// technique family for pruning: 2:4 (with or without a fraction) apart from unstructured pruning
std::string prune_family(std::string r) { std::string f = family_of(r); if ((f == "prune" || f == "permute") && r.contains("nm=2:4")) return "prune24"; return f; }
// value of `key` (e.g. "n") in a recipe, as "(key=" or ",key="; -1 when absent
double param_of(std::string r, std::string key) {
    for (auto pre : std::vector<std::string>{"(" + key + "=", "," + key + "="}) {
        auto p = r.find(pre);
        if (p == std::string::npos) continue;
        std::string v = "";
        for (int i = p + len(pre); i < len(r); ++i) { auto ch = r.substr(i, 1); if (ch == "," || ch == ")" || ch == "+") break; v += ch; }
        if (v.contains(":")) return -1;
        return std::stod(v);
    }
    return -1;
}
// string value of `key` in a recipe ("" when absent)
std::string str_param(std::string r, std::string key) {
    for (auto pre : std::vector<std::string>{"(" + key + "=", "," + key + "="}) {
        auto p = r.find(pre);
        if (p == std::string::npos) continue;
        std::string v = "";
        for (int i = p + len(pre); i < len(r); ++i) { auto ch = r.substr(i, 1); if (ch == "," || ch == ")") break; v += ch; }
        return v;
    }
    return "";
}
// how much a recipe removes, comparable within one family (larger = more aggressive); -1 = unknown
double level_of(std::string r) {
    std::string f = prune_family(r);
    if (f == "depth") return param_of(r, "n");
    if (f == "width" || f == "heads") return param_of(r, "s");
    if (f == "actsparse") return param_of(r, "p");
    if (f == "experts" || f == "merge") return param_of(r, "drop");
    if (f == "moe2dense") { double w = param_of(r, "width"); return w < 0 ? -1 : 1 - w; }
    if (f == "optimise") { double kl = param_of(r, "kl"); return kl < 0 ? 0.005 : kl; }   // optimise's default KL target
    if (f == "prune24") { double fr = param_of(r, "frac"); return fr < 0 ? 1.0 : fr; }
    if (f == "prune") return param_of(r, "s");
    return -1;
}
// why `opt` need not be tried at this cap ("" = try it)
std::string prune_reason(std::string name, std::string opt, double cap) {
    std::string f = prune_family(opt);
    double lo = level_of(opt);
    for (auto s : sweep) {
        if (s[0] != name || !results.contains(ekey(name + "/" + s[1], "P"))) continue;
        auto r = results[ekey(name + "/" + s[1], "P")];
        if (!r.contains("dnll") || double(r["dnll"]) <= cap) continue;
        if (s[2] == opt) return std::format("the same recipe alone: dnll {:.4f} > cap {:.4f}", double(r["dnll"]), cap);
        double ls = level_of(s[2]);
        if (str_param(s[2], "score") != str_param(opt, "score") || str_param(s[2], "method") != str_param(opt, "method")) continue;   // a different scoring rule or method says nothing
        if (prune_family(s[2]) == f && lo >= 0 && ls >= 0 && ls <= lo)
            return std::format("{} at level {} <= {} already: dnll {:.4f} > cap {:.4f}", s[1], ls, lo, double(r["dnll"]), cap);
    }
    return "";
}

// ------------------------------------------------------------------------------------------------ phase: chains
// Beam search over the stages of compress_plan.cpp's chains. caps[k]: share of the dnll budget the chain may have
// used after stage k (cumulative).
void beam_chain(std::string name, std::vector<std::vector<std::string>> stages, std::vector<double> caps) {
    auto b = results[ekey(name + "/base", "F")];
    // the beam: {id, file, chain, est_mb, kl, dnll, bpw}
    auto beam = json.parse("[]");
    auto root = json.parse("{}"); root["id"] = name + "/base"; root["file"] = base_fpk(name); root["chain"] = ""; root["est_mb"] = b["est_mb"]; root["kl"] = 0.0; root["dnll"] = 0.0; root["bpw"] = 32.0;
    beam.push_back(root);
    for (int k = 0; k < len(stages); ++k) {
        double cap = DNLL_BUDGET * caps[k];
        // First fit, in waves: wave j tries option j (options run from the most to the least aggressive) for every
        // beam state that has no option within the cap yet; a state stops at its first fit. Table pruning skips an
        // option when a sweep of the same family that is at most as aggressive already broke the cap on this model
        // (damage is assumed to grow with the level; the sweep ran on the uncompressed model, a chain stage on an
        // already compressed one, so the assumption errs towards trying).
        auto cands = std::vector<auto>{};
        auto pool = json.parse("[]");
        for (auto st : beam) pool.push_back(st);   // "skip this stage"
        std::vector<bool> fitted;
        for (int bi = 0; bi < len(beam); ++bi) fitted.push_back(false);
        for (int j = 0; j < len(stages[k]); ++j) {
            std::string opt = stages[k][j];
            std::string why = prune_reason(name, opt, cap);
            if (why != "") { std::println("  chain {} s{}: {} pruned by the sweep table ({})", name, k + 1, opt, why); continue; }
            auto tasks = json.parse("[]");
            auto wave = std::vector<auto>{};
            for (int bi = 0; bi < len(beam); ++bi) {
                if (fitted[bi]) continue;
                auto st = beam[bi];
                auto c = json.parse("{}");
                c["id"] = name + "/chain/s" + std::to_string(k + 1) + "_b" + std::to_string(bi) + "_o" + std::to_string(j + 1);
                c["model"] = name; c["parent"] = st["file"]; c["recipe"] = opt; c["codec"] = "f32"; c["bpw_in"] = st["bpw"]; c["state"] = bi;
                c["chain"] = std::string(st["chain"]) == "" ? opt : std::string(st["chain"]) + " + " + opt;
                if (opt.contains("optimise")) c["calib"] = "self";
                add_tasks(tasks, c, "S", true);
                wave.push_back(c); cands.push_back(c);
            }
            if (len(wave) == 0) break;   // every state has its fit
            std::println("chain {} stage {} wave {}: {} for {} state(s) (cap dnll {:.4f})", name, k + 1, j + 1, opt, len(wave), cap);
            run_tasks(tasks);
            for (auto c : wave) {
                auto r = collect(c, "S");
                if (r.contains("failed") || !r.contains("dnll")) continue;
                auto s = json.parse("{}"); s["id"] = c["id"]; s["file"] = model_out(c["id"]); s["chain"] = c["chain"]; s["est_mb"] = r["est_mb"]; s["kl"] = r["eval"]["kl"]["kl"]; s["dnll"] = r["dnll"]; s["bpw"] = r["bpw"];
                if (double(r["dnll"]) <= cap) { pool.push_back(s); fitted[int(c["state"])] = true; std::println("  fits: {} (dnll {:+.4f})", std::string(c["chain"]), double(r["dnll"])); }
                else std::println("  {} over the cap: dnll {:.4f}", std::string(c["chain"]), double(r["dnll"]));
            }
        }
        // rank: estimated size, then KL; keep the best BEAM distinct chains
        std::vector<int> idx;
        for (int i = 0; i < len(pool); ++i) idx.push_back(i);
        for (int i = 0; i < len(idx); ++i) for (int j = i + 1; j < len(idx); ++j) {
            auto a = pool[idx[i]], c2 = pool[idx[j]];
            bool better = double(c2["est_mb"]) < double(a["est_mb"]) - 0.5 || (std::abs(double(c2["est_mb"]) - double(a["est_mb"])) <= 0.5 && double(c2["kl"]) < double(a["kl"]));
            if (better) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        }
        auto nb = json.parse("[]");
        for (int i = 0; i < len(idx) && len(nb) < BEAM; ++i) nb.push_back(pool[idx[i]]);
        // delete candidate files that did not make the beam (beam states of earlier stages stay while referenced)
        for (auto c : cands) { bool keep = false; for (auto s : nb) if (std::string(s["id"]) == std::string(c["id"])) keep = true; if (!keep && fs.exists(model_out(c["id"]))) fs.remove(model_out(c["id"])); }
        for (auto s : nb) { std::println("  beam: {} (est {:.0f} MB, KL {:.4f}, dnll {:+.4f})", std::string(s["chain"]) == "" ? "(none)" : std::string(s["chain"]), double(s["est_mb"]), double(s["kl"]), double(s["dnll"]));
                            if (results.contains(ekey(s["id"], "S"))) { auto r = results[ekey(s["id"], "S")]; r["kept"] = "stage " + std::to_string(k + 1); record(r); } }
        beam = nb;
        write_report();
    }
    // leaves: level M (anchors, flips); the smallest passing the paired checks gets level F
    auto tasks = json.parse("[]");
    auto leaves = std::vector<auto>{};
    for (auto s : beam) {
        if (std::string(s["chain"]) == "") continue;
        auto c = json.parse("{}"); c["id"] = s["id"]; c["model"] = name; c["parent"] = s["file"]; c["recipe"] = ""; c["codec"] = "f32"; c["chain"] = s["chain"]; c["bpw"] = s["bpw"];
        add_tasks(tasks, c, "M", false);
        leaves.push_back(c);
    }
    run_tasks(tasks);
    std::string best = "", best_chain = "", best_file = "";
    double best_mb = 1e18, bacc = b["eval"]["mean_acc"];
    // level M perplexity uses the PPL_P test windows: compare with the baseline on the same windows
    double bppl = results.contains(ekey(name + "/base", "P")) ? double(results[ekey(name + "/base", "P")]["eval"]["ppl"]) : double(b["eval"]["ppl"]);
    for (auto c : leaves) {
        auto r = collect(c, "M");
        if (r.contains("failed")) continue;
        auto pr = paired(r, b);
        double acc_ratio = (bacc + (pr.contains("dacc") ? double(pr["dacc"]) : 0)) / bacc;
        bool okk = double(r["eval"]["ppl"]) <= bppl * (1 + BUDGET) && acc_ratio >= 1 - BUDGET;
        std::println("  leaf {}: dppl {} acc ratio {:.4f} -> {}", std::string(c["chain"]), pct(double(r["eval"]["ppl"]) / bppl - 1), acc_ratio, okk ? "ok" : "fails");
        if (okk && double(r["est_mb"]) < best_mb) { best_mb = r["est_mb"]; best = c["id"]; best_chain = c["chain"]; best_file = c["parent"]; }
    }
    if (best == "") { std::println("chain {}: no leaf passes the anchor checks", name); return; }
    auto c = json.parse("{}"); c["id"] = name + "/chain/final"; c["model"] = name; c["parent"] = best_file; c["recipe"] = ""; c["codec"] = "f32"; c["chain"] = best_chain;
    c["bpw"] = results[ekey(best, "M")]["bpw"];
    auto t3 = json.parse("[]");
    add_tasks(t3, c, "F", false);
    run_tasks(t3);
    auto f = collect(c, "F");
    std::println("chain {}: final {} -> ppl {:.3f}, mean acc {:.4f}", name, best_chain, double(f["eval"]["ppl"]), double(f["eval"]["mean_acc"]));
    write_report();
}

if (phases.contains("chains") && wanted("qwen3-0.6b")) {
    beam_chain("qwen3-0.6b", {
        {"depth(n=4,score=probe,shortlist=6,probe=4)", "depth(n=2,score=probe,shortlist=6,probe=4)", "depth(n=1,score=probe,shortlist=6,probe=4)"},
        {"width(s=0.2,score=obs,alloc=eba)", "width(s=0.1,score=obs,alloc=eba)", "width(s=0.05,score=obs,alloc=eba)"},
        {"heads(s=0.25,score=obs,alloc=eba)", "heads(s=0.125,score=obs)"},
        {"permute+prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.3,refit=mcr,iters=15)", "prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.15,refit=mcr,iters=15)", "prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.08,refit=mcr,iters=15)"},
        {"optimise(kl=0.02)", "optimise(kl=0.008)", "optimise(kl=0.003)"}},
        {0.2, 0.4, 0.55, 0.75, 0.92});
}
if (phases.contains("chains") && wanted("tinyllama-sparse24")) {
    beam_chain("tinyllama-sparse24", {
        {"depth(n=4,score=probe,shortlist=6,probe=4)", "depth(n=2,score=probe,shortlist=6,probe=4)", "depth(n=1,score=probe,shortlist=6,probe=4)"},
        {"width(s=0.15,score=obs)", "width(s=0.08,score=obs)", "width(s=0.04,score=obs)"},
        {"heads(s=0.25,score=obs)", "heads(s=0.125,score=obs)"},
        {"optimise(kl=0.02)", "optimise(kl=0.008)", "optimise(kl=0.003)"}},
        {0.3, 0.55, 0.7, 0.92});
}
if (phases.contains("chains") && wanted("tinymixtral")) {
    beam_chain("tinymixtral", {
        {"moe2dense(width=0.5)", "merge(drop=1,method=ls)", "experts(drop=1,score=reap)", "moe2dense(width=0.75)"},
        {"depth(n=2,score=probe,shortlist=6,probe=4)", "depth(n=1,score=probe,shortlist=6,probe=4)"},
        {"optimise(kl=0.02)", "optimise(kl=0.008)", "optimise(kl=0.003)"}},
        {0.5, 0.7, 0.92});
}

write_report();
fs.write_atomic(RUNS + "/plan" + LOCK_TAG + ".done", time.iso(time.now()) + "\n");
std::println("fast plan: done; report in {}", RES + "/REPORT_FAST" + LOCK_TAG + ".md");

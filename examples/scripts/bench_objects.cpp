// Strings, containers and objects: word frequencies and a small particle update. `forge script bench_objects.cpp [rounds]`.
// Used for the numbers in docs/aot.md (this work is done by the runtime's built-ins in every mode).
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct Particle { double x = 0; double v = 0; int hits = 0; };

int main() {
    int rounds = 40;
    if (args.size() > 0) rounds = std::stoi(args[0]);
    std::vector<std::string> words{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"};
    std::map<std::string, int> freq;
    std::vector<Particle> ps;
    for (int i = 0; i < 64; ++i) { Particle p; p.x = i * 0.5; p.v = (i % 7) - 3.0; ps.push_back(p); }
    std::string text;
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < 50; ++i) {
            const std::string& w = words[(i * 7 + r) % words.size()];
            freq[w] += 1;
            if (text.size() < 4000) text += w.substr(0, 2);
        }
        for (auto& p : ps) {
            p.x += p.v * 0.01;
            if (p.x < 0 || p.x > 32) { p.v = -p.v; p.hits += 1; }
        }
    }
    int hits = 0; double sum = 0;
    for (const auto& p : ps) { hits += p.hits; sum += p.x; }
    for (const auto& [w, n] : freq) std::print("{}={} ", w, n);
    std::println("");
    std::println("hits={} sum={:.3f} text={}", hits, sum, text.size());
    return 0;
}

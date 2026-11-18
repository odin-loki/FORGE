// Prints forge::import::tokimp::gguf_pre_regexes() for every name read from argv[1] (one per line) as JSON.
// Used by diff_llamacpp_pretok.py. Build: g++ -std=c++23 $(for d in */include; do printf -- "-I%s " $d; done) verify/golden/dump_pretok.cpp -o dump_pretok
#include <forge/import/tokenizer_import.hpp>
#include <cstdio>
#include <fstream>
#include <iostream>

static void jstr(const std::string& s) {
    std::putchar('"');
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { std::putchar('\\'); std::putchar(c); }
        else if (c < 0x20) std::printf("\\u%04x", c);
        else std::putchar(c);
    }
    std::putchar('"');
}

int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    std::string name; bool first = true;
    std::printf("{");
    while (std::getline(in, name)) {
        if (name.empty()) continue;
        std::vector<std::string> re; bool ig = false;
        bool found = forge::import::tokimp::gguf_pre_regexes(name, re, ig);
        std::printf("%s", first ? "" : ",\n"); first = false;
        jstr(name); std::printf(":{\"found\":%s,\"ignore_merges\":%s,\"re\":[", found ? "true" : "false", ig ? "true" : "false");
        for (size_t i = 0; i < re.size(); i++) { if (i) std::putchar(','); jstr(re[i]); }
        std::printf("]}");
    }
    std::printf("}\n");
}

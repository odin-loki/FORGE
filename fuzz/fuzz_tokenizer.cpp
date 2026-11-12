#include "fuzz.hpp"
#include <forge/import/tokenizer_import.hpp>
#include <forge/modelfile/tokenizer.hpp>
// input = tokenizer.json [\x01 text-to-encode]            every model type, normalizer, pre-tokenizer, decoder
//      or \x02 SentencePiece ModelProto [\x01 text]       the protobuf reader and its conversion to tokenizer.json
// The patterns in a tokenizer.json go to the regex engine and the text through every stage, so both are fuzzed.
FORGE_FUZZ({
    std::string_view s{reinterpret_cast<const char*>(data), size};
    auto cut = s.find('\x01');
    auto txt = cut == std::string_view::npos ? std::string_view("the theory of things, 12345 \xC3\xA9\xE4\xBD\xA0 <s>") : s.substr(cut + 1);
    if (txt.size() > 4096) txt = txt.substr(0, 4096);
    auto doc = s.substr(0, cut);
    std::string converted;
    if (!doc.empty() && doc[0] == '\x02') { converted = forge::import::sentencepiece_tokenizer_json(doc.substr(1)); doc = converted; }
    forge::modelfile::tokenizer t(doc);
    t.set_special(int(size % 7) - 1, int(size % 5) - 1);
    auto ids = t.encode(txt);
    for (int id : ids) if (id < 0 || std::size_t(id) >= t.vocab_size()) std::abort();   // ids always name a vocabulary entry
    (void)t.decode(ids);
    (void)t.decode(t.encode(txt, true), true);
    for (int id : ids) (void)t.decode_one(id);
    (void)t.pre_tokenize(txt.substr(0, 256));
    (void)t.decode({-1, 0, 1, int(t.vocab_size()), 1 << 30});
})

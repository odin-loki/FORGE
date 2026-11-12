// std::string and std::string_view member functions, conversions, <cctype>.
#include <algorithm>
#include <cctype>
#include <print>
#include <string>
#include <string_view>
#include <vector>

std::size_t count_words(std::string_view text) {
    std::size_t n = 0;
    bool in = false;
    for (char c : text) {
        if (c == ' ') in = false;
        else if (!in) { in = true; ++n; }
    }
    return n;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        std::size_t p = s.find(sep, start);
        if (p == std::string::npos) { out.push_back(s.substr(start)); break; }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
    return out;
}

int main() {
    std::string s = "hello world";
    std::println("{} {} {}", s.size(), s.length(), s.empty());
    std::println("{} {} {} {}", s[0], s.at(4), s.front(), s.back());
    std::println("{}|{}|{}", s.substr(6), s.substr(0, 5), s.substr(3, 100));
    std::println("{} {} {}", s.find("o"), s.find("o", 5), s.find('w'));
    std::println("{} {}", s.rfind("o"), s.rfind('l'));
    std::println("{} {}", s.find("zz") == std::string::npos, s.find("") );
    std::println("{} {}", s.find_first_of("aeiou"), s.find_last_of("aeiou"));
    std::println("{} {}", s.find_first_not_of("hel"), s.find_last_not_of("dl"));
    std::println("{} {} {}", s.starts_with("hello"), s.ends_with("world"), s.starts_with('x'));
    std::println("{} {}", s.contains("lo w"), s.contains('z'));
    std::println("{} {} {}", s.compare("hello world"), s.compare("hello") > 0, s.compare("z") < 0);

    // construction
    std::string a(3, 'x'), b("abcdef"), c(s, 6), d(s, 0, 4), e;
    std::string g(s.begin(), s.begin() + 5);
    std::println("{}|{}|{}|{}|{}|{}", a, b, c, d, e, g);

    // modification
    std::string t = "abc";
    t.push_back('d'); t += "ef"; t += 'g'; t.append("hi"); t.append(2, '!');
    std::println("{}", t);
    t.pop_back();
    t.insert(0, ">> "); t.insert(t.size(), " <<"); t.insert(3, 2, '*');
    std::println("{}", t);
    t.erase(0, 3); std::println("{}", t);
    t.erase(t.size() - 3); std::println("{}", t);
    t.replace(0, 2, "##"); std::println("{}", t);
    t.replace(t.find("def"), 3, "DEFDEF"); std::println("{}", t);
    t.erase(t.begin()); std::println("{}", t);
    t.resize(4); std::println("{}|{}", t, t.size());
    t.resize(6, '.'); std::println("{}|{}", t, t.size());
    t.clear(); std::println("[{}] {}", t, t.empty());
    t.assign("fresh"); std::println("{}", t);
    t.assign(2, 'z'); std::println("{}", t);
    std::string u = "left";
    std::string w = "right";
    u.swap(w);
    std::println("{} {}", u, w);
    t = "abc";
    t[1] = 'X'; t.at(2) = 'Y'; t.front() = 'W'; t.back() = 'Z';
    std::println("{}", t);

    // concatenation and comparison
    std::string first = "ab", second = "cd";
    std::println("{} {} {} {}", first + second, first + "x", "y" + second, first + '!');
    std::println("{} {} {} {}", first == "ab", first != second, first < second, std::string("B") < std::string("a"));
    std::println("{} {}", std::string("abc") < std::string("abd"), std::string("ab") < std::string("abc"));
    std::println("{} {}", first <= first, second > first);

    // iteration
    std::string rev = "stressed";
    std::reverse(rev.begin(), rev.end());
    std::println("{}", rev);
    std::string up;
    for (char ch : rev) up += static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    std::println("{}", up);
    for (char& ch : up) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    std::println("{}", up);
    std::println("{}", std::count(s.begin(), s.end(), 'l'));
    int vowels = 0;
    for (std::size_t i = 0; i < s.size(); ++i) if (std::string("aeiou").find(s[i]) != std::string::npos) ++vowels;
    std::println("{}", vowels);

    // <cctype>
    std::println("{} {} {} {}", std::isdigit('7') != 0, std::isalpha('x') != 0, std::isspace(' ') != 0, std::isupper('a') != 0);
    std::println("{} {} {} {}", std::isalnum('_') != 0, std::islower('q') != 0, std::ispunct('!') != 0, std::isdigit('f') != 0);
    std::println("{} {}", static_cast<char>(std::toupper('q')), static_cast<char>(std::tolower('Q')));

    // conversions
    std::println("{} {} {}", std::to_string(42), std::to_string(-7), std::to_string(3.5));
    std::println("{} {}", std::stoi("123"), std::stoi("  -45xyz"));
    std::println("{} {} {}", std::stol("99"), std::stoll("123456789012"), std::stoul("17"));
    std::println("{} {}", std::stod("2.5e2"), std::stof("0.5"));
    int total = 0;
    for (const auto& part : split("10,20,,30", ',')) total += part.empty() ? 1 : std::stoi(part);
    std::println("{} {}", total, split("a b c", ' ').size());

    // string_view
    std::string_view sv = "the quick brown fox";
    std::println("{} {} {}", sv.size(), sv.substr(4, 5), sv.find("brown"));
    std::println("{} {} {}", sv.starts_with("the"), sv.ends_with("fox"), sv.front());
    std::println("{} {}", count_words(sv), count_words("  two   words "));
    sv = sv.substr(4, sv.size() - 8);
    std::println("{}|{}", sv, sv.empty());
    std::string owned(sv);
    owned += "!";
    std::println("{} {}", owned, sv == "quick brown");
    std::string_view whole = s;
    std::println("{} {}", whole.substr(0, 5) == "hello", whole[6]);

    // multi-line and escapes
    std::string esc = "tab\there \"quoted\" back\\slash";
    std::println("{} {}", esc, esc.size());
    std::string raw = R"(raw \n "text")";
    std::println("{} {}", raw, raw.size());
    std::string joined = "con" "cat" "enated";
    std::println("{}", joined);
    return 0;
}

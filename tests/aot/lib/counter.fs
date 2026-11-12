// Imported as a module object: import "lib/counter" as counter;
int count = 0;
int next() { count += 1; return count; }
std::string label(const std::string& prefix, int n) { return prefix + "#" + std::to_string(n); }

// Static imports are bundled into the AOT binary; their functions are compiled natively like the main script's.
#include "lib/shapes.fs"
import "lib/counter" as counter;

int main() {
    std::println("{} {}", rect_area(3, 4), circle_area_scaled(2));
    std::println("{} {}", counter.next(), counter.next());
    std::println("{}", counter.label("n", counter.next()));
    try { auto missing = forge::import("lib/does_not_exist"); } catch (const std::exception& e) { std::println("import failed"); }
    return counter.next() == 4 ? 0 : 1;
}

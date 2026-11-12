// Constructors: delegating constructors, base classes with constructors of several arities, member initialisers
// that construct (vector(n, value), class members), and members that keep their declared arithmetic type.
#include <print>
#include <string>
#include <vector>

struct Size {
    double w = 0, h = 0;
    int count = 0;
};

class Range {
public:
    Range() : Range(0, 10) { std::println("Range() body"); }
    explicit Range(int hi) : Range(0, hi) { std::println("Range(int) body"); }
    Range(int lo, int hi) : lo_(lo), hi_(hi) { std::println("Range({}, {})", lo_, hi_); }
    int span() const { return hi_ - lo_; }
private:
    int lo_, hi_;
};

class Shape {
public:
    Shape() : name_("unnamed"), sides_(0) {}
    explicit Shape(std::string name) : name_(name), sides_(0) {}
    Shape(std::string name, int sides) : name_(name), sides_(sides) {}
    virtual ~Shape() = default;
    std::string describe() const { return name_ + "/" + std::to_string(sides_); }
private:
    std::string name_;
    int sides_;
};
class Blob : public Shape {
public:
    Blob() {}
};
class Circle : public Shape {
public:
    Circle() : Shape("circle") {}
};
class Square : public Shape {
public:
    explicit Square(double side) : Shape("square", 4), side_(side) {}
    Square() : Square(1) {}
    double area() const { return side_ * side_; }
private:
    double side_;
};

class Grid {
public:
    Grid(int rows, int cols) : rows_(rows), cols_(cols), cells_(rows * cols, 7), names_(rows, "r"), origin_() {}
    int sum() const { int t = 0; for (int c : cells_) t += c; return t; }
    std::size_t names() const { return names_.size(); }
    std::string first() const { return names_.empty() ? "" : names_[0]; }
    void set(int r, int c, int v) { cells_[r * cols_ + c] = v; }
    double ow() const { return origin_.w; }
private:
    int rows_, cols_;
    std::vector<int> cells_;
    std::vector<std::string> names_;
    Size origin_;
};

struct Money {
    double amount;
    int cents;
    Money(int whole) : amount(whole), cents(0) {}
    Money(double a, double c) : amount(a), cents(c) {}
};

int main() {
    Range a;
    Range b(5);
    Range c(2, 4);
    std::println("{} {} {}", a.span(), b.span(), c.span());

    Blob blob;
    Circle circle;
    Square sq(3);
    Square unit;
    std::println("{} {} {} {}", blob.describe(), circle.describe(), sq.describe(), unit.describe());
    std::println("{} {}", sq.area(), unit.area() / 2);

    Grid g(2, 3);
    g.set(1, 2, 1);
    std::println("{} {} {} {}", g.sum(), g.names(), g.first(), g.ow());

    Size s;
    s.w = 3;                  // a double member holds 3.0
    s.h = 7;
    s.count = 2.9;            // an int member truncates
    std::println("{} {} {}", s.w / 2, s.h / 2, s.count);
    s.w += 1;
    s.w = s.w * 2;
    std::println("{}", s.w / 3 > 2.6);
    Size t{1, 2, 3};
    std::println("{} {} {}", t.w / 2, t.h / 4, t.count / 2);

    Money m(5);
    std::println("{} {}", m.amount / 2, m.cents);
    Money n(7, 99.9);
    std::println("{} {}", n.amount / 2, n.cents);
    return 0;
}

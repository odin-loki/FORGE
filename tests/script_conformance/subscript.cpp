// User-defined operator[]: reading, assigning through a reference-returning operator, compound assignment,
// const and non-const pairs, keys of other types, and a read-only (by value) operator.
#include <map>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
    std::vector<double>& row(int r) { return scratch_; }
    double& operator[](int i) { return data_[i]; }
    const double& operator[](int i) const { return data_[i]; }
    double at(int r, int c) const { return data_[r * cols_ + c]; }
    int size() const { return rows_ * cols_; }
private:
    int rows_, cols_;
    std::vector<double> data_;
    std::vector<double> scratch_;
};

struct Cell { int value = 0; std::string tag; };

class Board {
public:
    explicit Board(int n) : cells_(n) {}
    Cell& operator[](int i) { return cells_[i]; }
private:
    std::vector<Cell> cells_;
};

class Registry {
public:
    int& operator[](const std::string& key) { return counts_[key]; }
    std::size_t size() const { return counts_.size(); }
private:
    std::map<std::string, int> counts_;
};

class Squares {   // read-only: returns by value
public:
    int operator[](int i) const { return i * i; }
};

class Checked {
public:
    Checked() : items_{1, 2, 3} {}
    int operator[](int i) const {
        if (i < 0 || i >= 3) throw std::out_of_range("Checked: index out of range");
        return items_[i];
    }
private:
    std::vector<int> items_;
};

double total(const Matrix& m) { double t = 0; for (int i = 0; i < m.size(); ++i) t += m[i]; return t; }

int main() {
    Matrix m(2, 3);
    m[0] = 1.5;
    m[4] = 2;
    m[4] += 0.25;
    m[5] = m[0] * 2;
    std::println("{} {} {} {}", m[0], m[4], m[5], total(m));
    std::println("{} {}", m.at(1, 1), m[4] / 2);

    Board b(3);
    b[1].value = 7;
    b[1].tag = "x";
    b[2] = Cell{9, "nine"};
    b[0].value += 1;
    std::println("{} {} {} {} {}", b[0].value, b[1].value, b[1].tag, b[2].value, b[2].tag);

    Registry reg;
    reg["a"] = 5;
    reg["b"] += 2;
    reg["a"] += 1;
    ++reg["c"];
    std::println("{} {} {} {}", reg["a"], reg["b"], reg["c"], reg.size());

    Squares sq;
    std::println("{} {}", sq[7], sq[sq[2]]);

    Checked c;
    try { std::println("{}", c[1]); std::println("{}", c[5]); }
    catch (const std::out_of_range& e) { std::println("caught: {}", e.what()); }
    return 0;
}

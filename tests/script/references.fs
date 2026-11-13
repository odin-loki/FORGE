// A reference is another name for the object: binding one does not copy it and does not run its destructor a
// second time (it used to: `Base& b = d;` printed the destructor twice and writes through b were lost).
struct Base {
    std::string name;
    Base(std::string n) : name(n) { std::println("ctor {}", name); }
    virtual ~Base() { std::println("dtor {}", name); }
    virtual int kind() const { return 1; }
};
struct Derived : Base {
    int extra = 7;
    Derived() : Base("d") {}
    ~Derived() { std::println("~Derived {}", name); }
    int kind() const override { return 2; }
};
struct Point { int x = 0; int y = 0; };
struct Box { Point p; std::vector<int> v; std::string s; double w = 0; int n = 0; };
struct Range {
    int lo; int hi;
    Range() : Range(0, 10) { std::println("Range()"); }
    Range(int h) : Range(0, h) {}
    Range(int l, int h) : lo(l), hi(h) { std::println("Range({}, {})", lo, hi); }
};
struct Shape {
    std::string name; int sides;
    Shape() : name("unnamed"), sides(0) {}
    Shape(std::string n) : name(n), sides(0) {}
    Shape(std::string n, int s) : name(n), sides(s) {}
};
struct Blob : Shape { Blob() {} };
struct Square : Shape { double side; Square(double s) : Shape("square", 4), side(s) {} Square() : Square(1) {} };
struct Grid { std::vector<int> cells; Point origin; Grid(int n) : cells(n, 7), origin() {} };

void scope_test() {
    Derived d;
    Base& b = d;
    std::println("{} {}", b.kind(), b.name);
    {
        Base& inner = b;
        inner.name = "x";
        const Derived& c = d;
        std::println("{} {}", c.name, c.extra);
    }
    std::println("after block {}", d.name);
    auto& a = d;
    a.extra = 9;
    std::println("{}", d.extra);
    int n = 1;
    int& r = n;
    r += 4; ++r;
    std::println("{} {}", n, r);
    double x = 1.5;
    double& xr = x;
    xr = 2;
    std::println("{}", x / 4);
    Box box;
    Point& bp = box.p;
    bp.x = 11;
    std::vector<int>& bv = box.v;
    bv.push_back(1);
    const std::string& bs = box.s;
    box.s.append("m");
    std::println("{} {} {}", box.p.x, box.v.size(), bs);
    std::vector<Point> pts(2);
    Point& last = pts.back();
    last.y = 22;
    std::println("{}", pts[1].y);
    std::println("leaving");
}
scope_test();
std::println("--");
void lambda_test() {
    int count = 0;
    int& cref = count;
    auto bump = [&cref]() { cref += 10; };
    bump(); bump();
    std::println("{}", count);
}
lambda_test();

// at file scope
Point gp;
Point& gq = gp;
gq.x = 5;
gq = Point{6, 7};
int gn = 1;
int& gr = gn;
gr = 8;
std::println("{} {} {}", gp.x, gp.y, gn);

// members keep their declared arithmetic type
Box bx;
bx.w = 3; bx.n = 2.9;
std::println("{} {}", bx.w / 2, bx.n);

// delegating constructors, bases with several constructors, typed member initialisers
Range r0; Range r1(5);
std::println("{} {}", r0.hi - r0.lo, r1.hi);
Blob blob; Square sq(3); Square unit;
std::println("{} {}/{} {} {}", blob.name, sq.name, sq.sides, sq.side * sq.side, unit.side / 2);
Grid g(3);
std::println("{} {} {}", g.cells.size(), g.cells[2], g.origin.x);

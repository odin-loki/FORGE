// References: a reference is another name for the object. Binding one must not copy the object, must not run
// its destructor a second time, and writes through it must reach the original (objects, scalars, members,
// elements). Base& to a derived object keeps the dynamic type.
#include <iostream>
#include <map>
#include <print>
#include <string>
#include <vector>

struct Base {
    std::string name;
    explicit Base(std::string n) : name(n) { std::println("ctor {}", name); }
    virtual ~Base() { std::println("dtor {}", name); }
    virtual int kind() const { return 1; }
};
struct Derived : Base {
    int extra = 7;
    Derived() : Base("d") {}
    ~Derived() override { std::println("~Derived {}", name); }
    int kind() const override { return 2; }
};
struct Point { int x = 0; int y = 0; };
struct Box { Point p; std::vector<int> v; std::string s; };

int total(const std::vector<int>& v) { int t = 0; for (int x : v) t += x; return t; }
void rename(Base& b, const std::string& n) { b.name = n; }

int main() {
    {
        Derived d;
        Base& b = d;                       // no copy, no destructor at the end of b's scope
        std::println("kind {} {}", b.kind(), b.name);
        {
            Base& inner = b;               // a reference to a reference
            inner.name = "x";
            const Derived& c = d;
            std::println("const ref sees {} extra {}", c.name, c.extra);
        }
        std::println("after inner block: {}", d.name);
        rename(b, "y");
        std::println("after call: {} {}", d.name, b.name);
        auto& a = d;
        a.extra = 9;
        const auto& ca = d;
        std::println("extra {} {}", d.extra, ca.extra);
        std::println("leaving scope");
    }
    std::println("-- scalars");
    int n = 1;
    int& r = n;
    r = 5;
    r += 2;
    ++r;
    std::println("{} {}", n, r);
    n = 40;
    std::println("{}", r);
    const int& cr = n;
    n = 41;
    std::println("{}", cr);
    double x = 1.5;
    double& xr = x;
    xr = 2;                                // stays a double
    std::println("{}", x / 4);
    std::string s = "ab";
    std::string& sr = s;
    sr += "cd";
    sr = "z" + sr;                         // assignment through the reference
    std::println("{} {}", s, sr.size());

    std::println("-- plain structs");
    Point p;
    Point& q = p;
    q.x = 3;
    auto& q2 = q;
    q2.y = 4;
    std::println("{} {}", p.x, p.y);
    Point copy = q;                        // a copy from a reference is a copy
    copy.x = 100;
    std::println("{} {}", p.x, copy.x);
    q = Point{7, 8};                       // assigns to p
    std::println("{} {}", p.x, p.y);

    std::println("-- members and elements");
    Box box;
    Point& bp = box.p;
    bp.x = 11;
    std::vector<int>& bv = box.v;
    bv.push_back(1); bv.push_back(2);
    std::string& bs = box.s;
    bs.append("member");                 // (assignment through a member reference is a diagnostic: docs/scripting.md)
    std::println("{} {} {} {}", box.p.x, box.v.size(), total(box.v), box.s);
    std::vector<Point> pts(3);
    Point& mid = pts[1];
    mid.x = 21;
    Point& last = pts.back();
    last.y = 22;
    const Point& first = pts.front();
    pts[0].x = 20;
    std::println("{} {} {} {}", pts[0].x, pts[1].x, pts[2].y, first.x);
    std::map<std::string, Point> named;
    Point& origin = named["origin"];
    origin.x = -1;
    std::println("{} {}", named["origin"].x, named.size());
    std::vector<std::vector<int>> grid(2);
    std::vector<int>& row = grid[1];
    row.push_back(5);
    std::println("{} {}", grid[0].size(), grid[1].size());

    std::println("-- lambdas");
    int count = 0;
    int& cref = count;
    auto bump = [&cref]() { cref += 10; };
    bump(); bump();
    std::println("{}", count);
    Point lp;
    Point& lpr = lp;
    auto setx = [&](int v) { lpr.x = v; };
    setx(33);
    std::println("{}", lp.x);
    return 0;
}

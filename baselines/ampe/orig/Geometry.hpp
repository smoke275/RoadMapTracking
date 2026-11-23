#include<bits/stdc++.h>
using namespace std;

const double epsilon = 1e-9;

struct pt {
    double x, y;
    pt() {}
    pt(double _x, double _y) : x(_x), y(_y) {}
    pt operator-(const pt& p) const { return pt(x - p.x, y - p.y); }
    double cross(const pt& p) const { return x * p.y - y * p.x; }
    double cross(const pt& a, const pt& b) const { return (a - *this).cross(b - *this); }
};

inline int sgn(const double& x) { return (abs(x) > epsilon) ? x > 0 ? 1 : -1 : 0; }

inline bool inter1(double a, double b, double c, double d) {
    if (a > b)  swap(a, b);
    if (c > d)  swap(c, d);

    return max(a, c) <= min(b, d);
}

inline bool check_inter(const pt a, const pt b, const pt c, const pt d) {
    if (abs(c.cross(a, d)) <= epsilon && abs(c.cross(b, d)) <= epsilon)
        return inter1(a.x, b.x, c.x, d.x) && inter1(a.y, b.y, c.y, d.y);

    return sgn(a.cross(b, c)) != sgn(a.cross(b, d)) &&
           sgn(c.cross(d, a)) != sgn(c.cross(d, b));
}

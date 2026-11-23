// Replacement for the authors' Environment.hpp with the same interface.
// The free space is the interior of ONE simple polygon (closed set), and
// intersection(a, b) is true when the segment ab is NOT contained in it.
#include <bits/stdc++.h>
using namespace std;
#define pdd pair<double, double>
const double eps = 1e-9;

class Environment {
public:
    vector<pdd> V;
    double tol = 1e-6;
    Environment() {}
    Environment(vector<pdd> poly) : V(poly) {}

    double distBoundary(double x, double y) const {
        double best = 1e300; int n = V.size();
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double ax = V[j].first, ay = V[j].second, bx = V[i].first, by = V[i].second;
            double dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
            double t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0.0;
            t = max(0.0, min(1.0, t));
            best = min(best, hypot(x - (ax + t * dx), y - (ay + t * dy)));
        }
        return best;
    }
    bool strictlyInside(double x, double y) const {
        bool in = false; int n = V.size();
        for (int i = 0, j = n - 1; i < n; j = i++)
            if ((V[i].second > y) != (V[j].second > y) &&
                x < (V[j].first - V[i].first) * (y - V[i].second) / (V[j].second - V[i].second) + V[i].first)
                in = !in;
        return in;
    }
    bool inside(double x, double y) const { return strictlyInside(x, y) || distBoundary(x, y) <= tol; }
    bool withinObstacle(double x, double y) const { return !inside(x, y); }

    bool intersection(double x1, double y1, double x2, double y2) const {
        double dx = x2 - x1, dy = y2 - y1, L = hypot(dx, dy);
        if (L < tol) return !inside(x1, y1);
        vector<double> ts = {0.0, 1.0};
        int n = V.size();
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double ax = V[j].first, ay = V[j].second, ex = V[i].first - ax, ey = V[i].second - ay;
            double den = dx * ey - dy * ex;
            if (fabs(den) > 1e-12) {
                double t = ((ax - x1) * ey - (ay - y1) * ex) / den;
                double u = ((ax - x1) * dy - (ay - y1) * dx) / den;
                double tu = tol / max(hypot(ex, ey), 1e-12), tt = tol / L;
                if (t > -tt && t < 1 + tt && u > -tu && u < 1 + tu) ts.push_back(max(0.0, min(1.0, t)));
            } else {
                // parallel: add projections of the edge endpoints if collinear
                double c = (ax - x1) * dy - (ay - y1) * dx;
                if (fabs(c) / L <= tol) {
                    for (double px : {0.0, 1.0}) {
                        double qx = ax + px * ex, qy = ay + px * ey;
                        double t = ((qx - x1) * dx + (qy - y1) * dy) / (L * L);
                        if (t > 0 && t < 1) ts.push_back(t);
                    }
                }
            }
        }
        sort(ts.begin(), ts.end());
        for (size_t k = 0; k + 1 < ts.size(); k++) {
            if ((ts[k + 1] - ts[k]) * L <= 2 * tol) continue;
            double m = 0.5 * (ts[k] + ts[k + 1]);
            if (!inside(x1 + m * dx, y1 + m * dy)) return true;
        }
        return false;
    }
    bool intersection(vector<double>& a, vector<double>& b) const { return intersection(a[0], a[1], b[0], b[1]); }
};

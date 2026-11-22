// Native per-frame core of the tracker (C ABI, loaded by ker_native.py).
//
//   ker_path_lengths  : geodesic distance from the evader to every corner
//   ker_optimal_guard : exact min-max position over all roadmap edges
//   ker_edge_offset   : minimizer along one given edge
//   ker_shortest_path : geodesic shortest path between two points
//
// These mirror ker_pipeline.compute_path_lengths, compute_optimal_guard and
// _opt_offset with visibility gating off (weights 1). The geodesic distances
// use a vertex-to-vertex distance matrix computed once per polygon and one
// visibility test per polygon vertex per query.
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace {
struct P { double x, y; };
const double INF = 1e300;

double cr(P a, P b, P c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }

bool properX(P a, P b, P c, P d) {
  const double e = 1e-9;
  double d1 = cr(a, b, c), d2 = cr(a, b, d), d3 = cr(c, d, a), d4 = cr(c, d, b);
  return ((d1 > e && d2 < -e) || (d1 < -e && d2 > e)) &&
         ((d3 > e && d4 < -e) || (d3 < -e && d4 > e));
}

struct Ker {
  int n, nc, nv, ne;
  std::vector<P> poly;
  std::vector<int> C, e1, e2;
  std::vector<double> D, el, G;        // D: nv x nc roadmap distances, G: n x n geodesic
  std::vector<int> nxt;                // next vertex on the geodesic from i to j

  bool inside(P q) const {
    bool in = false;
    for (int i = 0, j = n - 1; i < n; j = i++)
      if ((poly[i].y > q.y) != (poly[j].y > q.y) &&
          q.x < (poly[j].x - poly[i].x) * (q.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)
        in = !in;
    return in;
  }
  bool onBoundary(P m) const {
    for (int i = 0; i < n; i++) {
      P u = poly[i], v = poly[(i + 1) % n];
      double L = std::hypot(v.x - u.x, v.y - u.y);
      if (L > 0 && std::fabs(cr(u, v, m)) / L < 1e-7 &&
          (m.x - u.x) * (m.x - v.x) + (m.y - u.y) * (m.y - v.y) <= 1e-9)
        return true;
    }
    return false;
  }
  // The open segment ab lies in the closed polygon.
  bool visible(P a, P b) const {
    for (int i = 0; i < n; i++)
      if (properX(a, b, poly[i], poly[(i + 1) % n])) return false;
    // A segment that only touches vertices passes the test above; sample it.
    static const double ts[] = {0.5, 0.25, 0.75, 0.01, 0.99};
    for (double t : ts) {
      P m{a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
      if (!onBoundary(m) && !inside(m)) return false;
    }
    return true;
  }

  // min over x in [0, len] of max_c min(d1_c + x, d2_c + len - x) / L_c
  double edgeMin(int e, const double* L, const unsigned char* mask, double* xbest,
                 std::vector<double>& a1, std::vector<double>& a2, std::vector<double>& w) const {
    double len = el[e];
    int m = 0;
    if (len >= 1e-9)
      for (int c = 0; c < nc; c++) {
        if (mask && !mask[c]) continue;
        double d1 = D[(size_t)e1[e] * nc + c], d2 = D[(size_t)e2[e] * nc + c];
        if (!(d1 < INF && d2 < INF)) continue;      // vertex cannot reach the region
        a1[m] = d1; a2[m] = d2 + len; w[m] = 1.0 / L[c]; m++;
      }
    if (m == 0) { *xbest = 0.0; return 0.0; }       // same convention as the Python code
    auto h = [&](double x) {
      double v = -INF;
      for (int c = 0; c < m; c++) v = std::max(v, std::min(a1[c] + x, a2[c] - x) * w[c]);
      return v;
    };
    double hb = h(0.0), xb = 0.0, hl = h(len);
    if (hl < hb) { hb = hl; xb = len; }
    // An interior local minimum of the envelope is a point where a decreasing
    // piece of one corner meets an increasing piece of another.
    for (int i = 0; i < m; i++)
      for (int j = 0; j < m; j++) {
        double x = (a2[j] * w[j] - a1[i] * w[i]) / (w[i] + w[j]);
        if (x > 0.0 && x < len) { double v = h(x); if (v < hb) { hb = v; xb = x; } }
      }
    *xbest = xb;
    return hb;
  }
};
}  // namespace

extern "C" {

void* ker_create(int n, const double* xy, int nc, const int* C, int nv, const double* D,
                 int ne, const int* e1, const int* e2, const double* el) {
  Ker* k = new Ker;
  k->n = n; k->nc = nc; k->nv = nv; k->ne = ne;
  k->poly.resize(n);
  for (int i = 0; i < n; i++) k->poly[i] = P{xy[2 * i], xy[2 * i + 1]};
  k->C.assign(C, C + nc);
  k->D.resize((size_t)nv * nc);
  for (size_t i = 0; i < (size_t)nv * nc; i++) k->D[i] = std::isfinite(D[i]) ? D[i] : INF;
  k->e1.assign(e1, e1 + ne); k->e2.assign(e2, e2 + ne); k->el.assign(el, el + ne);
  k->G.assign((size_t)n * n, INF);
  k->nxt.assign((size_t)n * n, -1);
  for (int i = 0; i < n; i++) {
    k->G[(size_t)i * n + i] = 0.0;
    k->nxt[(size_t)i * n + i] = i;
    for (int j = i + 1; j < n; j++)
      if (k->visible(k->poly[i], k->poly[j])) {
        k->G[(size_t)i * n + j] = k->G[(size_t)j * n + i] =
            std::hypot(k->poly[i].x - k->poly[j].x, k->poly[i].y - k->poly[j].y);
        k->nxt[(size_t)i * n + j] = j;
        k->nxt[(size_t)j * n + i] = i;
      }
  }
  for (int m = 0; m < n; m++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        double v = k->G[(size_t)i * n + m] + k->G[(size_t)m * n + j];
        if (v < k->G[(size_t)i * n + j]) {
          k->G[(size_t)i * n + j] = v;
          k->nxt[(size_t)i * n + j] = k->nxt[(size_t)i * n + m];
        }
      }
  return k;
}

void ker_destroy(void* h) { delete static_cast<Ker*>(h); }

// Geodesic distance from (ex, ey) to every corner. Returns the number of
// corners that could not be reached (0 when the point is inside the polygon).
int ker_path_lengths(void* h, double ex, double ey, double* out) {
  const Ker& k = *static_cast<Ker*>(h);
  P e{ex, ey};
  std::vector<double> dv(k.n);
  for (int v = 0; v < k.n; v++)
    dv[v] = k.visible(e, k.poly[v]) ? std::hypot(ex - k.poly[v].x, ey - k.poly[v].y) : INF;
  int bad = 0;
  for (int c = 0; c < k.nc; c++) {
    double b = INF;
    for (int v = 0; v < k.n; v++) b = std::min(b, dv[v] + k.G[(size_t)v * k.n + k.C[c]]);
    if (!(b < INF)) bad++;
    out[c] = b;
  }
  return bad;
}

// Best edge (first one in edge order among equal values), with the optimal
// value and the offset from its first vertex. mask (nc bytes, may be null)
// restricts the corners.
void ker_optimal_guard(void* h, const double* L, const unsigned char* mask,
                       int* v1, int* v2, double* alpha, double* offset) {
  const Ker& k = *static_cast<Ker*>(h);
  std::vector<double> a1(k.nc), a2(k.nc), w(k.nc);
  double best = INF, xb = 0.0; int eb = -1;
  for (int e = 0; e < k.ne; e++) {
    double x, v = k.edgeMin(e, L, mask, &x, a1, a2, w);
    if (v < best) { best = v; xb = x; eb = e; }
  }
  if (eb < 0) { eb = 0; best = 0.0; xb = 0.0; }
  *v1 = k.e1[eb]; *v2 = k.e2[eb]; *alpha = best; *offset = xb;
}

// Offset of the minimizer on the first edge (v1, v2); 0 if there is none.
double ker_edge_offset(void* h, const double* L, const unsigned char* mask, int v1, int v2) {
  const Ker& k = *static_cast<Ker*>(h);
  std::vector<double> a1(k.nc), a2(k.nc), w(k.nc);
  for (int e = 0; e < k.ne; e++)
    if (k.e1[e] == v1 && k.e2[e] == v2) {
      double x; k.edgeMin(e, L, mask, &x, a1, a2, w);
      return x;
    }
  return 0.0;
}

// Geodesic shortest path from a to b as points (a, polygon vertices, b),
// written to out (x0, y0, x1, y1, ...). Every polygon vertex that lies on the
// path is listed, also where the path does not bend, and repeated points are
// dropped. Returns the number of points, 0 if there is no path (a point
// outside the polygon), or -1 if cap is too small.
int ker_shortest_path(void* h, double ax, double ay, double bx, double by, double* out, int cap) {
  const Ker& k = *static_cast<Ker*>(h);
  P a{ax, ay}, b{bx, by};
  std::vector<P> raw;
  raw.push_back(a);
  if (!k.visible(a, b)) {
    std::vector<double> da(k.n), db(k.n);
    for (int v = 0; v < k.n; v++) {
      da[v] = k.visible(a, k.poly[v]) ? std::hypot(ax - k.poly[v].x, ay - k.poly[v].y) : INF;
      db[v] = k.visible(b, k.poly[v]) ? std::hypot(bx - k.poly[v].x, by - k.poly[v].y) : INF;
    }
    double best = INF; int bu = -1, bv = -1;
    for (int u = 0; u < k.n; u++) {
      if (!(da[u] < INF)) continue;
      for (int v = 0; v < k.n; v++) {
        if (!(db[v] < INF)) continue;
        double d = da[u] + k.G[(size_t)u * k.n + v] + db[v];
        if (d < best) { best = d; bu = u; bv = v; }
      }
    }
    if (bu < 0 || !(best < INF)) return 0;
    for (int u = bu;; u = k.nxt[(size_t)u * k.n + bv]) {
      raw.push_back(k.poly[u]);
      if (u == bv) break;
    }
  }
  raw.push_back(b);
  const double tol = 1e-7;
  std::vector<P> path;
  auto push = [&](P q) {
    if (path.empty() || std::hypot(q.x - path.back().x, q.y - path.back().y) > tol) path.push_back(q);
  };
  for (size_t i = 0; i + 1 < raw.size(); i++) {
    P p = raw[i], q = raw[i + 1];
    push(p);
    double L = std::hypot(q.x - p.x, q.y - p.y);
    if (L <= tol) continue;
    std::vector<std::pair<double, int>> on;       // polygon vertices on the open segment
    for (int v = 0; v < k.n; v++) {
      double t = ((k.poly[v].x - p.x) * (q.x - p.x) + (k.poly[v].y - p.y) * (q.y - p.y)) / (L * L);
      if (t * L > tol && (1.0 - t) * L > tol && std::fabs(cr(p, q, k.poly[v])) / L < tol)
        on.push_back({t, v});
    }
    std::sort(on.begin(), on.end());
    for (auto& tv : on) push(k.poly[tv.second]);
  }
  push(b);
  if (path.size() < 2) { path.clear(); path.push_back(a); path.push_back(b); }
  if ((int)path.size() > cap) return -1;
  for (size_t i = 0; i < path.size(); i++) { out[2 * i] = path[i].x; out[2 * i + 1] = path[i].y; }
  return (int)path.size();
}

}  // extern "C"

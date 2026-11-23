// Driver for the minimax solver of Antonio, Becerra and Murrieta-Cid (orig/Minmax.hpp) on one
// of our polygons. The solver is used as provided, with the three-line patch applied by build.sh.
//   mode free    : pursuer samples anywhere in the polygon (the setting of their paper)
//   mode roadmap : pursuer samples on our patrol roadmap, neighbors by distance along the roadmap
// usage: ampe_poly poly.txt roadmap.txt N_e mode ratio(s_p/s_e) [seed] [spacing] [out_prefix]
#include "Minmax_patched.hpp"
#include "quasirand.hpp"
using namespace std;
using namespace quasirand;

int main(int argc, char** argv) {
    if (argc < 6) { cerr << "args" << endl; return 1; }
    string polyf = argv[1], roadf = argv[2];
    int Ne = stoi(argv[3]); string mode = argv[4]; double ratio = stod(argv[5]);
    int seed = argc > 6 ? stoi(argv[6]) : 1;
    double spacing = argc > 7 ? stod(argv[7]) : 3.0;
    string dump = argc > 8 ? argv[8] : "";
    omp_set_num_threads(omp_get_num_procs() - 2);

    ifstream pf(polyf); int n; pf >> n; vector<pdd> poly(n);
    for (auto& v : poly) pf >> v.first >> v.second;
    Environment env(poly);
    double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300, area = 0;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        minx = min(minx, poly[i].first); maxx = max(maxx, poly[i].first);
        miny = min(miny, poly[i].second); maxy = max(maxy, poly[i].second);
        area += poly[j].first * poly[i].second - poly[i].first * poly[j].second;
    }
    area = fabs(area) / 2;

    auto sample = [&](int N, double s) {
        vector<multidim_state> P; QuasiRandom<double> q(2, s);
        while ((int)P.size() < N) {
            auto u = q(); double x = minx + u[0] * (maxx - minx), y = miny + u[1] * (maxy - miny);
            if (env.strictlyInside(x, y) && env.distBoundary(x, y) > 1e-3) P.push_back(multidim_state(vector<double>{x, y}));
        }
        return P;
    };
    vector<multidim_state> Pe = sample(Ne, 0.5 + 0.001 * seed), Pp;
    double gamma = 2.0 * sqrt(1.5) * sqrt(area / pi);   // PRM* constant in 2D with the true free area
    double cnt_p = gamma;

    // roadmap samples and their adjacency (distance along the roadmap)
    vector<vector<pid>> radj;
    if (mode == "roadmap") {
        ifstream rf(roadf); int m; rf >> m;
        vector<array<double, 4>> segs(m); for (auto& s : segs) rf >> s[0] >> s[1] >> s[2] >> s[3];
        vector<pdd> pts; auto id = [&](double x, double y) {
            for (int i = 0; i < (int)pts.size(); i++) if (hypot(pts[i].first - x, pts[i].second - y) < 1e-3) return i;
            pts.push_back({x, y}); return (int)pts.size() - 1; };
        vector<array<int, 2>> ed; double Ltot = 0;
        for (auto& s : segs) {
            double L = hypot(s[2] - s[0], s[3] - s[1]); Ltot += L;
            int k = max(1, (int)ceil(L / spacing)); int prev = id(s[0], s[1]);
            for (int i = 1; i <= k; i++) {
                double t = (double)i / k; int cur = id(s[0] + t * (s[2] - s[0]), s[1] + t * (s[3] - s[1]));
                ed.push_back({prev, cur}); prev = cur;
            }
        }
        radj.resize(pts.size());
        for (auto& e : ed) { double d = hypot(pts[e[0]].first - pts[e[1]].first, pts[e[0]].second - pts[e[1]].second);
            radj[e[0]].push_back({e[1], d}); radj[e[1]].push_back({e[0], d}); }
        for (auto& p : pts) Pp.push_back(multidim_state(vector<double>{p.first, p.second}));
        cerr << "roadmap length " << Ltot << ", pursuer samples " << Pp.size() << endl;
        cnt_p = 0.0;
        g_ext_ngbs_p = [&](double R) {
            int N = radj.size(); vector<vector<pid>> ng(N);
            #pragma omp parallel for
            for (int s = 0; s < N; s++) {
                map<int, double> dist; priority_queue<pdi, vector<pdi>, greater<pdi>> pq; pq.push({0.0, s}); dist[s] = 0;
                while (!pq.empty()) { auto [d, u] = pq.top(); pq.pop(); if (d > dist[u] + 1e-12) continue;
                    for (auto& [v, w] : radj[u]) { double nd = d + w; if (nd > R) continue;
                        auto it = dist.find(v); if (it == dist.end() || nd < it->second - 1e-12) { dist[v] = nd; pq.push({nd, v}); } } }
                for (auto& [v, d] : dist) if (v != s) ng[s].push_back({v, d});
            }
            return ng;
        };
    } else {
        Pp = sample(Ne, 0.25 + 0.001 * seed);
    }

    {   // diagnostics: evader samples that no pursuer sample sees; components of the roadmap graph
        int unseen = 0;
        #pragma omp parallel for reduction(+:unseen)
        for (int e = 0; e < (int)Pe.size(); e++) { bool s = false;
            for (int p = 0; p < (int)Pp.size() && !s; p++) s = !env.intersection(Pp[p].coords, Pe[e].coords);
            if (!s) unseen++; }
        int comps = 0; vector<char> seen(radj.size(), 0);
        for (int s0 = 0; s0 < (int)radj.size(); s0++) if (!seen[s0]) { comps++; vector<int> st = {s0}; seen[s0] = 1;
            while (!st.empty()) { int u = st.back(); st.pop_back(); for (auto& [v, w] : radj[u]) if (!seen[v]) { seen[v] = 1; st.push_back(v); } } }
        fprintf(stderr, "DIAG evader_samples_unseen=%d roadmap_components=%d\n", unseen, comps);
    }
    MinMaxGame game(env, Pe, Pp, gamma, cnt_p, 1.0 / ratio);   // speed_pe = Ve / Vp

    int sp = game.size_p, se = game.size_e; long long vis0 = 0, win = 0; int hold = 0; double ne = 0, np = 0;
    for (auto& v : game.ngbs_e) ne += v.size(); for (auto& v : game.ngbs_p) np += v.size();
    vector<char> holdable(se, 0);
    for (int p = 0; p < sp; p++) for (int e = 0; e < se; e++) {
        auto& s = game.E[p][e];
        if (s.t.second - s.t.first > 0) { vis0++; if (s.best_sol > 0) { win++; holdable[e] = 1; } }
    }
    for (char h : holdable) hold += h;
    printf("RESULT mode=%s ratio=%.3f Ne=%d Np=%d Rprm_e=%.2f Rprm_p=%.2f ngb_e=%.1f ngb_p=%.1f sweeps=%lld "
           "visible_states=%lld pursuer_wins=%.4f evader_positions_holdable=%.4f\n",
           mode.c_str(), ratio, se, sp, game.Rprm_e, game.Rprm_p, ne / se, np / sp, game.cont,
           vis0, (double)win / vis0, (double)hold / se);
    if (!dump.empty()) {
        // raw tables for benchmark/ampe_pursuer.py
        auto W = [&](const string& ext, const void* d, size_t bytes) {
            FILE* f = fopen((dump + ext).c_str(), "wb"); fwrite(d, 1, bytes, f); fclose(f); };
        vector<double> pp, pe; for (auto& P : Pp) { pp.push_back(P.coords[0]); pp.push_back(P.coords[1]); }
        for (auto& P : Pe) { pe.push_back(P.coords[0]); pe.push_back(P.coords[1]); }
        W(".pp", pp.data(), pp.size() * 8); W(".pe", pe.data(), pe.size() * 8);
        vector<float> V((size_t)sp * se); vector<int> NX((size_t)sp * se);
        for (int p = 0; p < sp; p++) for (int e = 0; e < se; e++) {
            V[(size_t)p * se + e] = (float)game.E[p][e].best_sol; NX[(size_t)p * se + e] = game.E[p][e].next_state.first; }
        W(".V", V.data(), V.size() * 4); W(".nx", NX.data(), NX.size() * 4);
        vector<int> off = {0}, idx; for (auto& v : game.ngbs_p) { for (auto& q : v) idx.push_back(q.first); off.push_back(idx.size()); }
        W(".nbo", off.data(), off.size() * 4); W(".nbi", idx.data(), idx.size() * 4);
        FILE* f = fopen((dump + ".meta").c_str(), "w");
        fprintf(f, "{\"mode\": \"%s\", \"ratio\": %.6f, \"Np\": %d, \"Ne\": %d, \"Rprm_e\": %.6f, \"Rprm_p\": %.6f, "
                   "\"pursuer_wins\": %.6f, \"sweeps\": %lld, \"seed\": %d, \"spacing\": %.3f}\n",
                mode.c_str(), ratio, sp, se, game.Rprm_e, game.Rprm_p, (double)win / vis0, game.cont, seed, spacing);
        fclose(f);
    }
    return 0;
}

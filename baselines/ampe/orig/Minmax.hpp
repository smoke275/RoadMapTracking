#include <bits/stdc++.h>
#include <omp.h>
#include <future>
#include <dirent.h>
#include "utils.hpp"
#include "Environment.hpp"
//#include "Environment3D.hpp"
using namespace std;

#define ll long long
#define pii pair<int, int>
#define pid pair<int, double>
#define pdi pair<double, int>
#define pdd pair<double, double>

const int MAXN = 50000;
const double pi  = 3.1415926535;
const double dINF = 1000000000.0;

struct multidim_state
{
    int d;
    vector<double> coords;

    multidim_state(int dim)
    {
        d = dim;
        coords.resize(d);
    }

    multidim_state(vector<double> c)
    {
        coords = c;
        d = coords.size();
    }

    double dist(const multidim_state &s)
    {
        // Distance function between two states
        double dist = 0.0;
        for (int i = 0; i < d; i++)
            dist += ((coords[i] - s.coords[i]) * (coords[i] - s.coords[i]));
        return sqrt(dist);
    }
};

struct node
{
    double best_sol, sub_sol, prev_sol;
    //double minmax_gain, maxmin_gain;

    pdd t;
    bool in_queue;
    int ev_shortest_esc;
    pii index_sol, next_state;
};

class MinMaxGame
{
public:
    Environment env;
    ll cont;
    int size_e, size_p;
    double speed_ratio;
    double Rprm_e, Rprm_p;

    chrono::high_resolution_clock::time_point start_time;

    vector<vector<pid>> ngbs_e, ngbs_p;
    vector<multidim_state> pose_e, pose_p;
    vector<vector<node>> E, E_maxmin;

    MinMaxGame( Environment world,
                vector<multidim_state> &Pe, vector<multidim_state> &Pp,
                double rprm_cnt_e, double rprm_cnt_p, double speed_pe = 1.0);

    void run();
    void start_costs();
    void twoplay_minmax();
    void print_paths(int p, int e);
    pii find_cost_root(int p, int e);
    void plot_path(int p, int e, int nmap);
    void print_elapsed_time(bool end_time = false);
    void neighbors(vector<multidim_state> &poses, vector<vector<pid>> &ngbs, double Rprm, double min_dist = 0.0);

    int minimax_moves();
    int minimax_moves_up();
    int maximin_moves();

    bool minmax_step(int p, int e);
    bool minmax_up_step(int p, int e);
};

MinMaxGame::MinMaxGame( Environment world,
                        vector<multidim_state> &Pe, vector<multidim_state> &Pp,
                        double rprm_cnt_e, double rprm_cnt_p, double speed_pe)
{
    env = world;
    pose_e = Pe, pose_p = Pp;
    speed_ratio = speed_pe; // Ve/Vp

    size_e = pose_e.size();
    size_p = pose_p.size();

    double norm_test = (speed_pe > 1.0 / 0.7071) ? speed_pe : 1.0 / 0.7071;

    double rad_rprm = max( 1.01 * rprm_cnt_e * sqrt(log(size_e) / size_e),
                           1.01 * rprm_cnt_p * sqrt(log(size_p) / size_p) );

    Rprm_e = rad_rprm * norm_test;
    Rprm_p = rad_rprm * norm_test / speed_pe;

    cout << "Rprm evader : " << Rprm_e << " pursuer : " << Rprm_p << endl;

    run();
}

void MinMaxGame::print_elapsed_time(bool end_time){
    auto current_time = chrono::high_resolution_clock::now();
    auto time     = current_time - start_time;
    double execution_time = time/chrono::milliseconds(1);

    string lbl = (end_time) ? "Execution time : " : "Elapsed time : ";
    cout << lbl << fixed << setprecision(3) << execution_time / 1000.0 << "s" << endl;
}

void MinMaxGame::run()
{
    cout << "Runnnig... " << endl;
    start_time = chrono::high_resolution_clock::now();

    // Start E table
    E.resize(size_p);
    for (int p = 0; p < size_p; p++)
        E[p].resize(size_e);


    // Generate neighborhoods
    cout << "\t" << "Generating neighbors... " << endl;
    neighbors(pose_e, ngbs_e, Rprm_e, 0.7071 * Rprm_e);
    neighbors(pose_p, ngbs_p, Rprm_p);


    // Compute the costs te - tp
    cout << "\t" << "Starting costs... " << endl;
    start_costs();
    cout << "\tStart costs updated" << endl
         << "\t";
    print_elapsed_time();

    cout << "Start cost : " << E[size_p-1][size_e-1].best_sol << endl;
    cout << "Start cost : " << E[size_p-2][size_e-2].best_sol << endl;

    cout << "\tPlaying Min-Max game..." << endl;
    twoplay_minmax();

    cout << "Done" << endl;
    print_elapsed_time(true);
}

void MinMaxGame::neighbors(vector<multidim_state> &poses,
                           vector<vector<pid>> &ngbs,
                           double Rprm, double min_dist)
{
    int psize = poses.size();
    ngbs.resize(psize);

    #pragma omp parallel for
    for (int x = 0; x < psize; x++)
    {
        vector<pid> X; // possible neighbors
        vector<bool> joined;

        X.push_back({x, 0.0});
        for (int y = 0; y < psize; y++)
        {
            if (x == y || poses[x].dist(poses[y]) > Rprm)
                continue;

            X.push_back({y, dINF});

        }
        joined.resize(X.size());

        priority_queue<pdi, vector<pdi>, greater<pdi>> PQ_min;
        PQ_min.push({0.0, 0});

        while (!PQ_min.empty())
        {
            auto s = PQ_min.top();
            PQ_min.pop();

            int xn = s.second;
            if (joined[xn])
                continue;
            joined[xn] = true;

            for (int i = 1; i < X.size(); i++)
            {
                int y = X[i].first;
                if (joined[i] || env.intersection(poses[X[xn].first].coords, poses[y].coords) )
                    //env.intersection(poses[X[xn].first].coords[0], poses[X[xn].first].coords[1],
                    //                 poses[y].coords[0], poses[y].coords[1]))
                    
                    continue;

                double dist_xy = s.first + poses[X[xn].first].dist(poses[y]);

                if (dist_xy < X[i].second - eps)
                {
                    X[i].second = dist_xy;
                    PQ_min.push({dist_xy, i});
                }
            }
        }

        for (int y = 1; y < X.size(); y++)
        {
            if (X[y].second <= Rprm)
            {
                ngbs[x].push_back(X[y]);
            }
        }
    } 
}

void MinMaxGame::start_costs()
{
    // 1. (INF, 0) if there are not visibility [Tp, Te]
    //    or (0, INF) if there are visibility
    for (int p = 0; p < size_p; p++)
    {
        for (int e = 0; e < size_e; e++)
        {
            E[p][e].index_sol = {p, e};
            E[p][e].next_state = {p, e};

            //if (env.intersection(pose_p[p].coords[0], pose_p[p].coords[1], pose_e[e].coords[0], pose_e[e].coords[1])){
            if ( env.intersection(pose_p[p].coords, pose_e[e].coords) ){
                E[p][e].t = {dINF, 0.0};
            }
            else{
                E[p][e].t = {0.0, dINF};
            }
        }
    }

    // 2. Run Dijkstra to start costs

    // 2.1 Runnning Dijkstra in pursuer costs
    #pragma omp parallel for
    for (int e = 0; e < size_e; e++)
    {
        bitset<MAXN> vis;
        vis.reset();

        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

        for (int p = 0; p < size_p; p++)
        {
            if (E[p][e].t.second - E[p][e].t.first > 0.0)
                pq.push({0.0, p});
        }

        while (!pq.empty())
        {
            auto s = pq.top();
            pq.pop();

            if (vis[s.second])
                continue;
            vis[s.second] = true;

            E[s.second][e].t.first = s.first;
            for (auto pp : ngbs_p[s.second])
            {
                double dist_p_pp = s.first + pose_p[s.second].dist(pose_p[pp.first]);

                if (dist_p_pp < E[pp.first][e].t.first)
                {
                    E[pp.first][e].t.first = dist_p_pp;
                    pq.push({dist_p_pp, pp.first});
                }
            }
        }
    }
    cout << "\t\tPursuer costs updated" << endl;

    // 2.2 Runnning Dijkstra in evader costs
    #pragma omp parallel for
    for (int p = 0; p < size_p; p++)
    {
        bitset<MAXN> vis;
        vis.reset();

        priority_queue<pair<double, pii>, vector<pair<double, pii>>, greater<pair<double, pii>>> pq;

        for (int e = 0; e < size_e; e++)
        {
            if (E[p][e].t.second - E[p][e].t.first < 0.0)
            {
                pq.push({0.0, {e, e}});
                E[p][e].ev_shortest_esc = e;
            }
        }

        while (!pq.empty())
        {
            auto s = pq.top();
            pq.pop();

            if (vis[s.second.first])
                continue;
            vis[s.second.first] = true;

            E[p][s.second.first].t.second = s.first;
            E[p][s.second.first].ev_shortest_esc = s.second.second;

            for (auto ee : ngbs_e[s.second.first])
            {
                double dist_e_ee = s.first + pose_e[s.second.first].dist(pose_e[ee.first]);

                if (dist_e_ee < E[p][ee.first].t.second)
                {
                    E[p][ee.first].t.second = dist_e_ee;
                    pq.push({dist_e_ee, {ee.first, s.second.first}});
                }
            }
        }
    }
    cout << "\t\tEvader costs updated" << endl;

    int ev = 0, pu = 0;
    //#pragma omp parallel for
    for (int p = 0; p < size_p; p++)
    {
        for (int e = 0; e < size_e; e++)
        {
            //E[p][e].minmax_ngbs = (int)ngbs_p[p].size() * ngbs_e[e].size();

            E[p][e].index_sol = {p, e};
            //E[p][e].minmax_gain = E[p][e].maxmin_gain = E[p][e].t.second - E[p][e].t.first;
            E[p][e].best_sol = E[p][e].t.second - E[p][e].t.first;
            E[p][e].sub_sol = E[p][e].t.second - E[p][e].t.first;
            E[p][e].prev_sol = E[p][e].t.second - E[p][e].t.first;

            if(E[p][e].best_sol < 0.0)
                ev ++;
            else
                pu ++;
        }
    }

    cout << "Evader : " << ev << " Pursuer : " << pu << endl;

}

int MinMaxGame::minimax_moves()
{
    vector<pii> state_queue, next_queue;

    for (int p = 0; p < size_p; p++)
    {
        for (int e = 0; e < size_e; e++)
        {
            //E[p][e].in_queue = true;
            state_queue.push_back({p, e});
        }
    }

    //sort(state_queue.begin(), state_queue.end());

    cont = 0;
    while (!state_queue.empty())
    {
        cont ++;
        cout << "Step : " << cont << ", queue size : " << state_queue.size() << endl;

        #pragma omp parallel for
        for(int i=0; i<state_queue.size(); i++)
        {
            auto current_state = state_queue[i];

            int p = current_state.first,
                e = current_state.second;

            bool change = true;
            if (E[p][e].t.second - E[p][e].t.first > 0.0)
            {
                change = minmax_step(p, e);
            }
            else{
                continue;
            }
            
            if(!change)
                continue;

            for (pid pp : ngbs_p[p])
            {
                for (pid ee : ngbs_e[e])
                {   
                    if(pp.first == p && ee.first == e)
                        continue;

                    #pragma omp atomic
                    E[pp.first][ee.first].in_queue |= true;
                }
            }
        }

        next_queue.clear();
        for (int p = 0; p < size_p; p++)
        {
            for (int e = 0; e < size_e; e++)
            {
                if( E[p][e].in_queue )
                {
                    E[p][e].in_queue = false;
                    next_queue.push_back({p, e});
                }

                E[p][e].prev_sol = E[p][e].best_sol;
            }
        }
        state_queue.swap(next_queue);
    }

    return 0;
}

int MinMaxGame::minimax_moves_up()
{
    #pragma omp parallel for 
    for (int p = 0; p < size_p; p++)
    {
        for (int e = 0; e < size_e; e++)
        {
            if (abs(E[p][e].best_sol - (E[p][e].t.second - E[p][e].t.first)) < eps && E[p][e].best_sol > 0.0)
            {
                minmax_up_step(p, e);
            }
        }
    }
    return 0;
}

bool MinMaxGame::minmax_up_step(int p, int e)
{
    // minmax for self-loop states
    pair<pdd, pii> minmax_sol = {{2.0*dINF, 2.0*dINF}, {-1, -1}};

    for (auto ee : ngbs_e[e])
    {
        if (ee.first == e || ee.second < 0.7071 * Rprm_e )
            continue;

        pair<pdd, int> max_sol = {{-dINF, -dINF}, -1};
        for (auto pp : ngbs_p[p])
        {
            if (pp.first == p || ee.second < speed_ratio * pp.second)
                continue;

           pdd pp_ee_bestsol = {E[pp.first][ee.first].best_sol, E[p][ee.first].best_sol};

            if (pp_ee_bestsol > max_sol.first)
            {
                max_sol.first = pp_ee_bestsol,
                max_sol.second = pp.first;
            }
        }

        if (max_sol.first.first > -dINF && max_sol.first < minmax_sol.first)
        {
            minmax_sol.first = max_sol.first;
            minmax_sol.second = {max_sol.second, ee.first};
        }
    }

    // Didnt find any solution
    if(minmax_sol.second.first == -1 || minmax_sol.second.second == -1)
        return false;

    if(p == size_p -2 && e == size_e -2){
        cout << "\t" << "Info log : " << p << " " << e  << " state." << endl;
        cout << "\t" << "Prev sol : " << E[p][e].best_sol << " New sol : " << minmax_sol.first.first << ", " << E[p][e].sub_sol << endl;
    }

    E[p][e].sub_sol = minmax_sol.first.second;
    E[p][e].next_state = minmax_sol.second;
    //E[p][e].index_sol = E[minmax_sol.second.first][minmax_sol.second.second].index_sol;

    return true;
}

bool MinMaxGame::minmax_step(int p, int e){
    // minmax for evader
    pair<double, pii> minmax_sol = {dINF, {-1, -1}};
    //find_cost_root(p, e);

    for (auto ee : ngbs_e[e])
    {
        if (ee.first == e )
            continue;

        pdi max_sol = {-dINF, -1};
        double max_p_dist = 1.1 * max(ee.second, 0.7071 * Rprm_e);

        for (auto pp : ngbs_p[p])
        {
            if (pp.first == p || max_p_dist < speed_ratio * pp.second)
                continue;

            // Compute best sol for pp, ee
            /*
            auto pp_ee_indexsol = find_cost_root(pp.first, ee.first);
            auto pp_ee_tsol = E[pp_ee_indexsol.first][pp_ee_indexsol.second].t;
            double pp_ee_bestsol = pp_ee_tsol.second - pp_ee_tsol.first;
            */
           double pp_ee_bestsol = E[pp.first][ee.first].prev_sol;

            if (pp_ee_bestsol > max_sol.first)
            {
                max_sol.first = pp_ee_bestsol,
                max_sol.second = pp.first;
            }
        }

        if (max_sol.first > -dINF && max_sol.first < minmax_sol.first)
        {
            minmax_sol.first = max_sol.first;
            minmax_sol.second = {max_sol.second, ee.first};
        }
    }

    // Didnt find any solution
    if(minmax_sol.second.first == -1 || minmax_sol.second.second == -1)
        return false;

    // There is no update
    if( E[p][e].best_sol < minmax_sol.first + eps)
    {
        //E[p][e].next_state = minmax_sol.second;
        //E[p][e].index_sol = E[minmax_sol.second.first][minmax_sol.second.second].index_sol;

        return false;
    }

    if(p == size_p -2 && e == size_e -2){
        cout << "\t" << "Info log : " << p << " " << e  << " state." << endl;
        cout << "\t" << "Prev sol : " << E[p][e].best_sol << " New sol : " << minmax_sol.first << endl;
    }

    E[p][e].best_sol = minmax_sol.first;
    E[p][e].next_state = minmax_sol.second;
    E[p][e].index_sol = E[minmax_sol.second.first][minmax_sol.second.second].index_sol;

    return true;
}




/*
int MinMaxGame::maximin_moves()
{
    deque<pair<double, pii>> state_pq;

    for (int p = 0; p < size_p; p++)
    {
        for (int e = 0; e < size_e; e++)
        {
            state_pq.push_back({E_maxmin[p][e].best_sol, {p, e}});
        }
    }

    sort(state_pq.begin(), state_pq.end(), greater<pair<double, pii>>());

    while (!state_pq.empty())
    {
        auto current_state = state_pq.front();
        state_pq.pop_front();

        int p = current_state.second.first,
            e = current_state.second.second;

        bool change = true;
        if (E_maxmin[p][e].t.second - E_maxmin[p][e].t.first > 0.0)
        {
            change = maxmin_step(p, e);
        }
        else{
            continue;
        }
        
        if(!change)
            continue;

        for (pid pp : ngbs_p[p])
        {
            for (pid ee : ngbs_e[e])
            {   
                if(pp.first == p && ee.first == e)
                    continue;

                if(!E_maxmin[pp.first][ee.first].in_queue){
                    E_maxmin[pp.first][ee.first].in_queue = true;
                    state_queue.push({E_maxmin[pp.first][ee.first].best_sol, {pp.first, ee.first}});
                }
            }
        }
    }
}
*/

void MinMaxGame::twoplay_minmax()
{
    /*
    // Evader  - plays minmax for the rest of the game
    // Pursuer - plays maxmin for the rest of the game 

    auto minimax_status = std::async(std::launch::async, &MinMaxGame::minimax_moves, this);
    auto maximin_status = maximin_moves();

    if(maximin_status == 0 && minimax_status.get() == 0)
        cout << "Async finished ok" << endl;
    else
        cout << "Something went wrong : " << endl;
    */


   // Minimax 
   // lower bounding evaders step size
   // upper bounding pursuer step size
   minimax_moves();
   minimax_moves_up(); 
}

pii MinMaxGame::find_cost_root(int p, int e){
    // Update parent and value
    int steps = 0;
    int pp = p, ee = e;
    while(pp != E[pp][ee].index_sol.first || ee != E[pp][ee].index_sol.second){
        auto idx_sol = E[pp][ee].next_state;
        pp = idx_sol.first,
        ee = idx_sol.second;

        steps ++;
        if(steps > 500)
            cout << pp << " " << ee << " " << E[pp][ee].best_sol << " " 
            << E[pp][ee].t.second - E[pp][ee].t.first << endl;
        if(steps > 505){
            string aa;
            cin >> aa;
        }
    }
    E[p][e].best_sol = E[pp][ee].t.second - E[pp][ee].t.first;
    E[p][e].index_sol = {pp, ee};

    return {pp, ee};
}

void MinMaxGame::print_paths(int p, int e)
{
    
    // Debug : print random points and costs
    /*
    int num_prints = 25;
    for (int i = 0; i < num_prints; i++)
    {
        int pp = rand() % size_p,
            ee = rand() % size_e;

        cout << "[";
        for (auto x : pose_p[pp].coords)
            cout << x << ", ";
        for (auto x : pose_e[ee].coords)
            cout << x << ", ";
        cout << E[pp][ee].best_sol << ", ";
        //cout << env.intersection(pose_p[pp].coords[0], pose_p[pp].coords[1], pose_e[ee].coords[0], pose_e[ee].coords[1]) << " ";
        cout << env.intersection(pose_p[pp].coords, pose_e[ee].coords) << " ";
        cout << "],";
        cout << endl;
    }
    */
    // End debug


    //for(int i=0; i<size_p; i++)
    //    for(int j=0; j<size_e; j++){
    //        p = i, e = j;

            auto last_idx = E[p][e].index_sol;
            int prev_p = p, prev_e = e;

    //        if((p == last_idx.first && e == last_idx.second) || E[p][e].best_sol > 0.0)
    //            continue;

            int steps = 0;
            do
            {
                steps ++;
                cout << p << ", " << e << " - ";
                cout << last_idx.first << ", " << last_idx.second << " - ";

                for (auto x : pose_p[p].coords)
                    cout << x << ", ";
                for (auto x : pose_e[e].coords)
                    cout << x << ", ";
                cout << " - " << E[p][e].best_sol << ", ";
                cout << E[p][e].t.second - E[p][e].t.first;
                cout << endl;

                auto next_idx = E[p][e].next_state;

                prev_p = p,
                prev_e = e;

                p = next_idx.first,
                e = next_idx.second;

                if(steps > 50)
                    break;
            }while (prev_p != last_idx.first || prev_e != last_idx.second);

            cout << endl;

            return;
    //    }
}

bool folder_exists(string path){
    DIR* dir = opendir(path.c_str());
    if (dir) {
        /* Directory exists. */
        closedir(dir);
        return true;
    } else if (ENOENT == errno) {
        /* Directory does not exist. */
        return false;
    } else {
        /* opendir() failed for some other reason. */
        cout << "Could not open folder path : " << path << endl;
        return false;
    }
}

void MinMaxGame::plot_path(int p, int e, int nmap){
    int steps = 0;
    int pp = p, ee = e;
    int nxt_pp = E[p][e].next_state.first,
        nxt_ee = E[p][e].next_state.second;

    vector<double> px, py, ex, ey;
    //while(pp != E[pp][ee].index_sol.first || ee != E[pp][ee].index_sol.second)
    while (pp != nxt_pp || ee != nxt_ee)
    {
        px.push_back(pose_p[pp].coords[0]);
        py.push_back(pose_p[pp].coords[1]);
        ex.push_back(pose_e[ee].coords[0]);
        ey.push_back(pose_e[ee].coords[1]);

        auto idx_sol = E[pp][ee].next_state;
        pp = idx_sol.first,
        ee = idx_sol.second;
        nxt_pp = E[pp][ee].next_state.first,
        nxt_ee = E[pp][ee].next_state.second;

        steps++;
        if (steps > 200)
            cout << pp << " " << ee << " " << E[pp][ee].best_sol << " "
                 << E[pp][ee].t.second - E[pp][ee].t.first << endl;
        if (steps > 205)
        {
            cout << "Cycled! " << endl;
            break;
        }

        // if( abs((E[pp][ee].t.second - E[pp][ee].t.first) - E[pp][ee].best_sol) < eps )
        //     break;
    }
    px.push_back(pose_p[pp].coords[0]);
    py.push_back(pose_p[pp].coords[1]);
    ex.push_back(pose_e[ee].coords[0]);
    ey.push_back(pose_e[ee].coords[1]);
    

    string server_folder = "/home/est_posgrado_emmanuel.antonio/OptimalPlanningHolonomic/outputs";
    string prefix = (folder_exists(server_folder)) ? server_folder : "./plots";

    cout << "Prefix pathfile : " << prefix << endl;

    string compfilename = prefix + "/" +
                            to_string(nmap) + "_" +  
                            to_string(pose_p.size()) + "x" + to_string(pose_e.size()) + "_" +
                            to_string(p) + "x" + to_string(e) + "_" + 
                            to_string(cont)  + ".txt";

    ofstream pathfiles;
    pathfiles.open(compfilename);

    for(int i=0; i<px.size(); i++){
        pathfiles << px[i] << "\t" << py[i] << "\t"
                    << ex[i] << "\t" << ey[i] << endl;
    }
    pathfiles.close();
}
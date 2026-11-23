#include "Minmax.hpp"
#include "quasirand.hpp"

//#include<Windows.h>
//#include<Psapi.h>
#include <dirent.h>
#include <errno.h>

#define mp make_pair
#define pii pair<int, int>
#define pdd pair<double, double>
using namespace std;
using namespace quasirand;

std::mt19937 gen;
std::uniform_real_distribution<> gen_rand(0.0, 1.0);

Environment create_env(int env_case = 1)
{
    // Create obstacles
    if (env_case == 1)
    {
        // One corner case
        vector<pdd> o1 = {{5.0, 5.0}, {10.0, 5.0}, {10.0, 0.0}, {5.0, 0.0}};
        vector<vector<pdd>> obss = {o1};

        return Environment(obss);
    }

    if (env_case == 2)
    {
        // Two corner case
        vector<pdd>
            o1 = {{5.0, 4.0}, {6.0, 5.0}, {5.0, 6.0}},
            o2 = {{0.0, 10.0}, {10.0, 7.3}, {10.0, 10.0}},
            o3 = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 2.6}};
            //o4 = {{0.0, 0.0}, {2.0, 0.0}, {2.0, 10.0}, {0.0, 10.0}};
        vector<vector<pdd>> obss = {o1};

        return Environment(obss);
    }

    if (env_case == 3)
    {
        // Circle in the middle case
        vector<pdd> o1;
        double sides = 100;
        for (int i = 0; i < sides; i++)
            o1.push_back({
                5.0 + cos(2 * pi * i / sides),
                5.0 + sin(2 * pi * i / sides),
            });
        vector<vector<pdd>> obss = {o1};

        return Environment(obss);
    }

    return Environment();
}

double fill_pts_2D(Environment env, vector<multidim_state>& P, int samples, int nmap = 2)
{
    multidim_state pt(2);
    QuasiRandom<double> qrng(2);

    double free_samples = 0.0,
           total_samples = 0.0;

    for (int i = 0; i < samples - 1; i++)
    {
        double x, y;

        do
        {
            if(nmap == 2)
            {
                x = gen_rand(gen) *  5.9,
                y = gen_rand(gen) *  10.0;
            }
            else{
                //x = gen_rand(gen) *  10.0,
                //y = gen_rand(gen) *  10.0;
                vector<double> point = qrng();
                x = point[0] * 10.0;
                y = point[1] * 10.0;
            }

            total_samples += 1.0;
        } while (env.withinObstacle(x, y));
        free_samples += 1.0;

        pt.coords[0] = x, pt.coords[1] = y;
        P.push_back(pt);
    }

    double xfree = free_samples / total_samples;
    double rprm_cte = 2.0 * sqrt(1.0 + 1.0 / 2.0) * sqrt(23.6 * xfree / pi);

    return rprm_cte;
}

void player_poses_tofile(int nmap, vector<multidim_state>& Po, string pl)
{
    string server_folder = "/home/est_posgrado_emmanuel.antonio/OptimalPlanningHolonomic/outputs";
    string prefix = (folder_exists(server_folder)) ? server_folder : "./plots";

    string compfilename = prefix + "/" +
                            to_string(nmap) + "_" +  
                            to_string(Po.size()) + "x" + to_string(Po.size()) + "_" +
                            "poses_" + pl + ".txt";

    ofstream pathfiles;
    pathfiles.open(compfilename);

    for(auto p : Po)
        pathfiles << p.coords[0] << "\t" << p.coords[1] << endl;

    pathfiles.close();
}

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        std::cout << "Specify the number of samples and map" << endl;
        return 1;
    }

    if(argc <= 2){
        std::cout << "Specify map number" << endl;
        return 1;
    }

    int samples = stoi(argv[1]);
    int nmap = stoi(argv[2]);

    
    std::random_device rd;  
    gen.seed(time(nullptr));
    srand(time(NULL));
    

    // Set how many threads to use
    // Hasta ahora la mejor opción: 12
    int num_threads = (samples >= 20000) ? 4 : 12;
    omp_set_num_threads(num_threads);

    // Select environment
    auto env = create_env(nmap);

    // Generate Pursuer / Evader points and Rprm constants
    double rprm_cnt_e, rprm_cnt_p;
    vector<multidim_state> Pe, Pp;

    rprm_cnt_e = fill_pts_2D(env, Pe, samples, nmap); 
    rprm_cnt_p = fill_pts_2D(env, Pp, samples, nmap);

    double speed_ratio = 1.0;
    
    if( nmap == 1 )
    {   // One corner case starting points
        {
            multidim_state  start_e( {4.0, 4.0} ), 
                            start_p( {2.0, 3.0} );
            Pe.push_back(start_e),
            Pp.push_back(start_p);
        }
        {
            multidim_state  start_e( {4.0, 3.0} ), 
                            start_p( {2.0, 3.0} );
            Pe.push_back(start_e),
            Pp.push_back(start_p);
        }
    }

    if( nmap == 2 )
    {   // Two corner case starting points
        {
            multidim_state  start_e( {3.2679, 5.0} ), 
                            start_p( {1.0, 5.0} );
            Pe.push_back(start_e),
            Pp.push_back(start_p);
        }
        {
            multidim_state  start_e( {3.5, 5.0} ), 
                            start_p( {1.0, 5.0} );
            Pe.push_back(start_e),
            Pp.push_back(start_p);
        }

        speed_ratio = 0.565684; // Pursuer faster
    }
    
    if( nmap == 3 )
    {   // Circle in the middle starting points
        {
            multidim_state  start_e( {5.0, 3.0} ), 
                            start_p( {5.0, 1.0} );
            Pe.push_back(start_e),
            Pp.push_back(start_p);
        }
        {
            multidim_state  start_e( {5.0, 3.0} ), 
                            start_p( {3.0, 1.0} );
            Pe.push_back(start_e),
            Pp.push_back(start_p);
        }

        speed_ratio = 0.5; // Pursuer 2x faster
    }
    

    MinMaxGame game(env, Pe, Pp, rprm_cnt_e, rprm_cnt_p, speed_ratio); // Ve/Vp
    game.print_paths(Pp.size() -1, Pe.size()-1);
    std::cout << "---------------" << endl;
    game.print_paths(Pp.size() -2, Pe.size()-2);

    game.plot_path(Pp.size() -1, Pe.size()-1, nmap);
    game.plot_path(Pp.size() -2, Pe.size()-2, nmap);

    player_poses_tofile(nmap, Pe, "evader");
    player_poses_tofile(nmap, Pp, "pursuer");
}
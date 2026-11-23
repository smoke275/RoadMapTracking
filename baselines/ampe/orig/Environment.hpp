#include "Geometry.hpp"
#include<bits/stdc++.h>
using namespace std;

#define xval  first
#define yval  second
#define pdd   pair<double, double>

const double eps = 1e-9;

struct Obstacle {
    double  minx, maxx,
            miny, maxy;
    vector<pdd> vertices;

    public:
        Obstacle(vector<pdd>);
        bool within(double, double);
        bool intersection(double, double, double, double);
};

class Environment {
    vector<Obstacle> obs;

    public:
        Environment(){};
        Environment(vector< vector<pdd> >);

        void add_obstacle(vector<pdd>);
        void addmulti_obstacles(vector< vector<pdd> >);
        bool withinObstacle(double, double);
        bool intersection(double, double, double, double);
        bool intersection(vector<double>& coords_p, vector<double>& coords_e);
};

/*
-------------------------------------------
OBSTACLE CLASS FUNCTIONS IMPLEMENTATION
-------------------------------------------
*/

Obstacle::Obstacle(vector<pdd> v) {
    vertices = v;

    minx = maxx = vertices[0].xval;
    miny = maxy = vertices[0].yval;

    for(auto v : vertices) {
        minx = min(minx, v.xval); maxx = max(maxx, v.xval);
        miny = min(miny, v.yval); maxy = max(maxy, v.yval);
    }
}

bool Obstacle::within(double x, double y) {
    if ( x < minx || x > maxx || y < miny || y > maxy ) return false;

    bool inside = false;
    for(int i=0, j=vertices.size()-1 ; i<(int)vertices.size(); j=i++) {
        if ( (vertices[i].yval > y) != ( vertices[j].yval > y) &&
             x < (vertices[j].xval - vertices[i].xval )*(y - vertices[i].yval) / (vertices[j].yval - vertices[i].yval) + vertices[i].xval) {
            inside = !inside;
        }
    }

    return inside;
}

bool Obstacle::intersection(double x1, double y1, double x2, double y2) {
    for(int i=0, j=vertices.size()-1 ; i<(int)vertices.size(); j=i++) {
        if( check_inter(pt(x1, y1), pt(x2, y2), pt(vertices[i].xval, vertices[i].yval), pt(vertices[j].xval, vertices[j].yval) ) )  return true;
    }
    return false;
}



/*
-------------------------------------------
ENVIRONMENT CLASS FUNCTIONS IMPLEMENTATION
-------------------------------------------
*/

Environment::Environment(vector< vector<pdd> > obs_list) {
    for(auto o : obs_list) {
        obs.push_back( Obstacle(o) );
    }
}

bool Environment::withinObstacle(double x, double y) {
    for(auto o : obs) {
        if(o.within(x, y)) return true;
    }
    return false;
}

bool Environment::intersection(double x1, double y1, double x2, double y2) {
    for(auto o : obs) {
        if(o.intersection(x1, y1, x2, y2)) return true;
    }
    return false;
}

bool Environment::intersection(vector<double>& coords_p, vector<double>& coords_e) {
    if( coords_p.size() != 2  || coords_e.size() != 2 ) {
        std::cout << "Environment intersection call invalid. " << endl;
        std::cout << "\tExpected dimension 2. Founded " << coords_p.size() << " and " << coords_e.size() << endl;
        return true;
    }

    return intersection(coords_p[0], coords_p[1], coords_e[0], coords_e[1]);
}

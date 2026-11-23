#include "Environment3D.hpp"

class EuclideanNdim
{
    public:
        vector<double> coords;
        EuclideanNdim(){};
        inline double Distance(EuclideanNdim &state);
};

class NodePursuer: public EuclideanNdim
{
    public:
        NodePursuer(){};
        NodePursuer(int dim);

        bool Collision(Environment3D &env) const;
        bool MoveCollision(Environment3D &env, NodePursuer &next_state) const;
};

class NodeEvader: public EuclideanNdim
{
    
};

/// EuclideanNdim functions implementation
inline double EuclideanNdim::Distance(EuclideanNdim& state)
{
    if(state.coords.size() != coords.size())
        throw std::invalid_argument("Not matching sizes in Distance function");
        return;

    double squared_dist = 0.0;
    for(int i=0; i < coords.size(); i++)
        squared_dist += (state.coords[i]-coords[i]) * (state.coords[i]-coords[i]);
    
    return sqrt(squared_dist);
}

/// NodePursuer functions implementation
NodePursuer::NodePursuer(int dim)
{
    coords.resize(dim);
}

bool NodePursuer::Collision(Environment3D &env) const
{

}

bool NodePursuer::MoveCollision(Environment3D &env) const
{

}
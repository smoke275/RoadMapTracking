#include <bits/stdc++.h>
using namespace std;

const double epsilon = 1e-9;

// Axis-Aligned Bounding Boxes Environment
class vec3
{
    double x, y, z;
    vec3()
    {
        x = y = z = 0;
    }
    vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    double at(int i)
    {
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        throw std::invalid_argument("Invalid access in vec3::at");
    }
};

struct BoundingBox
{
    vec3 pt1, pt2;
    vec3 lims[2]; // Box edges [min, max]

    BoundingBox(vec3 p1,  vec3 p2);
    bool PointInside(vec3 pt) const;
    double Distance(vec3 pt) const;
    bool SegmentIntersection(vec3 pt1, vec3 pt2) const;
};

class Environment3D
{
    vector<BoundingBox> obs;

    public:
        Environment3D(){};
        Environment3D(string obs_file);
        Environment3D(vector< pair<vec3, vec3> >);
        bool PointIntersection(vec3 pt);
        bool SphereIntersection(vec3 pt, double rad);
        bool SegmentIntersection(vec3 pt1, vec3 pt2);
};


/// BoundingBox functions implementation
void BoundingBox::BoundingBox(vec3 p1,  vec3 p2)
{
    pt1 = p1,
    pt2 = p2;

    lims[0].x = min(p1.x, p2.x), lims[1].x = max(p1.x, p2.x);

    lims[0].y = min(p1.y, p2.y), lims[1].y = max(p1.y, p2.y);

    lims[0].z = min(p1.z, p2.z), lims[1].z = max(p1.z, p2.z);
}

inline bool BoundingBox::PointInside(vec3 pt) const
{
    // Test if a point lies inside the box
    for(int i=0; i<3; i++){
        if(pt.at(i) < lims[0].at(i) || pt.at(i) > lims[1].at(i))
            return false;
    }
    return true;
}

inline double BoundingBox::SquaredDistance(vec3 pt) const
{
    double distance = 0.0;
    for(int i=0; i<3; i++)
    {
        if( pt.at(i) < lims[0].at(i) ) 
            distance += (pt.at(i) - lims[0].at(i)) * (pt.at(i) - lims[0].at(i));
        else
            distance += (pt.at(i) - lims[1].at(i)) * (pt.at(i) - lims[1].at(i));
    }
    return distance;
}

inline bool BoundingBox::SegmentIntersection(vec3 pt1, vec3 pt2) const
{
    for(int i=0; i<3; i++)
    {
        if( pt1.at(i) < lims[0].at(i) && pt2.at(i) < lims[0].at(i) ) 
            return false;
        if( pt1.at(i) > lims[0].at(i) && pt2.at(i) > lims[0].at(i) ) 
            return false;
    }
    return true;
}

/// Environment3D functions implementation
Environment3D(string obs_file)
{
    // Open and read file
}

Environment3D::Environment3D(vector< pair<vec3, vec3> >& obs_list)
{
    for(auto o : obs_list)
        obs.push_back(BoundingBox(o.first, o.second));
}

bool Environment3D::PointIntersection(vec3 pt)
{
    for(auto o : obs)
    {
        if( o.PointInside(pt) )
            return true;
    }
    return false;
}

bool Environment3D::SphereIntersection(vec3 pt, double rad)
{
    for(auto o : obs)
    {
        if( o.SquaredDistance(pt) < rad*rad )
            return true;
    }
}

bool Environment3D::SegmentIntersection(vec3 pt1, vec3 pt2)
{
    for(auto o : obs)
    {
        if( o.SegmentIntersection(pt1, pt2) )
            return true;
    }
    return false;
}
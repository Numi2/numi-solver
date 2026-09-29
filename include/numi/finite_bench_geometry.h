#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace numi::bench {

struct Vec3 {
    double x{}, y{}, z{};
    double& operator[](int axis) { return axis==0 ? x : axis==1 ? y : z; }
    double operator[](int axis) const { return axis==0 ? x : axis==1 ? y : z; }
};
inline Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline Vec3 operator*(Vec3 a,double b){return {a.x*b,a.y*b,a.z*b};}
inline Vec3 operator/(Vec3 a,double b){return a*(1/b);}
inline double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double length(Vec3 a){return std::hypot(a.x,a.y,a.z);}
inline bool finite(Vec3 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}

struct Box {
    Vec3 minimum{-.75,-.5,-.08};
    Vec3 maximum{.75,.5,0};
};
struct Sample { double gap; Vec3 normal; };
struct Hit {
    bool contact{};
    double time{1};
    Vec3 normal{};
    // 1 planar face, 2 rounded edge, 3 rounded vertex, 4 existing contact.
    unsigned feature{};
};
inline void validate(const Box& box,Vec3 point,double radius){
    if(!finite(box.minimum)||!finite(box.maximum)||!finite(point)||
       !std::isfinite(radius)||!(radius>0))
        throw std::invalid_argument("finite bench geometry must be finite with positive radius");
    for(int axis=0;axis<3;++axis)if(!(box.maximum[axis]>box.minimum[axis])||
        !std::isfinite(box.maximum[axis]-box.minimum[axis]))
        throw std::invalid_argument("finite bench box needs positive extents");
}

// Exact Euclidean signed distance to the box, offset by the sphere radius.
// The exterior normal includes its curved edge/vertex contact features.
inline Sample sample(const Box& box,Vec3 point,double radius){
    validate(box,point,radius);
    Vec3 nearest{};
    for(int axis=0;axis<3;++axis)nearest[axis]=std::clamp(point[axis],box.minimum[axis],box.maximum[axis]);
    Vec3 offset=point-nearest;double distance=length(offset);
    if(!std::isfinite(distance))throw std::invalid_argument("unbounded finite bench distance");
    if(distance>0)return {distance-radius,offset/distance};
    double nearestFace=std::numeric_limits<double>::infinity();Vec3 normal{};
    for(int axis=2;axis>=0;--axis)for(int sign:{1,-1}){
        double gap=sign>0 ? box.maximum[axis]-point[axis] : point[axis]-box.minimum[axis];
        if(gap<nearestFace){nearestFace=gap;normal={};normal[axis]=sign;}
    }
    return {-nearestFace-radius,normal};
}

// Solve the relative quadratic with stable roots, including a tangent root.
inline std::array<double,2> roots(double a,double b,double c){
    std::array<double,2> none{2,2};
    if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c))
        throw std::invalid_argument("unbounded finite bench quadratic");
    if(!(a>1e-30))return none;
    double discriminant=b*b-4*a*c;
    if(!std::isfinite(discriminant))
        throw std::invalid_argument("unbounded finite bench quadratic");
    if(discriminant<0)return none;
    double q=-.5*(b+std::copysign(std::sqrt(discriminant),b));
    if(q==0)return {-b/(2*a),-b/(2*a)};
    double first=q/a,second=c/q;
    return {std::min(first,second),std::max(first,second)};
}

// The sphere/box Minkowski boundary has six rectangles, twelve finite
// quarter-cylinder edges and eight octant spheres. Testing their exact
// domains avoids both tunneling and the expanded-AABB corner false positive.
inline Hit cast(const Box& box,Vec3 start,Vec3 end,double radius){
    validate(box,start,radius);validate(box,end,radius);
    Vec3 delta=end-start;Hit hit;
    if(!finite(delta))throw std::invalid_argument("unbounded finite bench displacement");
    Sample initial=sample(box,start,radius);
    if(initial.gap< -1e-9)throw std::invalid_argument("finite bench sweep starts inside the collider");
    if(initial.gap<=1e-10&&dot(delta,initial.normal)<-1e-16)
        return {true,0,initial.normal,4};
    auto accept=[&](double time,Vec3 normal,unsigned feature){
        if(!std::isfinite(time)||time< -1e-10||time>1+1e-10||!finite(normal)||dot(delta,normal)>=-1e-16)return;
        time=std::clamp(time,0.0,1.0);
        if(!hit.contact||time<hit.time){hit={true,time,normal,feature};}
    };
    for(int axis=0;axis<3;++axis)for(int sign:{-1,1}){
        if(delta[axis]*sign>=0)continue;
        double plane=(sign>0?box.maximum[axis]:box.minimum[axis])+sign*radius;
        double time=(plane-start[axis])/delta[axis];Vec3 point=start+delta*time;
        bool onFace=true;
        for(int other=0;other<3;++other)if(other!=axis)
            onFace&=point[other]>=box.minimum[other]-1e-10&&point[other]<=box.maximum[other]+1e-10;
        if(onFace){Vec3 normal{};normal[axis]=sign;accept(time,normal,1);}
    }
    for(int axis=0;axis<3;++axis){
        int first=(axis+1)%3,second=(axis+2)%3;
        for(int firstSign:{-1,1})for(int secondSign:{-1,1}){
            Vec3 edge{};edge[first]=firstSign>0?box.maximum[first]:box.minimum[first];
            edge[second]=secondSign>0?box.maximum[second]:box.minimum[second];
            Vec3 offset=start-edge;offset[axis]=0;Vec3 motion=delta;motion[axis]=0;
            for(double time:roots(dot(motion,motion),2*dot(offset,motion),dot(offset,offset)-radius*radius)){
                Vec3 point=start+delta*time,normal=point-edge;normal[axis]=0;
                if(point[axis]<box.minimum[axis]-1e-10||point[axis]>box.maximum[axis]+1e-10||
                   normal[first]*firstSign< -1e-10||normal[second]*secondSign< -1e-10)continue;
                double size=length(normal);if(size>0)accept(time,normal/size,2);
            }
        }
    }
    for(int xSign:{-1,1})for(int ySign:{-1,1})for(int zSign:{-1,1}){
        Vec3 vertex{xSign>0?box.maximum.x:box.minimum.x,
                    ySign>0?box.maximum.y:box.minimum.y,
                    zSign>0?box.maximum.z:box.minimum.z};
        Vec3 offset=start-vertex;
        for(double time:roots(dot(delta,delta),2*dot(offset,delta),dot(offset,offset)-radius*radius)){
            Vec3 normal=start+delta*time-vertex;
            if(normal.x*xSign< -1e-10||normal.y*ySign< -1e-10||normal.z*zSign< -1e-10)continue;
            double size=length(normal);if(size>0)accept(time,normal/size,3);
        }
    }
    return hit;
}

inline Hit castFloor(Vec3 start,Vec3 end,double radius,double height){
    if(!finite(start)||!finite(end)||!std::isfinite(radius)||radius<=0||!std::isfinite(height))
        throw std::invalid_argument("invalid room floor geometry");
    double support=height+radius;
    if(!std::isfinite(support)||!std::isfinite(end.z-start.z))
        throw std::invalid_argument("unbounded room floor geometry");
    if(start.z<support-1e-9)throw std::invalid_argument("floor sweep starts below support");
    if(end.z>=support||end.z>=start.z)return {};
    return {true,std::clamp((support-start.z)/(end.z-start.z),0.0,1.0),{0,0,1},1};
}

} // namespace numi::bench

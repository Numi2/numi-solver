#include "numi/elastic_yarn_contact.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ccd {
constexpr double tolerance=2e-6;
constexpr unsigned outerBudget=128;
using V=std::array<double,3>;
using P=std::array<V,5>;
struct Motion {std::array<NumiStaticVec3,5> start,end;float radius=.004f;};
struct Interval {double lo,hi;};
double down(double v){return std::nextafter(v,-INFINITY);}
double up(double v){return std::nextafter(v,INFINITY);}
Interval exact(double v){return {v,v};}
Interval add(Interval a,Interval b){return {down(a.lo+b.lo),up(a.hi+b.hi)};}
Interval sub(Interval a,Interval b){return {down(a.lo-b.hi),up(a.hi-b.lo)};}
Interval mul(Interval a,Interval b){const double v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};return {down(*std::min_element(v,v+4)),up(*std::max_element(v,v+4))};}
Interval square(Interval a){double lo=a.lo<=0&&a.hi>=0?0:std::min(a.lo*a.lo,a.hi*a.hi);return {std::max(0.,down(lo)),up(std::max(a.lo*a.lo,a.hi*a.hi))};}
using IV=std::array<Interval,3>;
IV minus(IV a,IV b){IV r;for(unsigned k=0;k<3;++k)r[k]=sub(a[k],b[k]);return r;}
IV cross(IV a,IV b){return {sub(mul(a[1],b[2]),mul(a[2],b[1])),sub(mul(a[2],b[0]),mul(a[0],b[2])),sub(mul(a[0],b[1]),mul(a[1],b[0]))};}
Interval norm(IV a){auto n=add(add(square(a[0]),square(a[1])),square(a[2]));return {down(std::sqrt(std::max(0.,n.lo))),up(std::sqrt(std::max(0.,n.hi)))};}
double component(NumiStaticVec3 p,unsigned k){return numiStaticComponent(p,k);}
IV position(const Motion& m,unsigned i,double t){IV p;for(unsigned k=0;k<3;++k)p[k]=add(exact(component(m.start[i],k)),mul(sub(exact(component(m.end[i],k)),exact(component(m.start[i],k))),exact(t)));return p;}
P points(const Motion& m,double t){P p;for(unsigned i=0;i<5;++i)for(unsigned k=0;k<3;++k)p[i][k]=std::fma(component(m.end[i],k)-component(m.start[i],k),t,component(m.start[i],k));return p;}
NumiElasticYarnWitness witness(const Motion& m,double t){auto p=points(m,t);NumiStaticVec3 f[5];for(unsigned i=0;i<5;++i)f[i]={float(p[i][0]),float(p[i][1]),float(p[i][2])};return numiElasticYarnClosestSegmentTriangle(f[0],f[1],f+2,m.radius);}
V minus(V a,V b){V r;for(unsigned k=0;k<3;++k)r[k]=a[k]-b[k];return r;}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}

// Any represented finite plane is admissible. The normal need not be exact.
// Every owning vertex follows a linear path, so projection extrema over an
// interval occur at its endpoints. All scalar bounds are outward rounded.
double planeLower(const Motion& m,double l,double h,V n){
    if(!std::isfinite(l)||!std::isfinite(h)||l<0||h>1||l>h||
       !numiStaticFinite(m.radius)||!(m.radius>0))return -INFINITY;
    for(double x:n)if(!std::isfinite(x))return -INFINITY;
    for(unsigned i=0;i<5;++i)if(!numiStaticFinite3(m.start[i])||!numiStaticFinite3(m.end[i]))return -INFINITY;
    IV ni;for(unsigned k=0;k<3;++k)ni[k]=exact(n[k]);
    const auto length=norm(ni);if(!(length.lo>0)||!std::isfinite(length.hi))return -INFINITY;
    double yarn=INFINITY,triangle=-INFINITY;
    for(double t:{l,h})for(unsigned i=0;i<5;++i){auto p=position(m,i,t);Interval d=exact(0);for(unsigned k=0;k<3;++k)d=add(d,mul(p[k],exact(n[k])));if(i<2)yarn=std::min(yarn,d.lo);else triangle=std::max(triangle,d.hi);}
    const double separation=down(down(yarn-triangle)-up(double(m.radius)*length.hi));
    return down(separation/(separation>=0?length.hi:length.lo));
}
double certifiedLower(const Motion& m,double l,double h,const NumiElasticYarnWitness& w){
    double best=-INFINITY;
    for(unsigned k=0;k<3;++k)for(double sign:{-1.,1.}){V n{};n[k]=sign;best=std::max(best,planeLower(m,l,h,n));}
    if(w.normalDefined)best=std::max(best,planeLower(m,l,h,{w.normal.x,w.normal.y,w.normal.z}));
    // Feature planes are directions only. Their correctness is never assumed:
    // each is checked by the same interval support certificate on ALL owners.
    // Avoid deriving a near-contact normal solely from an FP32 point delta,
    // whose tiny angle error is amplified by a much longer yarn axis.
    for(double time:{l,(l+h)*.5}){
        auto p=points(m,time);auto axis=minus(p[1],p[0]);
        const auto consider=[&](V n){best=std::max(best,planeLower(m,l,h,n));for(auto& x:n)x=-x;best=std::max(best,planeLower(m,l,h,n));};
        consider(cross(minus(p[3],p[2]),minus(p[4],p[2])));
        for(unsigned i=0;i<3;++i){const auto edge=minus(p[2+(i+1)%3],p[2+i]);consider(cross(axis,edge));
            const auto point=minus(p[2+i],p[0]);const double aa=dot(axis,axis);const double u=aa>0?std::clamp(dot(point,axis)/aa,0.,1.):0;
            V n;for(unsigned k=0;k<3;++k)n[k]=p[0][k]+axis[k]*u-p[2+i][k];consider(n);
        }
    }
    return best;
}
// A feasible material-point pair supplies an upper distance bound. A single
// downward-rounded complement proves REAL simplex membership; iterating over
// subnormal/near-vertex predecessors can otherwise take unbounded time.
double feasibleTriangleThird(double second,double third){
    return std::min(third,std::max(0.,down(1-second)));
}
double upperGap(const Motion& m,double t,const NumiElasticYarnWitness& w){
    if(!w.valid||!std::isfinite(t)||t<0||t>1||
       !numiStaticFinite(m.radius)||!(m.radius>0)||
       !numiStaticFinite(w.yarnParameter)||!numiStaticFinite(w.triangleWeights[1])||
       !numiStaticFinite(w.triangleWeights[2]))return INFINITY;
    if(w.yarnParameter<0||w.yarnParameter>1||w.triangleWeights[1]<0||
       w.triangleWeights[1]>1||w.triangleWeights[2]<0||w.triangleWeights[2]>1)return INFINITY;
    for(unsigned i=0;i<5;++i)if(!numiStaticFinite3(m.start[i])||!numiStaticFinite3(m.end[i]))return INFINITY;
    double u=std::clamp(double(w.yarnParameter),0.,1.);
    double v=std::clamp(double(w.triangleWeights[1]),0.,1.);
    double z=feasibleTriangleThird(v,std::clamp(double(w.triangleWeights[2]),0.,1.));
    auto a=position(m,0,t),b=position(m,1,t),c=position(m,2,t),d=position(m,3,t),e=position(m,4,t);
    IV delta;for(unsigned k=0;k<3;++k){auto y=add(a[k],mul(sub(b[k],a[k]),exact(u)));auto x=add(c[k],add(mul(sub(d[k],c[k]),exact(v)),mul(sub(e[k],c[k]),exact(z))));delta[k]=sub(y,x);}
    return up(norm(delta).hi-double(m.radius));
}
bool faceRange(const Motion& m,double l,double h){
    IV e[2];for(unsigned j=0;j<2;++j){auto a=minus(position(m,j+3,l),position(m,2,l)),b=minus(position(m,j+3,h),position(m,2,h));for(unsigned k=0;k<3;++k)e[j][k]={std::min(a[k].lo,b[k].lo),std::max(a[k].hi,b[k].hi)};}
    auto n=cross(e[0],e[1]);double edge=0;for(unsigned j=0;j<2;++j){double value=0;for(unsigned k=0;k<3;++k)value=up(value+square(e[j][k]).hi);edge=std::max(edge,value);}
    const double threshold=up(up(std::sqrt(128.)*std::numeric_limits<float>::epsilon())*edge);
    for(auto a:n)if(a.lo>threshold||a.hi<-threshold)return true;
    return false;
}
enum class Status {clear,support,bracket,initialOverlap,degenerate,unresolved,invalid};
const char* name(Status s){switch(s){case Status::clear:return "clear";case Status::support:return "support";case Status::bracket:return "bracket";case Status::initialOverlap:return "initial_overlap";case Status::degenerate:return "degenerate";case Status::unresolved:return "unresolved";default:return "invalid";}}
struct Result {Status status{Status::unresolved};double lowerTime{},upperTime{1},gapUpper{INFINITY},motionBound{};unsigned outer{};};
Result sweep(const Motion& m,unsigned budget=outerBudget){
    Result out;
    if(!numiStaticFinite(m.radius)||!(m.radius>0)){out.status=Status::invalid;return out;}
    for(unsigned i=0;i<5;++i)if(!numiStaticFinite3(m.start[i])||!numiStaticFinite3(m.end[i])){out.status=Status::invalid;return out;}
    double travel[2]={};for(unsigned i=0;i<5;++i){IV d;for(unsigned k=0;k<3;++k)d[k]=sub(exact(component(m.end[i],k)),exact(component(m.start[i],k)));travel[i>=2]=std::max(travel[i>=2],norm(d).hi);}
    out.motionBound=up(travel[0]+travel[1]);
    const auto admit=[&](auto&& self,double l,double h)->bool {
        if(out.outer>=budget){out.status=Status::unresolved;return false;}++out.outer;
        if(faceRange(m,l,h))return true;
        const double mid=(l+h)*.5;
        if(!witness(m,l).valid||!witness(m,mid).valid||!witness(m,h).valid){out.status=Status::degenerate;return false;}
        if(!(mid>l&&mid<h)){out.status=Status::unresolved;return false;}
        return self(self,l,mid)&&self(self,mid,h);
    };
    if(!admit(admit,0,1))return out;
    auto first=witness(m,0);if(!first.valid){out.status=Status::degenerate;return out;}
    const double startUpper=upperGap(m,0,first);
    if(startUpper<-tolerance){out.status=Status::initialOverlap;out.gapUpper=startUpper;return out;}
    // Inclusive initial support has an interval separation proof. Its tiny
    // arithmetic uncertainty is bounded independently and below 2 micrometres.
    const double scale=std::max(1.,double(numiStaticLength(m.start[0]))+double(numiStaticLength(m.start[2])));
    const double supportError=std::min(tolerance/32,64*std::numeric_limits<double>::epsilon()*scale);
    if(startUpper<=tolerance&&first.normalDefined&&certifiedLower(m,0,1,first)>=-supportError&&
       certifiedLower(m,0,0,first)>=-tolerance){out.status=Status::support;out.upperTime=0;out.gapUpper=startUpper;return out;}
    if(startUpper<=0||!first.normalDefined){out.status=Status::initialOverlap;out.gapUpper=startUpper;return out;}
    // A feasible-pair upper distance never proves global initial separation.
    // Search starts only from a strict interval support-plane certificate;
    // uncertain initial touch without the full support proof stays unresolved.
    if(!(certifiedLower(m,0,0,first)>0)){out.status=Status::unresolved;out.gapUpper=startUpper;return out;}
    const auto search=[&](auto&& self,double l,double h)->bool {
        if(out.outer>=budget){out.status=Status::unresolved;return true;}++out.outer;
        const double mid=(l+h)*.5;
        auto lo=witness(m,l),mi=witness(m,mid),hi=witness(m,h);
        if(!lo.valid||!mi.valid||!hi.valid){out.status=Status::degenerate;return true;}
        if(certifiedLower(m,l,h,mi)>0||certifiedLower(m,l,h,lo)>0){out.lowerTime=h;return false;}
        double contactUpper=INFINITY;
        if(upperGap(m,mid,mi)<=0)contactUpper=mid;
        if(upperGap(m,h,hi)<=0)contactUpper=std::min(contactUpper,h);
        const double gap=upperGap(m,l,lo);
        if(contactUpper<=h&&up((h-l)*out.motionBound)<=tolerance&&gap<=tolerance&&lo.normalDefined&&
           certifiedLower(m,l,l,lo)>=-tolerance){out.status=Status::bracket;out.lowerTime=l;out.upperTime=contactUpper;out.gapUpper=gap;return true;}
        if(!(mid>l&&mid<h)){out.status=Status::unresolved;return true;}
        if(self(self,l,mid))return true;
        return self(self,mid,h);
    };
    if(!search(search,0,1)){out.status=Status::clear;out.lowerTime=out.upperTime=1;}
    return out;
}

// Independent FP64 active-set QP over box x triangle simplex. It does not
// use the production Voronoi/edge/plane-intersection implementation.
double oracle(const Motion& m,double time){
    auto p=points(m,time);const auto r=minus(p[0],p[2]);const V c[3]={minus(p[1],p[0]),minus(p[2],p[3]),minus(p[2],p[4])};
    const double a[5][3]={{1,0,0},{1,0,0},{0,1,0},{0,0,1},{0,1,1}};const double rhs[5]={0,1,0,0,1};
    double best=INFINITY;
    for(unsigned mask=0;mask<32;++mask){const unsigned k=std::popcount(mask);if(k>3)continue;const unsigned n=3+k;double matrix[6][7]{};unsigned active[3]{},count=0;for(unsigned j=0;j<5;++j)if(mask&(1u<<j))active[count++]=j;
        for(unsigned i=0;i<3;++i){for(unsigned j=0;j<3;++j)matrix[i][j]=dot(c[i],c[j]);matrix[i][n]=-dot(c[i],r);}
        for(unsigned j=0;j<k;++j){for(unsigned i=0;i<3;++i)matrix[i][3+j]=matrix[3+j][i]=a[active[j]][i];matrix[3+j][n]=rhs[active[j]];}
        bool valid=true;for(unsigned column=0;column<n;++column){unsigned pivot=column;for(unsigned row=column+1;row<n;++row)if(std::abs(matrix[row][column])>std::abs(matrix[pivot][column]))pivot=row;if(std::abs(matrix[pivot][column])<1e-14){valid=false;break;}for(unsigned j=0;j<=n;++j)std::swap(matrix[column][j],matrix[pivot][j]);const double divisor=matrix[column][column];for(unsigned j=column;j<=n;++j)matrix[column][j]/=divisor;for(unsigned row=0;row<n;++row)if(row!=column){const double factor=matrix[row][column];for(unsigned j=column;j<=n;++j)matrix[row][j]-=factor*matrix[column][j];}}
        if(!valid)continue;const double u=matrix[0][n],v=matrix[1][n],w=matrix[2][n];if(u<-1e-10||u>1+1e-10||v<-1e-10||w<-1e-10||v+w>1+1e-10)continue;V delta=r;for(unsigned j=0;j<3;++j)for(unsigned axis=0;axis<3;++axis)delta[axis]+=c[j][axis]*matrix[j][n];best=std::min(best,std::sqrt(std::max(0.,dot(delta,delta))));
    }
    return best-double(m.radius);
}
double oracleTime(const Motion& m){double previous=0;for(unsigned i=1;i<=128;++i){double t=double(i)/128;if(oracle(m,t)<-1e-12){double l=previous,h=t;for(unsigned j=0;j<45;++j){double mid=(l+h)*.5;if(oracle(m,mid)<=0)h=mid;else l=mid;}return h;}previous=t;}return INFINITY;}
} // namespace ccd

int main(int argc,char** argv){try {
    using namespace ccd;
    unsigned passed=0,retained=0,analyticControls=0;double maxSpace=0,maxOracleError=0;
    const std::array<NumiStaticVec3,5> base={NumiStaticVec3{-.8f,0,.1f},{.8f,0,.1f},{-.3f,-.3f,0},{.3f,-.3f,0},{0,.3f,0}};
    struct Case {std::string name;Motion motion;Status expected;bool temporal;};std::vector<Case> cases;
    Motion yarn{base,base};yarn.end[0].z=yarn.end[1].z=-.1f;cases.push_back({"interior_crossing_endpoints_clear",yarn,Status::bracket,true});
    Motion triangle{base,base};triangle.start[0].z=triangle.start[1].z=0;triangle.end[0].z=triangle.end[1].z=0;for(unsigned i=2;i<5;++i){triangle.start[i].z=.1f;triangle.end[i].z=-.1f;}cases.push_back({"static_yarn_moving_triangle",triangle,Status::bracket,true});
    auto deform=triangle;deform.end[2].z=-.12f;deform.end[3].z=-.08f;deform.end[4].z=-.15f;cases.push_back({"static_yarn_deforming_triangle",deform,Status::bracket,true});
    auto all=deform;all.end[0].z=.03f;all.end[1].z=.02f;all.end[0].x-=.02f;all.end[1].x+=.05f;cases.push_back({"all_five_owners_move_independently",all,Status::bracket,true});
    auto reverse=yarn;for(unsigned i=0;i<5;++i)std::swap(reverse.start[i],reverse.end[i]);cases.push_back({"reverse_motion",reverse,Status::bracket,true});
    Motion edge=yarn;edge.start[0]={-.2f,-.002f,.05f};edge.start[1]={.2f,-.002f,.05f};edge.end[0]={-.2f,-.002f,-.05f};edge.end[1]={.2f,-.002f,-.05f};edge.start[2]=edge.end[2]={0,0,0};edge.start[3]=edge.end[3]={1,0,0};edge.start[4]=edge.end[4]={0,1,0};cases.push_back({"triangle_edge_event",edge,Status::bracket,true});
    auto vertex=edge;vertex.start[0].x=vertex.end[0].x=0;vertex.start[1].x=vertex.end[1].x=-.1f;vertex.start[0].y=vertex.end[0].y=vertex.start[1].y=vertex.end[1].y=-.003f;cases.push_back({"triangle_vertex_event",vertex,Status::bracket,true});
    auto adjacent=edge;adjacent.start[3]=adjacent.end[3]={0,-1,0};adjacent.start[4]=adjacent.end[4]={1,0,0};cases.push_back({"adjacent_triangle_event",adjacent,Status::bracket,true});
    auto tangent=yarn;for(unsigned i=0;i<2;++i){tangent.start[i].z=tangent.end[i].z=tangent.radius;tangent.end[i].x+=.02f;}cases.push_back({"supported_tangent",tangent,Status::support,false});
    auto outward=tangent;outward.end[0].z=outward.end[1].z=.1f;cases.push_back({"supported_outward",outward,Status::support,false});
    auto miss=yarn;miss.end[0].z=miss.end[1].z=.05f;cases.push_back({"certified_clear",miss,Status::clear,false});
    auto grazing=yarn;for(unsigned i=0;i<2;++i){grazing.start[i].z=grazing.end[i].z=grazing.radius;grazing.start[i].y=.6f;grazing.end[i].y=-.6f;}cases.push_back({"later_grazing_without_penetration",grazing,Status::unresolved,false});
    auto collapsed=yarn;collapsed.end[2]=collapsed.start[3];collapsed.end[3]=collapsed.start[2];cases.push_back({"interior_triangle_collapse",collapsed,Status::degenerate,false});
    auto overlap=yarn;overlap.start[0].z=overlap.start[1].z=0;cases.push_back({"initial_axis_crossing",overlap,Status::initialOverlap,false});
    const unsigned original=cases.size();for(unsigned i=0;i<original;++i){auto c=cases[i];c.name+="_reversed_winding";std::swap(c.motion.start[3],c.motion.start[4]);std::swap(c.motion.end[3],c.motion.end[4]);cases.push_back(c);}
    if(argc>1&&std::string(argv[1])=="--grazing-frontier"){
        auto result=sweep(grazing);std::cout<<std::setprecision(17)<<"grazing_frontier status="<<name(result.status)<<" certified_clear_prefix_t="<<result.lowerTime<<" outer="<<result.outer<<" acceptance=0\n";return result.status==Status::unresolved?1:2;
    }
    if(argc>1&&std::string(argv[1])=="--endpoint-only-negative-control"){
        const bool endpointOnly=oracle(yarn,0)>tolerance&&oracle(yarn,1)>tolerance;auto actual=sweep(yarn);
        std::cout<<"endpoint_only_negative falsely_clear="<<endpointOnly<<" actual_status="<<name(actual.status)<<" interior_gap_m="<<oracle(yarn,.5)<<'\n';return endpointOnly&&actual.status==Status::bracket?1:2;
    }
    for(const auto& c:cases){auto result=sweep(c.motion),replay=sweep(c.motion);if(result.status!=c.expected)throw std::runtime_error(c.name+" unexpected "+name(result.status));if(result.status!=replay.status||result.lowerTime!=replay.lowerTime||result.upperTime!=replay.upperTime||result.gapUpper!=replay.gapUpper||result.outer!=replay.outer)throw std::runtime_error("replay mismatch");double time=c.temporal?oracleTime(c.motion):INFINITY;if(c.temporal){if(!std::isfinite(time)||result.lowerTime>time+1e-12||result.upperTime<time-1e-12||result.gapUpper>tolerance||up((result.upperTime-result.lowerTime)*result.motionBound)>tolerance)throw std::runtime_error(c.name+" temporal certificate mismatch");maxSpace=std::max(maxSpace,(time-result.lowerTime)*result.motionBound);double analytic=INFINITY;
        if(c.name.starts_with("interior_crossing")||c.name.starts_with("static_yarn_moving_triangle")||c.name.starts_with("reverse_motion")||c.name.starts_with("adjacent_triangle")){const double start=double(c.motion.start[0].z)-c.motion.start[2].z,end=double(c.motion.end[0].z)-c.motion.end[2].z;analytic=(std::abs(start)-c.motion.radius)/std::abs(end-start);}
        if(c.name.starts_with("triangle_edge")||c.name.starts_with("triangle_vertex")){const double y=double(c.motion.start[0].y)-c.motion.start[2].y,r=c.motion.radius;const double contactZ=std::sqrt(r*r-y*y);analytic=(double(c.motion.start[0].z)-contactZ)/(double(c.motion.start[0].z)-c.motion.end[0].z);}
        if(std::isfinite(analytic)){const double error=std::abs(time-analytic);maxOracleError=std::max(maxOracleError,error);if(error>1e-10)throw std::runtime_error("independent QP temporal oracle disagrees with analytic contact time");++analyticControls;}
        }if(result.status==Status::degenerate||result.status==Status::initialOverlap||result.status==Status::unresolved)++retained;else ++passed;std::cout<<std::setprecision(17)<<"case="<<c.name<<" status="<<name(result.status)<<" lower_t="<<result.lowerTime<<" upper_t="<<result.upperTime<<" oracle_t="<<time<<" gap_upper_m="<<result.gapUpper<<" outer="<<result.outer<<'\n';}
    const auto bounded=sweep(yarn,1);if(bounded.status!=Status::unresolved)throw std::runtime_error("budget exhaustion admitted endpoint-clear crossing");++retained;
    auto malformed=yarn;malformed.radius=0;if(sweep(malformed).status!=Status::invalid)throw std::runtime_error("nonpositive radius admitted");++retained;
    if(!(oracle(yarn,0)>tolerance&&oracle(yarn,1)>tolerance&&oracle(yarn,.5)<-tolerance&&sweep(yarn).status==Status::bracket))throw std::runtime_error("endpoint-only crossing negative control failed");
    std::cout<<"endpoint_only_negative_control falsely_clear=1 actual_five_owner_contact=1\n";
    std::cout<<"passed_contact_clear_support_cases="<<passed<<" retained_failure_controls="<<retained<<" maximum_conservative_spatial_earliness_m="<<maxSpace<<" analytic_temporal_controls="<<analyticControls<<" maximum_oracle_analytic_time_error="<<maxOracleError<<" tolerance_m="<<tolerance<<" outer_budget="<<outerBudget<<" exact_replays=2 no_gpu=1\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}

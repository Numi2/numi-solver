#define main fixturePrototypeMain
#include "prototype.cpp"
#undef main
#include <chrono>
int main(){
    using namespace ccd;
    unsigned cases=0,failures=0;
    const auto check=[&](bool pass){++cases;failures+=!pass;};
    std::cout<<std::setprecision(17);
    const std::array<NumiStaticVec3,5> vertices={NumiStaticVec3{1,1e-20f,.004f},
        {1,1e-20f,.004f},{0,0,0},{1,0,0},{0,1,0}};
    Motion nearVertex{vertices,vertices};
    const auto begin=std::chrono::steady_clock::now();
    auto w=witness(nearVertex,0);
    double gap=upperGap(nearVertex,0,w);
    auto result=sweep(nearVertex);
    check(w.valid&&w.normalDefined&&w.triangleWeights[1]==1&&w.triangleWeights[2]>0);
    check(std::isfinite(gap)&&gap<=tolerance&&result.status==Status::support&&result.outer<=128);
    std::cout<<"near_vertex_actual_production_witness_v="<<w.triangleWeights[1]
        <<" z="<<w.triangleWeights[2]<<" bounded_gap_upper="<<gap
        <<" sweep_status="<<name(result.status)<<" seconds="
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
    const auto negativeInfinity=[](double x){return std::isinf(x)&&x<0;};
    const auto positiveInfinity=[](double x){return std::isinf(x)&&x>0;};
    for(unsigned k=0;k<3;++k)for(double bad:{NAN,INFINITY,-INFINITY}){
        V n{0,0,1};n[k]=bad;check(negativeInfinity(planeLower(nearVertex,0,1,n)));
    }
    for(V n:std::array<V,3>{{{0,0,0},{1e300,0,0},{1e-300,0,0}}})
        check(negativeInfinity(planeLower(nearVertex,0,1,n)));
    for(auto interval:std::array<std::array<double,2>,7>{{{NAN,1},{0,NAN},{INFINITY,1},
        {0,-INFINITY},{.75,.25},{-.01,1},{0,1.01}}})
        check(negativeInfinity(planeLower(nearVertex,interval[0],interval[1],{0,0,1})));
    for(unsigned i=0;i<5;++i){
        auto m=nearVertex;m.start[i].x=NAN;
        check(negativeInfinity(planeLower(m,0,1,{0,0,1}))&&positiveInfinity(upperGap(m,0,w)));
        m=nearVertex;m.end[i].y=INFINITY;
        check(negativeInfinity(planeLower(m,0,1,{0,0,1}))&&positiveInfinity(upperGap(m,0,w)));
    }
    for(float radius:{0.f,-1.f,NAN,INFINITY}){
        auto m=nearVertex;m.radius=radius;
        check(negativeInfinity(planeLower(m,0,1,{0,0,1}))&&positiveInfinity(upperGap(m,0,w)));
    }
    for(unsigned field=0;field<3;++field)for(float bad:{NAN,INFINITY,-INFINITY}){
        auto broken=w;
        if(field==0)broken.yarnParameter=bad;else broken.triangleWeights[field]=bad;
        check(positiveInfinity(upperGap(nearVertex,0,broken)));
    }
    for(unsigned field=0;field<3;++field)for(float bad:{-.01f,1.01f}){
        auto broken=w;
        if(field==0)broken.yarnParameter=bad;else broken.triangleWeights[field]=bad;
        check(positiveInfinity(upperGap(nearVertex,0,broken)));
    }
    for(double time:std::array<double,5>{NAN,INFINITY,-INFINITY,-.01,1.01})
        check(positiveInfinity(upperGap(nearVertex,time,w)));
    auto incoming=nearVertex;incoming.end[0].z=incoming.end[1].z=0;
    auto incomingResult=sweep(incoming);
    check(incomingResult.status==Status::unresolved||incomingResult.status==Status::initialOverlap);
    std::cout<<"initial_uncertified_incoming_touch="<<name(incomingResult.status)
        <<" acceptance=0\n";
    for(double value:std::array<double,9>{0,1,std::numeric_limits<float>::denorm_min(),
        double(1e-20f),double(1e-8f),.25,.5,.75,double(std::nextafter(1.f,0.f))})
        for(double third:std::array<double,9>{0,1,std::numeric_limits<float>::denorm_min(),
            double(1e-20f),double(1e-8f),.25,.5,.75,double(std::nextafter(1.f,0.f))})
            std::cout<<"simplex v="<<value<<" z="<<third<<" result="<<feasibleTriangleThird(value,third)<<'\n';
    std::cout<<"bounded_helper_controls="<<cases<<" failures="<<failures
        <<" exact_binary_rational_simplex_controls_pending=81 no_gpu=1\n";
    return failures?1:0;
}

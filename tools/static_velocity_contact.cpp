#include "numi/static_velocity_contact.h"
#include "numi/static_segment_geometry.h"
#include "numi/finite_bench_geometry.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
using V = NumiStaticVec3;
using D = numi::bench::Vec3;
struct Case { std::string name; NumiStaticVelocityBody first, second; V ga, gb; float cap; bool normal; };
D d(V v) { return {v.x,v.y,v.z}; }
V add(V a,V b){return numiStaticAdd(a,b);} V scale(V a,float s){return numiStaticScale(a,s);}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
numi::bench::Sample independentSample(const NumiStaticVelocityBody& b) {
    const numi::bench::Box box{{-.75,-.5,double(-.08f)},{.75,.5,0}};
    auto s=numi::bench::sample(box,d(b.position),b.radius);
    const double floor=b.position.z-double(-.75f)-double(b.radius);
    if(floor<s.gap)s={floor,{0,0,1}};
    return s;
}
struct OracleBody { D velocity; double support; };
OracleBody oracleBody(const NumiStaticVelocityBody& b,V gradient,double impulse) {
    OracleBody out{d(b.velocity)+d(gradient)*(double(b.inverseMass)*impulse),0};
    if(b.inverseMass==0)return {d(b.velocity),0};
    const auto s=independentSample(b);
    if(s.gap<=numiStaticTolerance(b.position,b.position)){
        const double incoming=numi::bench::dot(out.velocity,s.normal);
        if(incoming<0){out.velocity=out.velocity-s.normal*incoming;out.support=-incoming/b.inverseMass;}
    }
    return out;
}
struct Oracle { OracleBody a,b; double impulse,derivative; };
Oracle oracle(const Case& c){
    const auto evaluate=[&](double impulse){auto a=oracleBody(c.first,c.ga,impulse),b=oracleBody(c.second,c.gb,impulse);return Oracle{a,b,impulse,numi::bench::dot(d(c.ga),a.velocity)+numi::bench::dot(d(c.gb),b.velocity)};};
    auto out=evaluate(0);if(out.derivative>=0||(!c.normal&&c.cap==0))return out;
    double lo=0,hi=-out.derivative/(double(c.first.inverseMass)*numi::bench::dot(d(c.ga),d(c.ga))+double(c.second.inverseMass)*numi::bench::dot(d(c.gb),d(c.gb)));
    if(!c.normal)hi=std::min(hi,double(c.cap));
    for(unsigned i=0;i<256;++i){out=evaluate(hi);if(out.derivative>=0)break;if(!c.normal&&hi==c.cap)return out;lo=hi;hi*=2;if(!c.normal)hi=std::min(hi,double(c.cap));}
    for(unsigned i=0;i<100;++i){double mid=lo+(hi-lo)*.5;auto v=evaluate(mid);if(v.derivative>=0)hi=mid;else lo=mid;}
    return evaluate(hi);
}
long double energyDelta(const NumiStaticVelocityBody& b,V output){
    if(b.inverseMass==0)return 0;
    long double total=0;
    for(unsigned i=0;i<3;++i){long double a=numiStaticComponent(b.velocity,i),z=numiStaticComponent(output,i);total+=(z-a)*(z+a)/(2*static_cast<long double>(b.inverseMass));}
    return total;
}
std::array<std::uint32_t,12> words(const NumiStaticVelocityPairResult& r){return {
    std::bit_cast<std::uint32_t>(r.firstVelocity.x),std::bit_cast<std::uint32_t>(r.firstVelocity.y),std::bit_cast<std::uint32_t>(r.firstVelocity.z),
    std::bit_cast<std::uint32_t>(r.secondVelocity.x),std::bit_cast<std::uint32_t>(r.secondVelocity.y),std::bit_cast<std::uint32_t>(r.secondVelocity.z),
    std::bit_cast<std::uint32_t>(r.impulse),std::bit_cast<std::uint32_t>(r.firstSupportImpulse),std::bit_cast<std::uint32_t>(r.secondSupportImpulse),r.iterations,r.status,unsigned(r.valid)};}
V oldResponse(const NumiStaticVelocityBody& body,V gradient){
    V response=scale(gradient,body.inverseMass);auto s=numiStaticSceneSample(body.position,body.radius);
    if(s.gap<=numiStaticTolerance(body.position,body.position)&&numiStaticDot(response,s.normal)<0)
        response=scale(cross(s.normal,cross(response,s.normal)),1/numiStaticDot(s.normal,s.normal));
    return response;
}
std::array<V,2> oldPair(const Case& c){
    auto ra=oldResponse(c.first,c.ga),rb=oldResponse(c.second,c.gb);
    const float derivative=numiStaticDot(c.ga,c.first.velocity)+numiStaticDot(c.gb,c.second.velocity);
    const float slope=numiStaticDot(c.ga,ra)+numiStaticDot(c.gb,rb);
    float impulse=slope>0?std::max(0.f,-derivative/slope):0;if(!c.normal)impulse=std::min(impulse,c.cap);
    return {add(c.first.velocity,scale(ra,impulse)),add(c.second.velocity,scale(rb,impulse))};
}
}

int main(){
    std::vector<Case> cases;
    const auto freeBody=[](float inverse,V velocity,float x=2.f){return NumiStaticVelocityBody{{x,0,1},velocity,inverse,.004f};};
    const auto floorBody=[](float inverse,V velocity,float clearance=2e-7f){return NumiStaticVelocityBody{{2,0,-.75f+.004f+clearance},velocity,inverse,.004f};};
    cases.push_back({"free_equal",freeBody(2,{1,0,0}),freeBody(2,{-1,0,0},3),{-1,0,0},{1,0,0},0,true});
    cases.push_back({"free_unequal",freeBody(2,{1,0,0}),freeBody(1,{-1,0,0},3),{-1,0,0},{1,0,0},0,true});
    cases.push_back({"inactive",freeBody(2,{-1,0,0}),freeBody(1,{1,0,0},3),{-1,0,0},{1,0,0},0,true});
    cases.push_back({"zero",freeBody(2,{}),freeBody(1,{},3),{-1,0,0},{1,0,0},0,true});
    cases.push_back({"pinned_first",freeBody(0,{}),freeBody(2,{-1,0,0},3),{-1,0,0},{1,0,0},0,true});
    cases.push_back({"pinned_both",freeBody(0,{}),freeBody(0,{},3),{-1,0,0},{1,0,0},0,true});
    cases.push_back({"friction_limited",freeBody(20000,{0,2,0}),freeBody(5,{0,1,0},3),{0,-.25f,0},{0,-.75f,0},.000001f,false});
    cases.push_back({"friction_zero_cap",freeBody(2,{0,2,0}),freeBody(5,{0,1,0},3),{0,-.25f,0},{0,-.75f,0},0,false});
    cases.push_back({"friction_stop",freeBody(20,{0,2,0}),freeBody(5,{0,1,0},3),{0,-.25f,0},{0,-.75f,0},1,false});
    cases.push_back({"floor_slack",floorBody(20000,{0,0,100}),freeBody(20000,{},3),{0,0,-.25f},{0,0,-.75f},0,true});
    cases.push_back({"floor_slack_exhausted",floorBody(20000,{0,0,2}),freeBody(10,{0,0,10},3),{0,0,-.5f},{0,0,-.5f},0,true});
    cases.push_back({"floor_initial_projection",floorBody(20,{2,0,-2}),floorBody(5,{1,0,-3}),{-.25f,0,0},{-.75f,0,0},1,false});
    cases.push_back({"floor_tangent",floorBody(20,{2,0,0}),floorBody(5,{1,0,0}),{-.25f,0,0},{-.75f,0,0},1,false});
    cases.push_back({"zero_gradient_support",floorBody(20,{0,0,-2}),freeBody(5,{},3),{},{},0,true});
    const float radius=.004f,root=std::sqrt(.5f),inverse=20000;
    const V impact{.75f+radius*root,0,-.08f-radius*root};
    NumiStaticVelocityBody normalA{{impact.x+(-.75f+radius+2e-7f-impact.z),0,-.75f+radius+2e-7f},{0,0,100},inverse,radius};
    NumiStaticVelocityBody normalB{{1,0,impact.z+(1-impact.x)},{},inverse,radius};
    auto normalSample=numiStaticSegmentSceneSample(normalA.position,normalB.position,radius);
    cases.push_back({"normal_energy_regression",normalA,normalB,scale(normalSample.normal,1-normalSample.parameter),scale(normalSample.normal,normalSample.parameter),0,true});
    NumiStaticVelocityBody frictionA{{.75f+radius,0,-.75f+radius+2e-7f},{0,0,100},inverse,radius};
    NumiStaticVelocityBody frictionB{{.75f+radius,0,.1f},{0,0,-1},inverse,radius};
    auto frictionSample=numiStaticSegmentSceneSample(frictionA.position,frictionB.position,radius);
    cases.push_back({"friction_energy_regression",frictionA,frictionB,{0,0,-(1-frictionSample.parameter)},{0,0,-frictionSample.parameter},.0045f,false});
    bool pass=normalSample.valid&&frictionSample.valid;
    double maximumVelocityError=0,maximumImpulseError=0,maximumSupportError=0,maximumMomentumError=0;
    long double maximumEnergyGain=0;
    std::vector<std::array<std::uint32_t,12>> firstReplay;
    for(unsigned replay=0;replay<2;++replay){
        for(const auto& c:cases){
            auto r=numiStaticResolveVelocityPair(c.first,c.second,c.ga,c.gb,c.cap,c.normal);
            const auto reference=oracle(c);
            const double velocityError=std::max(numi::bench::length(d(r.firstVelocity)-reference.a.velocity),numi::bench::length(d(r.secondVelocity)-reference.b.velocity));
            const double impulseError=std::abs(r.impulse-reference.impulse);
            const double supportError=std::max(std::abs(r.firstSupportImpulse-reference.a.support),std::abs(r.secondSupportImpulse-reference.b.support));
            const long double delta=energyDelta(c.first,r.firstVelocity)+energyDelta(c.second,r.secondVelocity);
            double momentumError=0;
            for(unsigned endpoint=0;endpoint<2;++endpoint){
                const auto& body=endpoint==0?c.first:c.second;
                if(body.inverseMass==0)continue;
                const auto output=endpoint==0?r.firstVelocity:r.secondVelocity;
                const auto gradient=endpoint==0?c.ga:c.gb;
                const auto support=endpoint==0?r.firstSupportImpulse:r.secondSupportImpulse;
                const auto sample=independentSample(body);
                const D residual=(d(output)-d(body.velocity))/body.inverseMass-d(gradient)*r.impulse-sample.normal*support;
                momentumError=std::max(momentumError,numi::bench::length(residual));
            }
            maximumMomentumError=std::max(maximumMomentumError,momentumError);
            maximumVelocityError=std::max(maximumVelocityError,velocityError);maximumImpulseError=std::max(maximumImpulseError,impulseError);maximumSupportError=std::max(maximumSupportError,supportError);maximumEnergyGain=std::max(maximumEnergyGain,delta);
            bool row=r.valid&&velocityError<2e-4&&impulseError<2e-6&&supportError<2e-6&&momentumError<2e-6&&delta<=0&&
                (c.normal||r.impulse<=c.cap);
            if(replay==0)firstReplay.push_back(words(r));else row&=words(r)==firstReplay[&c-cases.data()];
            pass&=row;
            if(replay==0)std::cout<<std::setprecision(17)<<"case="<<c.name<<" valid="<<r.valid<<" status="<<r.status<<" impulse_Ns="<<r.impulse<<" velocity_error_m_s="<<velocityError<<" support_error_Ns="<<supportError<<" kinetic_delta_J="<<double(delta)<<" passed="<<row<<'\n';
        }
    }
    unsigned invalidCases=0;
    const auto invalid=[&](NumiStaticVelocityBody a,NumiStaticVelocityBody b,V ga,V gb,float cap){
        auto r=numiStaticResolveVelocityPair(a,b,ga,gb,cap,true);
        const bool rejected=!r.valid&&words(r)==words(numiStaticResolveVelocityPair(a,b,ga,gb,cap,true));
        pass&=rejected;++invalidCases;
    };
    const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
    auto a=freeBody(2,{1,0,0}),b=freeBody(1,{-1,0,0},3),bad=a;
    bad.velocity.x=nan;invalid(bad,b,{-1,0,0},{1,0,0},0);bad=a;bad.position.x=inf;invalid(bad,b,{-1,0,0},{1,0,0},0);
    bad=a;bad.inverseMass=-1;invalid(bad,b,{-1,0,0},{1,0,0},0);bad=a;bad.radius=0;invalid(bad,b,{-1,0,0},{1,0,0},0);
    bad=a;bad.inverseMass=0;invalid(bad,b,{-1,0,0},{1,0,0},0);invalid(a,b,{nan,0,0},{1,0,0},0);invalid(a,b,{-1,0,0},{1,0,0},-1);invalid(a,b,{-1,0,0},{1,0,0},nan);
    bad=floorBody(2,{0,0,0},-.01f);invalid(bad,b,{-1,0,0},{1,0,0},0);
    auto tiny=freeBody(std::numeric_limits<float>::denorm_min(),{-1,0,0});
    invalid(tiny,freeBody(0,{}),{std::numeric_limits<float>::denorm_min(),0,0},{},0);
    unsigned negativeControls=0;
    for(const auto& c:cases)if(c.name=="normal_energy_regression"||c.name=="friction_energy_regression"){
        const auto old=oldPair(c);
        const long double oldGain=energyDelta(c.first,old[0])+energyDelta(c.second,old[1]);
        const float bound=numiStaticVelocityBodyEnergyUpper(c.first,old[0])+numiStaticVelocityBodyEnergyUpper(c.second,old[1]);
        pass&=oldGain>0&&bound>0;++negativeControls;
        std::cout<<std::setprecision(17)<<"negative_control="<<c.name<<" old_kinetic_gain_J="<<double(oldGain)<<" rejected_by_energy_oracle="<<(oldGain>0)<<'\n';
    }
    std::cout<<std::setprecision(17)<<"static_velocity_contact cases="<<cases.size()<<" invalid_controls="<<invalidCases<<" old_algebra_controls="<<negativeControls<<" exact_replays=2 maximum_velocity_error_m_s="<<maximumVelocityError<<" maximum_impulse_error_Ns="<<maximumImpulseError<<" maximum_support_error_Ns="<<maximumSupportError<<" maximum_momentum_error_Ns="<<maximumMomentumError<<" maximum_kinetic_gain_J="<<double(maximumEnergyGain)<<" backend=CPU_FP32 result="<<(pass?"PASS":"FAIL")<<'\n';
    return pass?0:1;
}

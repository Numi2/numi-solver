#include "numi/finite_bench_geometry.h"
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace numi::bench;
constexpr double mass=.2,radius=.07,inertia=.4*mass*radius*radius;
constexpr Vec3 gravity{0,0,-9.81};
constexpr double floorHeight=-.75,friction=.35;
struct State {
    Vec3 position{0,0,radius},velocity{1.1,0,0},angularVelocity{0,1.1/radius,0};
    Vec3 impulse{},angularImpulse{};
    double maximumFrictionRatio{},contactKineticLoss{};
    std::uint64_t benchContacts{},floorContacts{},edgeContacts{},vertexContacts{};
};
struct Trace { std::vector<State> frames; State final; double minimumGap{1e30};
    double maxMomentumError{},maxAngularMomentumError{}; bool leftBench{},reachedFloor{}; };
double kinetic(const State& state){return .5*mass*dot(state.velocity,state.velocity)+.5*inertia*dot(state.angularVelocity,state.angularVelocity);}
Vec3 angularMomentum(const State& state){return cross(state.position,state.velocity*mass)+state.angularVelocity*inertia;}

// A candidate owns its whole kick/drift/contact transaction. Failure does not
// publish a partially advanced body or partially accumulated impulse ledger.
State advance(const State& accepted,const Box& box,double dt,double coefficient=friction){
    validate(box,accepted.position,radius);
    if(!finite(accepted.velocity)||!finite(accepted.angularVelocity)||!std::isfinite(dt)||dt<=0||
       !std::isfinite(coefficient)||coefficient<0)
        throw std::invalid_argument("invalid finite bench step");
    State candidate=accepted;
    Vec3 gravityImpulse=gravity*(mass*dt);
    candidate.impulse=candidate.impulse+gravityImpulse;
    candidate.angularImpulse=candidate.angularImpulse+cross(candidate.position,gravityImpulse);
    candidate.velocity=candidate.velocity+gravity*dt;
    double remaining=dt;
    for(unsigned event=0;event<4&&remaining>0;++event){
        Vec3 end=candidate.position+candidate.velocity*remaining;
        Hit bench=cast(box,candidate.position,end,radius);
        Hit floor=castFloor(candidate.position,end,radius,floorHeight);
        bool onBench=bench.contact&&(!floor.contact||bench.time<=floor.time);
        Hit hit=onBench?bench:floor;
        if(!hit.contact){candidate.position=end;remaining=0;break;}
        candidate.position=candidate.position+(end-candidate.position)*hit.time;
        if(!onBench)candidate.position.z=floorHeight+radius;
        double before=kinetic(candidate);
        double normalImpulse=-mass*dot(candidate.velocity,hit.normal);
        if(!(normalImpulse>0))throw std::runtime_error("nonclosing finite bench contact");
        Vec3 impulse=hit.normal*normalImpulse;
        candidate.velocity=candidate.velocity+impulse/mass;
        Vec3 arm=hit.normal*(-radius);
        Vec3 slip=candidate.velocity+cross(candidate.angularVelocity,arm);
        slip=slip-hit.normal*dot(slip,hit.normal);
        double speed=length(slip),tangentialImpulse=0;
        if(speed>1e-12){
            Vec3 tangent=slip/speed;
            double effectiveInverseMass=1/mass+dot(cross(arm,tangent),cross(arm,tangent))/inertia;
            tangentialImpulse=std::min(speed/effectiveInverseMass,coefficient*normalImpulse);
            Vec3 tangentImpulse=tangent*(-tangentialImpulse);
            candidate.velocity=candidate.velocity+tangentImpulse/mass;
            candidate.angularVelocity=candidate.angularVelocity+cross(arm,tangentImpulse)/inertia;
            impulse=impulse+tangentImpulse;
        }
        if(coefficient*normalImpulse>0)candidate.maximumFrictionRatio=std::max(candidate.maximumFrictionRatio,
            tangentialImpulse/(coefficient*normalImpulse));
        candidate.impulse=candidate.impulse+impulse;
        candidate.angularImpulse=candidate.angularImpulse+cross(candidate.position+arm,impulse);
        candidate.contactKineticLoss+=before-kinetic(candidate);
        candidate.benchContacts+=onBench;candidate.floorContacts+=!onBench;
        candidate.edgeContacts+=onBench&&hit.feature==2;candidate.vertexContacts+=onBench&&hit.feature==3;
        remaining*=1-hit.time;
    }
    if(remaining>1e-15||!finite(candidate.position)||!finite(candidate.velocity)||!finite(candidate.angularVelocity)||
       sample(box,candidate.position,radius).gap< -1e-9||candidate.position.z<floorHeight+radius-1e-9)
        throw std::runtime_error("finite bench candidate did not close contact");
    return candidate;
}
Trace simulate(double dt){
    Box box;Trace trace;State state;State initial=state;
    std::uint32_t steps=std::uint32_t(std::lround(2/dt)),stride=steps/240;
    trace.frames.push_back(state);
    for(std::uint32_t step=1;step<=steps;++step){
        state=advance(state,box,dt);
        trace.minimumGap=std::min({trace.minimumGap,sample(box,state.position,radius).gap,state.position.z-floorHeight-radius});
        trace.maxMomentumError=std::max(trace.maxMomentumError,
            length((state.velocity-initial.velocity)*mass-state.impulse));
        trace.maxAngularMomentumError=std::max(trace.maxAngularMomentumError,
            length(angularMomentum(state)-angularMomentum(initial)-state.angularImpulse));
        trace.leftBench|=state.position.x>box.maximum.x+radius&&state.position.z<radius;
        trace.reachedFloor|=state.floorContacts>0&&std::abs(state.position.z-floorHeight-radius)<1e-10;
        if(step%stride==0)trace.frames.push_back(state);
    }
    trace.final=state;return trace;
}
std::uint64_t hash(const Trace& trace){
    std::uint64_t value=1469598103934665603ull;
    auto append=[&](std::uint64_t bits){for(unsigned byte=0;byte<8;++byte){value^=(bits>>(8*byte))&255;value*=1099511628211ull;}};
    for(const auto& frame:trace.frames){
        for(Vec3 vector:{frame.position,frame.velocity,frame.angularVelocity,frame.impulse,frame.angularImpulse})
            for(double x:{vector.x,vector.y,vector.z})append(std::bit_cast<std::uint64_t>(x));
        append(std::bit_cast<std::uint64_t>(frame.maximumFrictionRatio));
        append(std::bit_cast<std::uint64_t>(frame.contactKineticLoss));
        for(auto count:{frame.benchContacts,frame.floorContacts,frame.edgeContacts,frame.vertexContacts})append(count);
    }
    return value;
}
bool geometryProbe(){
    const Box box;bool passed=true;
    struct Case { Vec3 start,end,normal; double time; unsigned feature; };
    const double edgeTime=(.3-.1/std::sqrt(2.0))/.6;
    const double cornerTime=(.3-.1/std::sqrt(3.0))/.6;
    std::array<Case,4> cases{{
        {{0,0,1},{0,0,-1},{0,0,1},.45,1},
        {{1.2,0,-.04},{0,0,-.04},{1,0,0},(1.2-.85)/1.2,1},
        {{1.05,0,.3},{.45,0,-.3},{1/std::sqrt(2.0),0,1/std::sqrt(2.0)},edgeTime,2},
        {{1.05,.8,.3},{.45,.2,-.3},{1/std::sqrt(3.0),1/std::sqrt(3.0),1/std::sqrt(3.0)},cornerTime,3},
    }};
    for(unsigned index=0;index<cases.size();++index){
        auto c=cases[index];auto hit=cast(box,c.start,c.end,.1);
        bool valid=hit.contact&&hit.feature==c.feature&&std::abs(hit.time-c.time)<1e-12&&length(hit.normal-c.normal)<1e-12;
        passed&=valid;std::cout<<"finite_bench_geometry_case="<<index<<" feature="<<hit.feature<<" impact_time="<<hit.time<<" passed="<<valid<<'\n';
    }
    bool cornerMiss=!cast(box,{.84,.59,1},{.84,.59,-1},.1).contact;
    bool tangentMiss=!cast(box,{-1,0,.1001},{1,0,.1001},.1).contact;
    bool insideDistance=std::abs(sample(box,{0,0,-.04},.1).gap+.14)<1e-12;
    bool outsideDistance=std::abs(sample(box,{.84,.59,.09},.1).gap-(.09*std::sqrt(3.0)-.1))<1e-12;
    bool topSupport=cast(box,{0,0,.1},{0,0,.099},.1).time==0;
    bool sideClear=!cast(box,{1,0,.1},{1,0,-.1},.1).contact;
    passed&=cornerMiss&&tangentMiss&&insideDistance&&outsideDistance&&topSupport&&sideClear;
    std::cout<<"expanded_AABB_corner_false_contact_avoided="<<cornerMiss<<" grazing_miss="<<tangentMiss
             <<" signed_distances_exact="<<(insideDistance&&outsideDistance)<<" initial_top_support="<<topSupport
             <<" unsupported_outside_edge="<<sideClear<<'\n';
    unsigned rejections=0;State accepted;
    for(unsigned index=0;index<4;++index){
        try{Box malformed=box;State input=accepted;
            if(index==0)malformed.maximum.x=malformed.minimum.x;
            if(index==1)input.position.z=-.04;
            if(index==2)input.velocity.x=std::numeric_limits<double>::quiet_NaN();
            (void)advance(input,malformed,index==3?-1:1e-4);
        }catch(const std::exception&){++rejections;}
    }
    std::cout<<"invalid_step_rejections="<<rejections<<'\n';
    return passed&&rejections==4;
}
bool frictionProbe(){
    State input;input.angularVelocity={};double dt=1.0/6000;
    State sliding=advance(input,{},dt,.35),sticking=advance(input,{},dt,10000);
    double slidingImpulse=.35*mass*9.81*dt;
    double expectedSliding=1.1-slidingImpulse/mass;
    double expectedSpin=radius*slidingImpulse/inertia;
    // Effective inverse mass is 1/m + r^2/I, so the final translation
    // is 5/7 of its initial value once surface slip is eliminated.
    const double expectedSticking=1.1*(1-1/(1+mass*radius*radius/inertia));
    bool passed=std::abs(sliding.velocity.x-expectedSliding)<1e-12&&
        std::abs(sliding.angularVelocity.y-expectedSpin)<1e-12&&
        std::abs(sticking.velocity.x-expectedSticking)<1e-12&&
        std::abs(sticking.velocity.x-radius*sticking.angularVelocity.y)<1e-12&&
        kinetic(sliding)<kinetic(input)&&kinetic(sticking)<kinetic(input)&&
        sliding.maximumFrictionRatio<=1+1e-12&&sticking.maximumFrictionRatio<=1+1e-12;
    std::cout<<"finite_bench_sliding_vx_m_s="<<sliding.velocity.x<<" sticking_vx_m_s="<<sticking.velocity.x
        <<" sticking_slip_m_s="<<sticking.velocity.x-radius*sticking.angularVelocity.y
        <<" friction_analytic_cases_passed="<<passed<<'\n';
    return passed;
}
int run(const std::string& trajectory){
    std::cout<<std::setprecision(12);
    bool geometry=geometryProbe(),frictionCorrect=frictionProbe();
    auto first=simulate(1.0/6000),second=simulate(1.0/6000),fine=simulate(1.0/12000);
    double refinementError=0;
    for(std::size_t index=0;index<first.frames.size();++index)
        refinementError=std::max(refinementError,length(first.frames[index].position-fine.frames[index].position));
    bool exact=hash(first)==hash(second);
    bool qualified=geometry&&frictionCorrect&&exact&&first.leftBench&&first.reachedFloor&&fine.leftBench&&fine.reachedFloor&&
        first.minimumGap>=-1e-9&&fine.minimumGap>=-1e-9&&refinementError<.002&&
        first.final.maximumFrictionRatio<=1+1e-12&&fine.final.maximumFrictionRatio<=1+1e-12&&
        first.maxMomentumError<1e-10&&fine.maxMomentumError<1e-10&&
        first.maxAngularMomentumError<1e-10&&fine.maxAngularMomentumError<1e-10;
    if(!trajectory.empty()){
        std::ofstream output(trajectory);if(!output)throw std::runtime_error("cannot write finite bench trajectory");
        output<<"frame,time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,wx_rad_s,wy_rad_s,wz_rad_s,bench_contacts,floor_contacts,edge_contacts\n"<<std::setprecision(17);
        for(std::size_t index=0;index<first.frames.size();++index){auto s=first.frames[index];
            output<<index<<','<<double(index)/120<<','<<s.position.x<<','<<s.position.y<<','<<s.position.z<<','
                <<s.velocity.x<<','<<s.velocity.y<<','<<s.velocity.z<<','<<s.angularVelocity.x<<','<<s.angularVelocity.y<<','<<s.angularVelocity.z<<','
                <<s.benchContacts<<','<<s.floorContacts<<','<<s.edgeContacts<<'\n';}
    }
    std::cout<<"backend=CPU_FP64 finite_bench_x_half_extent_m=.75 y_half_extent_m=.5 thickness_m=.08 floor_height_m=-.75\n"
             <<"mass_kg="<<mass<<" radius_m="<<radius<<" friction="<<friction<<" restitution=0 simulated_seconds=2\n"
             <<"replay_exact="<<exact<<" hash=0x"<<std::hex<<hash(first)<<std::dec<<" captured_frames="<<first.frames.size()
             <<" maximum_timestep_position_difference_m="<<refinementError<<'\n'
             <<"left_finite_bench="<<first.leftBench<<" reached_lower_floor="<<first.reachedFloor<<" minimum_contact_gap_m="<<first.minimumGap<<'\n'
             <<"final_height_m="<<first.final.position.z<<" final_vertical_velocity_m_s="<<first.final.velocity.z
             <<" bench_contacts="<<first.final.benchContacts<<" curved_edge_contacts="<<first.final.edgeContacts<<" floor_contacts="<<first.final.floorContacts<<'\n'
             <<"maximum_linear_momentum_discrepancy_Ns="<<first.maxMomentumError
             <<" maximum_angular_momentum_discrepancy_Nms="<<first.maxAngularMomentumError
             <<" maximum_friction_cone_ratio="<<first.final.maximumFrictionRatio<<'\n'
             <<"finite_bench_reference_qualified="<<qualified<<'\n';
    return qualified?0:1;
}
}
int main(int argc,char** argv)try{
    std::string trajectory;
    if(argc==3&&std::string(argv[1])=="--trajectory")trajectory=argv[2];
    else if(argc!=1)throw std::invalid_argument("usage: numi-solver-finite-bench [--trajectory CSV]");
    return run(trajectory);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}

#include "numi/transactional_substep.h"

#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr unsigned collisionFailure=256,advanceFailure=128,nonfiniteFailure=2;
struct State {
    double time{},position{},velocity{},normalImpulse{},controller{};
    unsigned generation{7};
    std::array<double,3> warm{{.1,.2,.3}};
};
bool exact(const State& a,const State& b) {
    return a.time==b.time && a.position==b.position && a.velocity==b.velocity &&
           a.normalImpulse==b.normalImpulse && a.controller==b.controller &&
           a.generation==b.generation && a.warm==b.warm;
}
void require(bool condition,const std::string& message) {if(!condition)throw std::runtime_error(message);}
}

int main() {
    try {
        numi::TransactionPolicy policy{collisionFailure|advanceFailure,8,512};
        const State initial{0,2,-3,0,0,7,{{.1,.2,.3}}};
        const float duration=.125f;
        unsigned cases=0;
        // An inelastic mass-2 floor impact has an independent analytic result.
        // Refinement is requested by a physical travel/clearance admission
        // bound, never by a preselected recursion depth. Rejected copies are
        // deliberately corrupted to detect warm/clock/controller leakage.
        const auto floorAttempt=[&](State& s,const numi::TransactionStep& step) {
            const double travel=-s.velocity*step.timestep;
            if(travel>s.position*.75 && s.position>0) {
                s.time=123;s.position=-456;s.generation=99;s.warm={{8,9,10}};
                return collisionFailure|advanceFailure;
            }
            const double next=s.position+s.velocity*step.timestep;
            if(next<0) {s.normalImpulse+=-2*s.velocity;s.velocity=0;s.position=0;}else s.position=next;
            s.time+=step.timestep;
            s.controller=step.endFraction*step.endFraction;
            s.warm[0]=s.time;s.generation=step.endFraction==1?8:7;
            return 0u;
        };
        // This short interval is collision-free and must preserve the exact
        // analytic time/position, arbitrary warm state, and final controller.
        const auto free=numi::advanceTransactional(initial,duration,policy,floorAttempt);
        require(free.accepted()&&free.state.time==duration&&free.state.position==2-3*duration&&
                free.state.controller==1&&free.state.generation==8&&free.state.warm[1]==.2,"free analytic drift failed");++cases;
        // A backend rejects travel exceeding 0.25m and otherwise integrates
        // the actual point impact. Complete time and J=m*3 remain invariant.
        const auto impactAttempt=[&](State& s,const numi::TransactionStep& step) {
            if(std::abs(s.velocity*step.timestep)>.25) {
                s.time=-999;s.position=-999;s.generation=42;s.warm={{99,99,99}};
                return advanceFailure;
            }
            const double next=s.position+s.velocity*step.timestep;
            if(next<0) {s.normalImpulse+=-2*s.velocity;s.velocity=0;s.position=0;}else s.position=next;
            s.time+=step.timestep;s.controller=step.endFraction*step.endFraction;
            s.generation=step.endFraction==1?8:7;s.warm[0]=s.time;
            return 0u;
        };
        const auto impact=numi::advanceTransactional(initial,1.f,policy,impactAttempt);
        const auto replay=numi::advanceTransactional(initial,1.f,policy,impactAttempt);
        require(impact.accepted()&&impact.state.time==1&&impact.state.position==0&&impact.state.velocity==0&&
                impact.state.normalImpulse==6&&impact.state.controller==1&&impact.state.generation==8&&
                impact.ledger.rejectedAttempts>0&&impact.ledger.committedSeconds==1&&exact(impact.state,replay.state),
                "refined analytic impact/time/impulse/replay failed");++cases;
        const auto lateFatal=numi::advanceTransactional(initial,1.f,policy,[&](State& s,const numi::TransactionStep& step) {
            if(step.depth==0){s.position=-999;return advanceFailure;}
            if(step.beginFraction>=.5){s.time=-999;s.generation=99;return nonfiniteFailure|advanceFailure;}
            return impactAttempt(s,step);
        });
        require(!lateFatal.accepted()&&lateFatal.ledger.status==numi::TransactionStatus::fatalFailure&&
                lateFatal.ledger.provisionalAcceptedLeaves>0&&lateFatal.ledger.committedSeconds==0&&exact(lateFatal.state,initial),
                "accepted left branch leaked after later fatal rejection");++cases;
        for(const unsigned depth:{0u,2u,8u}) {
            auto bounded=policy;bounded.maximumDepth=depth;
            const auto exhausted=numi::advanceTransactional(initial,1.f,bounded,[](State& s,const auto&) {
                s.position=-999;s.time=-999;s.warm={{9,9,9}};return collisionFailure;
            });
            require(!exhausted.accepted()&&exhausted.ledger.status==numi::TransactionStatus::depthLimit&&
                    exhausted.ledger.committedSeconds==0&&exact(exhausted.state,initial),"depth exhaustion published partial state");++cases;
        }
        auto bounded=policy;bounded.maximumAttempts=2;
        const auto attemptLimit=numi::advanceTransactional(initial,1.f,bounded,[](State& s,const auto&) {
            s.velocity=999;return collisionFailure;
        });
        require(attemptLimit.ledger.status==numi::TransactionStatus::attemptLimit&&exact(attemptLimit.state,initial)&&
                attemptLimit.ledger.attempts.size()==2&&attemptLimit.ledger.committedSeconds==0,"attempt limit changed state/time");++cases;
        const auto underflow=numi::advanceTransactional(initial,std::numeric_limits<float>::denorm_min(),policy,
            [](State& s,const auto&) {s.time=1;return collisionFailure;});
        require(underflow.ledger.status==numi::TransactionStatus::timestepUnderflow&&exact(underflow.state,initial),
                "underflow accepted a zero-duration subdivision");++cases;
        unsigned oddHalfControls=0;
        for(const std::uint32_t bits:{3u,5u,7u,0x00800001u}) {
            const float dt=std::bit_cast<float>(bits);
            const auto roundedHalf=numi::advanceTransactional(initial,dt,policy,
                [](State& s,const auto& step) {s.time+=step.timestep;return step.depth==0?advanceFailure:0u;});
            require(roundedHalf.ledger.status==numi::TransactionStatus::timestepUnderflow&&
                    roundedHalf.ledger.committedSeconds==0&&exact(roundedHalf.state,initial),
                    "rounded nonzero halves changed total duration");++oddHalfControls;
        }
        unsigned invalidControls=0;
        for(float dt:{0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            bool called=false;
            const auto invalid=numi::advanceTransactional(initial,dt,policy,[&](State&,const auto&) {called=true;return 0u;});
            require(invalid.ledger.status==numi::TransactionStatus::invalidInput&&!called&&exact(invalid.state,initial),"invalid timestep entered backend");++invalidControls;
        }
        bool threw=false;
        try {numi::advanceTransactional(initial,1.f,policy,[](State& s,const auto&)->unsigned {
            s.position=-999;throw std::runtime_error("injected backend exception");
        });}catch(const std::runtime_error&){threw=true;}
        require(threw&&initial.position==2&&initial.time==0&&initial.generation==7,"backend exception damaged caller state");++cases;
        std::cout<<std::setprecision(17)<<"transaction_cases="<<cases<<" invalid_controls="<<invalidControls
                 <<" odd_half_controls="<<oddHalfControls
                 <<" analytic_impact_normal_impulse_Ns="<<impact.state.normalImpulse
                 <<" committed_time_s="<<impact.ledger.committedSeconds
                 <<" rejected_attempts="<<impact.ledger.rejectedAttempts
                 <<" accepted_leaves="<<impact.ledger.provisionalAcceptedLeaves
                 <<" exact_replay="<<exact(impact.state,replay.state)<<" transaction_probe_passed=1\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

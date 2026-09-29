#pragma once

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// Host driver contract: State must own its physical/warm state by value.
// An attempt may mutate only the provided private copy. A rejected candidate
// cannot advance the caller's state or clock. The callback's failure word is
// authoritative; subdivision never changes its physical acceptance budgets.
namespace numi {
struct TransactionStep {
    double beginFraction{}, endFraction{1};
    float timestep{};
    unsigned depth{};
};
struct TransactionAttempt {
    TransactionStep step;
    std::uint32_t failure{};
};
enum class TransactionStatus {
    accepted, invalidInput, fatalFailure, depthLimit, attemptLimit, timestepUnderflow
};
struct TransactionPolicy {
    std::uint32_t retryableFailureMask{};
    unsigned maximumDepth{8};
    unsigned maximumAttempts{512};
};
struct TransactionLedger {
    std::vector<TransactionAttempt> attempts;
    unsigned rejectedAttempts{}, provisionalAcceptedLeaves{}, maximumDepth{};
    double committedSeconds{};
    TransactionStatus status{TransactionStatus::invalidInput};
    std::uint32_t terminalFailure{};
};
template<class State> struct TransactionResult {
    State state;
    TransactionLedger ledger;
    bool accepted() const { return ledger.status == TransactionStatus::accepted; }
};

template<class State, class Attempt>
TransactionResult<State> advanceTransactional(const State& initial, float timestep,
                                              const TransactionPolicy& policy, Attempt&& attempt) {
    TransactionResult<State> output{initial,{}};
    if (!std::isfinite(timestep) || !(timestep > 0) || policy.maximumAttempts == 0 ||
        policy.maximumDepth > 30 || policy.retryableFailureMask == 0) return output;
    TransactionLedger& ledger=output.ledger;
    const auto refine=[&](auto&& self,const State& source,const TransactionStep& step,State& accepted)->bool {
        ledger.maximumDepth=ledger.maximumDepth>step.depth?ledger.maximumDepth:step.depth;
        if(ledger.attempts.size()>=policy.maximumAttempts) {
            ledger.status=TransactionStatus::attemptLimit; return false;
        }
        State candidate=source;
        const auto failure=static_cast<std::uint32_t>(attempt(candidate,step));
        ledger.attempts.push_back({step,failure});
        if(failure==0) {
            ++ledger.provisionalAcceptedLeaves;
            accepted=std::move(candidate); return true;
        }
        ++ledger.rejectedAttempts;
        ledger.terminalFailure=failure;
        if((failure & ~policy.retryableFailureMask)!=0) {
            ledger.status=TransactionStatus::fatalFailure; return false;
        }
        if(step.depth>=policy.maximumDepth) {
            ledger.status=TransactionStatus::depthLimit; return false;
        }
        const float half=step.timestep*.5f;
        const double middle=step.beginFraction+(step.endFraction-step.beginFraction)*.5;
        // An odd subnormal may round half upward or downward while remaining
        // nonzero. Two such children would change the requested duration.
        if(!(half>0) || static_cast<double>(half)*2!=static_cast<double>(step.timestep) ||
           !(middle>step.beginFraction) || !(middle<step.endFraction)) {
            ledger.status=TransactionStatus::timestepUnderflow; return false;
        }
        State left=source;
        if(!self(self,source,{step.beginFraction,middle,half,step.depth+1},left)) return false;
        return self(self,left,{middle,step.endFraction,half,step.depth+1},accepted);
    };
    State complete=initial;
    if(refine(refine,initial,{0,1,timestep,0},complete)) {
        output.state=std::move(complete);
        ledger.committedSeconds=static_cast<double>(timestep);
        ledger.terminalFailure=0;
        ledger.status=TransactionStatus::accepted;
    }
    // On any incomplete branch, output.state remains the original, including
    // warm/controller generations. Accepted leaves were private proposals.
    return output;
}
}

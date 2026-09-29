#pragma once

#include "static_contact_geometry.h"

// One contact impulse, coupled to each endpoint's existing static support.
// Static support is unilateral in trial velocity: an outward-moving endpoint
// can spend its velocity slack before requiring a blocked support response.
// This shared FP32 path requires IEEE arithmetic (no fast-math).
#ifdef __METAL_VERSION__
inline float numiVelocityDown(float x) { return metal::nextafter(x, -INFINITY); }
inline float numiVelocityUp(float x) { return metal::nextafter(x, INFINITY); }
#else
inline float numiVelocityDown(float x) { return std::nextafter(x, -INFINITY); }
inline float numiVelocityUp(float x) { return std::nextafter(x, INFINITY); }
#endif

struct NumiStaticVelocityBody {
    NumiStaticVec3 position, velocity;
    float inverseMass, radius;
};
enum NumiStaticVelocityStatus : unsigned {
    NumiStaticVelocityInactive = 0,
    NumiStaticVelocitySolved = 1,
    NumiStaticVelocityLimited = 2,
    NumiStaticVelocityInvalid = 3,
    NumiStaticVelocityInfeasible = 4,
    NumiStaticVelocityIterationLimit = 5,
    NumiStaticVelocityEnergyFailure = 6
};
struct NumiStaticVelocityPairResult {
    NumiStaticVec3 firstVelocity, secondVelocity;
    float impulse, firstSupportImpulse, secondSupportImpulse;
    unsigned iterations, status;
    bool valid;
};
struct NumiStaticVelocityTrial {
    NumiStaticVec3 velocity;
    float supportImpulse;
    bool valid;
};
struct NumiStaticVelocityPairTrial {
    NumiStaticVelocityTrial first, second;
    float derivative;
    bool valid;
};

inline bool numiStaticVelocityValidBody(NumiStaticVelocityBody body) {
    if (!numiStaticFinite3(body.position) || !numiStaticFinite3(body.velocity) ||
        !numiStaticFinite(body.inverseMass) || body.inverseMass < 0 ||
        !numiStaticFinite(body.radius) || !(body.radius > 0)) return false;
    // A moving kinematic boundary needs an external-work ledger. This pair
    // helper certifies passive dynamic bodies and stationary pinned endpoints.
    if (body.inverseMass == 0 &&
        (body.velocity.x != 0 || body.velocity.y != 0 || body.velocity.z != 0)) return false;
    const auto sample = numiStaticSceneSample(body.position, body.radius);
    return sample.valid && sample.gap >= -numiStaticTolerance(body.position, body.position) &&
        numiStaticFinite3(sample.normal);
}

inline NumiStaticVelocityTrial numiStaticVelocityTrial(
    NumiStaticVelocityBody body, NumiStaticVec3 gradient, float impulse
) {
    NumiStaticVelocityTrial out{body.velocity, 0, true};
    if (body.inverseMass == 0) return out;
    const auto response = numiStaticScale(gradient, body.inverseMass);
    out.velocity = {
        numiStaticFma(response.x, impulse, body.velocity.x),
        numiStaticFma(response.y, impulse, body.velocity.y),
        numiStaticFma(response.z, impulse, body.velocity.z)
    };
    if (!numiStaticFinite3(out.velocity)) { out.valid = false; return out; }
    const auto sample = numiStaticSceneSample(body.position, body.radius);
    if (!sample.valid) { out.valid = false; return out; }
    if (sample.gap <= numiStaticTolerance(body.position, body.position)) {
        const float squared = numiStaticDot(sample.normal, sample.normal);
        const float incoming = numiStaticDot(out.velocity, sample.normal);
        if (!numiStaticFinite(squared) || !(squared > 0) || !numiStaticFinite(incoming)) {
            out.valid = false; return out;
        }
        if (incoming < 0) {
            const auto before = out.velocity;
            out.velocity = numiStaticSub(out.velocity,
                numiStaticScale(sample.normal, incoming / squared));
            out.supportImpulse = numiStaticLength(numiStaticSub(out.velocity, before)) / body.inverseMass;
        }
    }
    out.valid = numiStaticFinite3(out.velocity) && numiStaticFinite(out.supportImpulse) && out.supportImpulse >= 0;
    return out;
}

inline NumiStaticVelocityPairTrial numiStaticVelocityEvaluate(
    NumiStaticVelocityBody first, NumiStaticVelocityBody second,
    NumiStaticVec3 gradientA, NumiStaticVec3 gradientB, float impulse
) {
    NumiStaticVelocityPairTrial out{
        numiStaticVelocityTrial(first, gradientA, impulse),
        numiStaticVelocityTrial(second, gradientB, impulse), 0, true};
    out.derivative = numiStaticDot(gradientA, out.first.velocity) +
                     numiStaticDot(gradientB, out.second.velocity);
    out.valid = out.first.valid && out.second.valid && numiStaticFinite(out.derivative);
    return out;
}

// Outward-rounded upper bound on the actual kinetic-energy change of the
// represented input/output velocities: (v_out-v_in) dot (v_out+v_in) / (2*m^-1).
// This is independent of the contact derivative used to find the impulse.
// Identical represented states contribute exactly zero, avoiding roundoff-only
// rejection of inactive contacts. Positive or unresolved energy change rejects.
inline float numiStaticVelocityBodyEnergyUpper(
    NumiStaticVelocityBody body, NumiStaticVec3 output
) {
    if (body.inverseMass == 0) return 0;
    float total = 0;
    bool changed = false;
    for (unsigned axis = 0; axis < 3; ++axis) {
        const float before = numiStaticComponent(body.velocity, axis);
        const float after = numiStaticComponent(output, axis);
        if (before == after) continue;
        changed = true;
        const float dl = numiVelocityDown(after - before), du = numiVelocityUp(after - before);
        const float sl = numiVelocityDown(after + before), su = numiVelocityUp(after + before);
        const float product = numiStaticMax(numiStaticMax(dl * sl, dl * su),
                                           numiStaticMax(du * sl, du * su));
        const float term = numiVelocityUp(numiVelocityUp(product) / body.inverseMass);
        total = numiVelocityUp(total + term);
    }
    return changed ? numiVelocityUp(total * 0.5f) : 0;
}

inline NumiStaticVelocityPairResult numiStaticResolveVelocityPair(
    NumiStaticVelocityBody first, NumiStaticVelocityBody second,
    NumiStaticVec3 gradientA, NumiStaticVec3 gradientB,
    float requestedMaximumImpulse, bool uncappedNormal
) {
    NumiStaticVelocityPairResult result{first.velocity, second.velocity, 0, 0, 0,
        0, NumiStaticVelocityInvalid, false};
    if (!numiStaticVelocityValidBody(first) || !numiStaticVelocityValidBody(second) ||
        !numiStaticFinite3(gradientA) || !numiStaticFinite3(gradientB) ||
        !numiStaticFinite(requestedMaximumImpulse) || requestedMaximumImpulse < 0) return result;
    auto final = numiStaticVelocityEvaluate(first, second, gradientA, gradientB, 0);
    if (!final.valid) return result;
    unsigned status = NumiStaticVelocityInactive;
    float impulse = 0;
    if (final.derivative < 0 && (uncappedNormal || requestedMaximumImpulse > 0)) {
        const float freeSlope = first.inverseMass * numiStaticDot(gradientA, gradientA) +
                                second.inverseMass * numiStaticDot(gradientB, gradientB);
        if (!numiStaticFinite(freeSlope) || !(freeSlope > 0)) {
            result.status = NumiStaticVelocityInfeasible; return result;
        }
        float low = 0;
        float high = -final.derivative / freeSlope;
        if (!numiStaticFinite(high)) { result.status = NumiStaticVelocityInfeasible; return result; }
        if (!(high > 0)) high = numiVelocityUp(0);
        if (!uncappedNormal) high = numiStaticMin(high, requestedMaximumImpulse);
        bool bracketed = false;
        for (unsigned iteration = 0; iteration < 160; ++iteration) {
            ++result.iterations;
            final = numiStaticVelocityEvaluate(first, second, gradientA, gradientB, high);
            if (!final.valid) return result;
            if (final.derivative >= 0) { bracketed = true; break; }
            if (!uncappedNormal && high == requestedMaximumImpulse) {
                impulse = high; status = NumiStaticVelocityLimited; break;
            }
            low = high;
            float next = high * 2;
            if (!numiStaticFinite(next)) { result.status = NumiStaticVelocityInfeasible; return result; }
            if (!(next > high)) next = numiVelocityUp(high);
            if (!uncappedNormal) next = numiStaticMin(next, requestedMaximumImpulse);
            if (!numiStaticFinite(next) || !(next > high)) {
                result.status = NumiStaticVelocityInfeasible; return result;
            }
            high = next;
        }
        if (bracketed) {
            bool exhausted = true;
            for (unsigned iteration = 0; iteration < 96; ++iteration) {
                if (numiVelocityUp(low) >= high) { exhausted = false; break; }
                ++result.iterations;
                const float middle = low + (high - low) * 0.5f;
                if (!(middle > low) || !(middle < high)) { exhausted = false; break; }
                const auto trial = numiStaticVelocityEvaluate(first, second, gradientA, gradientB, middle);
                if (!trial.valid) return result;
                if (trial.derivative >= 0) { high = middle; final = trial; }
                else low = middle;
            }
            if (exhausted) { result.status = NumiStaticVelocityIterationLimit; return result; }
            impulse = high; status = NumiStaticVelocitySolved;
        } else if (status != NumiStaticVelocityLimited) {
            result.status = NumiStaticVelocityIterationLimit; return result;
        }
    } else if (final.derivative < 0) status = NumiStaticVelocityLimited;
    const float firstEnergy = numiStaticVelocityBodyEnergyUpper(first, final.first.velocity);
    const float secondEnergy = numiStaticVelocityBodyEnergyUpper(second, final.second.velocity);
    const float energyUpper = firstEnergy == 0 && secondEnergy == 0 ? 0 : numiVelocityUp(firstEnergy + secondEnergy);
    if (!numiStaticFinite(energyUpper) || energyUpper > 0) {
        result.status = NumiStaticVelocityEnergyFailure; return result;
    }
    result.firstVelocity = final.first.velocity;
    result.secondVelocity = final.second.velocity;
    result.impulse = impulse;
    result.firstSupportImpulse = final.first.supportImpulse;
    result.secondSupportImpulse = final.second.supportImpulse;
    result.status = status; result.valid = true;
    return result;
}

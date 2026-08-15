#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_TEMPORAL_CONE_PROBE_ABI_VERSION 1u
#define NUMI_TEMPORAL_CONE_PROBE_MAX_ITERATIONS 64u

enum NumiTemporalConeProbeStatus : mr_u32 {
    NUMI_TEMPORAL_CONE_PROBE_SUCCESS = 0u,
    NUMI_TEMPORAL_CONE_PROBE_INVALID_ABI = 1u,
    NUMI_TEMPORAL_CONE_PROBE_INVALID_INPUT = 2u,
    NUMI_TEMPORAL_CONE_PROBE_FACTORIZATION_FAILED = 3u,
    NUMI_TEMPORAL_CONE_PROBE_NONFINITE_RESULT = 4u,
};

// One local contact problem:
//
//     minimize 1/2 lambda^T A lambda + v_free^T lambda
//
// under a unilateral normal impulse and an elliptic Coulomb friction bound.
// A is the coupled normal/tangent point response used by Temporal Cone.
typedef struct MR_ALIGN16 NumiTemporalConeProbeInput {
    mr_float4 responseRow0;
    mr_float4 responseRow1;
    mr_float4 responseRow2;

    // xyz free relative contact velocity; w effective friction in tangent U.
    mr_float4 freeVelocityAndFrictionU;
    // xyz warm-start impulse; w effective friction in tangent V.
    mr_float4 warmImpulseAndFrictionV;
    // x maximum normal impulse (zero means unbounded); yzw reserved.
    mr_float4 limits;

    // x ABI version, y projected iterations, z emits authored violation,
    // w reserved.
    mr_uint4 control;
} NumiTemporalConeProbeInput;

typedef struct MR_ALIGN16 NumiTemporalConeProbeOutput {
    // xyz accepted impulse; w maximum impulse delta in the final iteration.
    mr_float4 impulseAndDelta;
    // xyz conditioned inverse row; w authored warm-start cone violation.
    mr_float4 inverseRow0;
    mr_float4 inverseRow1;
    mr_float4 inverseRow2;
    // xyz A * lambda + v_free; w elliptic-cone violation.
    mr_float4 residualAndConeViolation;
    // x status, y completed iterations, zw reserved.
    mr_uint4 status;
} NumiTemporalConeProbeOutput;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiTemporalConeProbeInput) == 112);
static_assert(sizeof(NumiTemporalConeProbeOutput) == 96);
#endif

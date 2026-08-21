#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_CLOTH_BAG_GPU_ABI_VERSION 7u
#define NUMI_CLOTH_BAG_GPU_INVALID_PARTICLE 0xffffffffu

enum NumiClothBagGPUFailure : mr_u32 {
    NUMI_CLOTH_BAG_GPU_FAILURE_NONE = 0u,
    NUMI_CLOTH_BAG_GPU_FAILURE_ABI = 1u << 0u,
    NUMI_CLOTH_BAG_GPU_FAILURE_RANGE = 1u << 1u,
    NUMI_CLOTH_BAG_GPU_FAILURE_BATCH = 1u << 2u,
    NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE = 1u << 3u,
};

typedef struct MR_ALIGN16 NumiClothBagGPUConfig {
    // x ABI, y particle count, z distance count, w grip count.
    mr_uint4 control;
    // x crossing-angle knot count, y yarn-bend count,
    // z 1 when ground response is active, w fruit count.
    mr_uint4 constraintCounts;
    // x fruit-pair count, y sphere/yarn candidate count,
    // z nonlocal yarn/yarn pair count, w yarn/yarn batch count.
    mr_uint4 contactCounts;
    // xyz gravitational acceleration, w substep timestep.
    mr_float4 gravityAndTimestep;
    // xyz virtual-handle position, w 1 when the grip is active.
    mr_float4 gripTargetAndActive;
    // x yarn radius, y guarded self-contact cell size,
    // z cloth/ground friction, w cloth/self friction.
    mr_float4 clothMaterial;
    // x fruit-pair friction, y fruit/ground friction,
    // z fruit rolling resistance, w fruit/yarn friction.
    mr_float4 fruitMaterial;
} NumiClothBagGPUConfig;

typedef struct MR_ALIGN16 NumiClothBagGPUParticle {
    // xyz current position, w inverse mass.
    mr_float4 positionAndInverseMass;
    // xyz position at the beginning of the substep, w authored mass.
    mr_float4 previousAndMass;
    // xyz velocity, w accumulated cloth/ground normal impulse after finalize.
    mr_float4 velocity;
} NumiClothBagGPUParticle;

typedef struct MR_ALIGN16 NumiClothBagGPUDistance {
    // x first particle, y second particle, z graph color, w yarn kind.
    mr_uint4 particlesAndColor;
    // x rest length, y XPBD compliance, z accumulated lambda,
    // w maximum relative extension for the unilateral limiter.
    mr_float4 material;
} NumiClothBagGPUDistance;

typedef struct MR_ALIGN16 NumiClothBagGPUGrip {
    // x particle, yzw reserved.
    mr_uint4 particle;
    // xyz target offset from the virtual handle, w XPBD compliance.
    mr_float4 targetOffsetAndCompliance;
    // xyz accumulated vector lambda, w reserved.
    mr_float4 lambda;
} NumiClothBagGPUGrip;

typedef struct MR_ALIGN16 NumiClothBagGPUKnot {
    // xy warp endpoints, zw weft endpoints.
    mr_uint4 particles;
    // x graph color, yzw reserved.
    mr_uint4 control;
    // x rest cosine, y XPBD compliance, z accumulated lambda, w reserved.
    mr_float4 material;
} NumiClothBagGPUKnot;

typedef struct MR_ALIGN16 NumiClothBagGPUBend {
    // x first particle, y middle particle, z third particle, w graph color.
    mr_uint4 particlesAndColor;
    // x rest chord, y rest arc, z XPBD compliance, w accumulated lambda.
    mr_float4 material;
} NumiClothBagGPUBend;

typedef struct MR_ALIGN16 NumiClothBagGPUFruit {
    // xyz center, w inverse mass.
    mr_float4 positionAndInverseMass;
    // xyz center at the beginning of the substep, w radius.
    mr_float4 previousAndRadius;
    // xyz linear velocity, w accumulated ground-normal impulse.
    mr_float4 velocityAndGroundImpulse;
    // xyz angular velocity, w reserved.
    mr_float4 angularVelocity;
    // Quaternion xyzw.
    mr_float4 orientation;
    // x appearance, yzw reserved.
    mr_uint4 identity;
} NumiClothBagGPUFruit;

typedef struct MR_ALIGN16 NumiClothBagGPUFruitPair {
    // x first fruit, y second fruit, z graph color, w stable pair index.
    mr_uint4 fruitsAndColor;
    // xyz accumulated impulse-weighted normal, w accumulated normal impulse.
    mr_float4 contact;
} NumiClothBagGPUFruitPair;

typedef struct MR_ALIGN16 NumiClothBagGPUYarnContact {
    // x fruit, y distance/yarn segment, z first knot, w second knot.
    mr_uint4 identity;
    // xyz current normal from fruit toward yarn, w signed surface separation.
    mr_float4 currentNormalAndSeparation;
    // xyz swept-impact normal from fruit toward yarn,
    // w removable post-impact normal advance.
    mr_float4 sweptNormalAndAdvance;
    // x current segment weight, y swept-impact segment weight,
    // z impact time in [0,1], w combined sphere/yarn radius.
    mr_float4 weightsAndTime;
    // x current overlap, y swept impact, z degenerate current normal,
    // w accepted normal-response count.
    mr_uint4 control;
    // xyz accumulated normal impulse on the fruit, w total normal impulse.
    mr_float4 fruitNormalAndImpulse;
    // x/y accumulated normal impulse weights on the first/second yarn knot,
    // z first accepted swept-impact time, w accumulated swept advance.
    mr_float4 segmentImpulse;
} NumiClothBagGPUYarnContact;

typedef struct NumiClothBagGPUSelfPair {
    // Indices of the two yarn segments in the distance table. Their four
    // particle endpoints are resolved from that owning table on demand.
    mr_u32 firstSegment;
    mr_u32 secondSegment;
} NumiClothBagGPUSelfPair;

typedef struct MR_ALIGN16 NumiClothBagGPUSelfStatus {
    // x accepted present contacts, y accepted swept contacts,
    // z maximum correction encoded as positive-float bits, w reserved.
    mr_uint4 counters;
} NumiClothBagGPUSelfStatus;

typedef struct MR_ALIGN16 NumiClothBagGPUFrictionStatus {
    // x fruit-pair, y fruit/yarn, z cloth/ground, w fruit/ground contacts.
    mr_uint4 counters;
    // x maximum Coulomb-cone ratio as positive-float bits,
    // y maximum rolling-resistance ratio as positive-float bits,
    // z rolling-resistance contact count, w reserved.
    mr_uint4 metrics;
} NumiClothBagGPUFrictionStatus;

typedef struct MR_ALIGN16 NumiClothBagGPUBatch {
    // x first constraint, y constraint count, z expected graph color,
    // w reserved.
    mr_uint4 control;
} NumiClothBagGPUBatch;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiClothBagGPUConfig) == 112);
static_assert(sizeof(NumiClothBagGPUParticle) == 48);
static_assert(sizeof(NumiClothBagGPUDistance) == 32);
static_assert(sizeof(NumiClothBagGPUGrip) == 48);
static_assert(sizeof(NumiClothBagGPUKnot) == 48);
static_assert(sizeof(NumiClothBagGPUBend) == 32);
static_assert(sizeof(NumiClothBagGPUFruit) == 96);
static_assert(sizeof(NumiClothBagGPUFruitPair) == 32);
static_assert(sizeof(NumiClothBagGPUYarnContact) == 112);
static_assert(sizeof(NumiClothBagGPUSelfPair) == 8);
static_assert(sizeof(NumiClothBagGPUSelfStatus) == 16);
static_assert(sizeof(NumiClothBagGPUFrictionStatus) == 32);
static_assert(sizeof(NumiClothBagGPUBatch) == 16);
#endif

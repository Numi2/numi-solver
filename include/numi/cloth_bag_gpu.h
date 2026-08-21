#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_CLOTH_BAG_GPU_ABI_VERSION 1u
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
    // xyz gravitational acceleration, w substep timestep.
    mr_float4 gravityAndTimestep;
    // xyz virtual-handle position, w 1 when the grip is active.
    mr_float4 gripTargetAndActive;
} NumiClothBagGPUConfig;

typedef struct MR_ALIGN16 NumiClothBagGPUParticle {
    // xyz current position, w inverse mass.
    mr_float4 positionAndInverseMass;
    // xyz position at the beginning of the substep, w authored mass.
    mr_float4 previousAndMass;
    // xyz velocity, w reserved.
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

typedef struct MR_ALIGN16 NumiClothBagGPUBatch {
    // x first constraint, y constraint count, z expected graph color,
    // w reserved.
    mr_uint4 control;
} NumiClothBagGPUBatch;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiClothBagGPUConfig) == 48);
static_assert(sizeof(NumiClothBagGPUParticle) == 48);
static_assert(sizeof(NumiClothBagGPUDistance) == 32);
static_assert(sizeof(NumiClothBagGPUGrip) == 48);
static_assert(sizeof(NumiClothBagGPUBatch) == 16);
#endif

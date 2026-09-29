#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_DEFORMABLE_TET_ABI_VERSION 1u

enum NumiDeformableTetFailure : mr_u32 {
    NUMI_DEFORMABLE_TET_FAILURE_ABI = 1u,
    NUMI_DEFORMABLE_TET_FAILURE_NONFINITE = 2u,
    NUMI_DEFORMABLE_TET_FAILURE_REST_GEOMETRY = 4u,
    NUMI_DEFORMABLE_TET_FAILURE_MATERIAL = 8u,
    NUMI_DEFORMABLE_TET_FAILURE_INVERSION = 16u,
};

typedef struct MR_ALIGN16 NumiDeformableTetInput {
    // Four current nodal positions in meters; w reserved.
    mr_float4 positions[4];
    // Rows of the inverse rest edge matrix [X1-X0, X2-X0, X3-X0].
    // Row0.w is positive reference volume in m^3; other w values reserved.
    mr_float4 inverseRestRows[3];
    // x shear modulus mu in Pa, y Lame lambda in Pa; zw reserved.
    mr_float4 material;
    // x ABI version; yzw reserved.
    mr_uint4 control;
} NumiDeformableTetInput;

typedef struct MR_ALIGN16 NumiDeformableTetOutput {
    // Elastic nodal forces in N, summing to zero; w reserved.
    mr_float4 forces[4];
    // First Piola stress rows in Pa; w reserved.
    mr_float4 firstPiolaRows[3];
    // x elastic energy in J, y det(F), zw reserved.
    mr_float4 energyAndVolumeRatio;
    // x failure bits (zero means valid); advance retains cumulative failure
    // bits in y across every step, including failures between captures.
    // Initialize advance output buffers to zero; zw reserved.
    mr_uint4 control;
} NumiDeformableTetOutput;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiDeformableTetInput) == 144);
static_assert(sizeof(NumiDeformableTetOutput) == 144);
#endif

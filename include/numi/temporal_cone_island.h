#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_TEMPORAL_CONE_ISLAND_ABI_VERSION 1u
#define NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS 32u
#define NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS \
    (3u * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS)
#define NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS \
    (NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS * \
     NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS)
#define NUMI_TEMPORAL_CONE_ISLAND_MAX_ITERATIONS 1024u

#define NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION 2u
#define NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS \
    (NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS * \
     NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS)
#define NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS 9u

enum NumiTemporalConeIslandStatusCode : mr_u32 {
    NUMI_TEMPORAL_CONE_ISLAND_SUCCESS = 0u,
    NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI = 1u,
    NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT = 2u,
    NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED = 3u,
    NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT = 4u,
    NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE = 5u,
};

typedef struct MR_ALIGN16 NumiTemporalConeIslandHeader {
    // x ABI, y contacts, z minimum iterations, w maximum iterations.
    mr_uint4 control;
    // x absolute tolerance, y relative tolerance, z relaxation, w reserved.
    mr_float4 tolerances;
} NumiTemporalConeIslandHeader;

// Packed block-CSR operator contract. Each row is one three-axis contact
// block. Row offsets are relative to ranges.z, source-contact indices are
// strictly increasing within a row, and every row contains its diagonal.
typedef struct MR_ALIGN16 NumiTemporalConeStreamHeader {
    // x ABI, y contacts, z minimum iterations, w maximum iterations.
    mr_uint4 control;
    // x contact/output base, y row-offset base, z block base, w block count.
    mr_uint4 ranges;
    // x absolute tolerance, y relative tolerance, z relaxation, w reserved.
    mr_float4 tolerances;
} NumiTemporalConeStreamHeader;

typedef struct MR_ALIGN16 NumiTemporalConeIslandContact {
    // xyz free contact velocity; w friction coefficient in tangent U.
    mr_float4 freeVelocityAndFrictionU;
    // xyz warm-start impulse; w friction coefficient in tangent V.
    mr_float4 warmImpulseAndFrictionV;
    // x maximum normal impulse (zero is unbounded); yzw reserved.
    mr_float4 limits;
} NumiTemporalConeIslandContact;

typedef struct MR_ALIGN16 NumiTemporalConeIslandStatus {
    // x status, y completed iterations, z converged, w contact count.
    mr_uint4 control;
    // x normalized KKT gradient-mapping residual, y cone violation,
    // z maximum impulse magnitude, w objective.
    mr_float4 residuals;
    // x maximum raw contact residual, y relaxation, zw reserved.
    mr_float4 diagnostics;
} NumiTemporalConeIslandStatus;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiTemporalConeIslandHeader) == 32);
static_assert(sizeof(NumiTemporalConeStreamHeader) == 48);
static_assert(sizeof(NumiTemporalConeIslandContact) == 48);
static_assert(sizeof(NumiTemporalConeIslandStatus) == 48);
#endif

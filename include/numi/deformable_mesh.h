#pragma once
#include "numi/deformable_tet.h"

#define NUMI_DEFORMABLE_MESH_ABI_VERSION 2u
#define NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY 32u
#define NUMI_DEFORMABLE_MESH_FAILURE_STATE 64u

typedef struct MR_ALIGN16 NumiDeformableMeshElement {
    mr_uint4 nodes;
    mr_float4 inverseRestRows[3];
    mr_float4 material;
    mr_uint4 control;
} NumiDeformableMeshElement;

typedef struct MR_ALIGN16 NumiDeformableMeshConfig {
    // x nodes, y elements, z incidence entries, w mesh ABI.
    mr_uint4 counts;
    // xyz acceleration in m/s^2; w timestep in seconds.
    mr_float4 gravityAndTimestep;
    // x plane height in m, y enabled (0 or 1), zw reserved.
    mr_float4 plane;
} NumiDeformableMeshConfig;

typedef struct MR_ALIGN16 NumiDeformableMeshStatus {
    // x current failure, y cumulative failure, z rejected steps, w accepted steps.
    mr_uint4 control;
    // x accumulated support normal impulse in Ns; y minimum accepted det(F);
    // z maximum net internal force in N; w removed normal kinetic energy in J.
    // The kinetic ledger alone is not total contact-work or energy closure.
    mr_float4 ledger;
    // Accepted plane projection: cumulative elastic potential change (x),
    // gravity potential change (y), total mechanical change (z), and largest
    // positive single-step mechanical change (w), in joules. Normal kinetic
    // removal remains ledger.w; z = x + y - ledger.w.
    mr_float4 contactWork;
    // Unconstrained integration: signed mechanical change (x), accumulated
    // absolute change (y), maximum absolute step change (z), and accumulated
    // previous elastic force dotted with free nodal displacement (w), in J.
    // This measures numerical integration error; it does not certify zero error.
    mr_float4 integration;
    // x sum of absolute elastic/gravity projection changes; y their maximum
    // step bound; z maximum absolute total contact step; w sum of positive
    // total contact changes. These nonnegative bounds retain cancellation.
    mr_float4 projectionBounds;
} NumiDeformableMeshStatus;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiDeformableMeshElement) == 96);
static_assert(sizeof(NumiDeformableMeshConfig) == 48);
static_assert(sizeof(NumiDeformableMeshStatus) == 80);
#endif

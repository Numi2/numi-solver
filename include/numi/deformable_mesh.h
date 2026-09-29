#pragma once
#include "numi/deformable_tet.h"

#define NUMI_DEFORMABLE_MESH_ABI_VERSION 3u
#define NUMI_DEFORMABLE_MESH_INTEGRATOR_EULER 0u
#define NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET 1u
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
    // x integration method (Euler=0, support-aware velocity Verlet=1), yzw zero.
    mr_uint4 integration;
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
    // positive single-step mechanical change (w), in joules.
    // z = x + y + contactKinetic.x; normal removal remains ledger.w.
    mr_float4 contactWork;
    // Pre-impact integration: signed mechanical change (x), accumulated
    // absolute change (y), maximum absolute step change (z), and accumulated
    // previous elastic force dotted with free nodal displacement (w), in J.
    // Euler uses unconstrained prediction. Support Verlet includes exact
    // zero-work normal reactions at resting plane nodes in this prediction.
    // This measures numerical integration error; it does not certify zero error.
    mr_float4 integration;
    // x sum of absolute elastic/gravity projection changes; y their maximum
    // step bound; z maximum absolute total contact step; w sum of positive
    // total contact changes. Bounds prevent signed totals hiding cancellation.
    mr_float4 projectionBounds;
    // x cumulative projected-minus-pre-impact kinetic energy; y cumulative
    // kinetic change beyond explicit incoming normal removal; z accumulated
    // absolute value of that force-response change; w maximum absolute step.
    mr_float4 contactKinetic;
} NumiDeformableMeshStatus;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiDeformableMeshElement) == 96);
static_assert(sizeof(NumiDeformableMeshConfig) == 64);
static_assert(sizeof(NumiDeformableMeshStatus) == 96);
#endif

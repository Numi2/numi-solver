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

#define NUMI_TEMPORAL_CONE_ASSEMBLY_ABI_VERSION 1u
#define NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_TERMS_PER_CONTACT 32u
#define NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_DOF_PER_TERM 32u

#define NUMI_TEMPORAL_CONE_RIGID_ABI_VERSION 1u
#define NUMI_TEMPORAL_CONE_RIGID_MAX_BODIES 32u
#define NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY 0xffffffffu
#define NUMI_TEMPORAL_CONE_RIGID_DOF 6u
#define NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM \
    (3u * NUMI_TEMPORAL_CONE_RIGID_DOF)

#define NUMI_TEMPORAL_CONE_INTEGRATION_ABI_VERSION 1u

enum NumiTemporalConeIslandStatusCode : mr_u32 {
    NUMI_TEMPORAL_CONE_ISLAND_SUCCESS = 0u,
    NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI = 1u,
    NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT = 2u,
    NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED = 3u,
    NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT = 4u,
    NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE = 5u,
};

enum NumiTemporalConeAssemblyStatusCode : mr_u32 {
    NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS = 0u,
    NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_ABI = 1u,
    NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT = 2u,
    NUMI_TEMPORAL_CONE_ASSEMBLY_CAPACITY_EXCEEDED = 3u,
    NUMI_TEMPORAL_CONE_ASSEMBLY_NONFINITE_RESULT = 4u,
    NUMI_TEMPORAL_CONE_ASSEMBLY_MISSING_COUPLING = 5u,
    NUMI_TEMPORAL_CONE_ASSEMBLY_ASYMMETRIC_RESPONSE = 6u,
};

enum NumiTemporalConeRigidStatusCode : mr_u32 {
    NUMI_TEMPORAL_CONE_RIGID_SUCCESS = 0u,
    NUMI_TEMPORAL_CONE_RIGID_INVALID_ABI = 1u,
    NUMI_TEMPORAL_CONE_RIGID_INVALID_INPUT = 2u,
    NUMI_TEMPORAL_CONE_RIGID_NONFINITE_RESULT = 3u,
    NUMI_TEMPORAL_CONE_RIGID_UPSTREAM_FAILURE = 4u,
};

enum NumiTemporalConeIntegrationStatusCode : mr_u32 {
    NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS = 0u,
    NUMI_TEMPORAL_CONE_INTEGRATION_INVALID_ABI = 1u,
    NUMI_TEMPORAL_CONE_INTEGRATION_INVALID_INPUT = 2u,
    NUMI_TEMPORAL_CONE_INTEGRATION_NONFINITE_RESULT = 3u,
    NUMI_TEMPORAL_CONE_INTEGRATION_UPSTREAM_FAILURE = 4u,
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

// Builds the streamed operator from packed contact-owner Jacobians and
// response columns. Terms within a contact are strictly owner-sorted.
typedef struct MR_ALIGN16 NumiTemporalConeAssemblyHeader {
    // x ABI, y contacts, z solver minimum iterations, w solver maximum.
    mr_uint4 control;
    // x solver contact/output base, y row-offset base,
    // z block base, w block count.
    mr_uint4 outputRanges;
    // x contact-span base, y regularization-value base, zw reserved.
    mr_uint4 inputRanges;
    // Forwarded to the streamed solver on successful assembly.
    mr_float4 tolerances;
} NumiTemporalConeAssemblyHeader;

typedef struct MR_ALIGN16 NumiTemporalConeAssemblyContactSpan {
    // x term base, y term count, zw reserved.
    mr_uint4 ranges;
} NumiTemporalConeAssemblyContactSpan;

typedef struct MR_ALIGN16 NumiTemporalConeAssemblyTerm {
    // x stable owner, y DOFs, z Jacobian-value base, w response-value base.
    // Jacobian is 3 x DOFs; response is DOFs x 3, both row-major.
    mr_uint4 control;
} NumiTemporalConeAssemblyTerm;

typedef struct MR_ALIGN16 NumiTemporalConeAssemblyStatus {
    // x status, y contacts, z blocks, w maximum terms per contact.
    mr_uint4 control;
    // x maximum symmetry error, y maximum absolute coefficient,
    // z minimum diagonal coefficient, w missing/extraneous topology count.
    mr_float4 diagnostics;
} NumiTemporalConeAssemblyStatus;

// Generates rigid-body contact Jacobians and M^-1 J^T response columns.
// Body indices in contacts are local to inputRanges.x. The response owner is
// the corresponding global body index, so shared-body coupling is explicit.
typedef struct MR_ALIGN16 NumiTemporalConeRigidHeader {
    // x ABI, y dynamic bodies, z contacts, w reserved.
    mr_uint4 control;
    // x input body base, y rigid-contact base, zw reserved.
    mr_uint4 inputRanges;
    // x span base, y term base, z Jacobian base, w response base.
    mr_uint4 responseRanges;
    // x solver-contact base, y output-body base, zw reserved.
    mr_uint4 solverRanges;
} NumiTemporalConeRigidHeader;

typedef struct MR_ALIGN16 NumiTemporalConeRigidBody {
    // xyz linear velocity; w inverse mass.
    mr_float4 linearVelocityAndInverseMass;
    // xyz angular velocity; w reserved.
    mr_float4 angularVelocity;
    // Symmetric positive-definite world-space inverse inertia tensor.
    mr_float4 inverseInertiaRow0;
    mr_float4 inverseInertiaRow1;
    mr_float4 inverseInertiaRow2;
} NumiTemporalConeRigidBody;

typedef struct MR_ALIGN16 NumiTemporalConeRigidContact {
    // x body A, y body B; UINT_MAX denotes the static world.
    mr_uint4 bodies;
    // Contact-point offsets from each body's center of mass.
    mr_float4 offsetA;
    mr_float4 offsetB;
    // xyz right-handed orthonormal contact frame; w cone parameter.
    mr_float4 normalAndFrictionU;
    mr_float4 tangentUAndFrictionV;
    mr_float4 tangentVAndMaximumNormal;
    // xyz additive free contact velocity in (normal, tangent U, tangent V).
    mr_float4 bias;
    // xyz warm-start impulse in the same frame; w reserved.
    mr_float4 warmImpulse;
} NumiTemporalConeRigidContact;

typedef struct MR_ALIGN16 NumiTemporalConeRigidStatus {
    // x status, y bodies, z contacts, w generated owner terms.
    mr_uint4 control;
    // x max frame error, y minimum inverse mass,
    // z minimum inertia principal-minor proxy, w maximum speed delta.
    mr_float4 diagnostics;
} NumiTemporalConeRigidStatus;

typedef struct MR_ALIGN16 NumiTemporalConeIntegrationHeader {
    // x ABI, y bodies, zw reserved.
    mr_uint4 control;
    // x input-pose base, y velocity-body base, z output-pose base, w reserved.
    mr_uint4 ranges;
    // x timestep; yzw reserved.
    mr_float4 timestep;
} NumiTemporalConeIntegrationHeader;

typedef struct MR_ALIGN16 NumiTemporalConeRigidPose {
    // xyz center-of-mass position; w reserved and preserved.
    mr_float4 position;
    // Unit body-to-world quaternion (x, y, z, w).
    mr_float4 orientation;
} NumiTemporalConeRigidPose;

typedef struct MR_ALIGN16 NumiTemporalConeIntegrationStatus {
    // x status, y bodies, zw reserved.
    mr_uint4 control;
    // x max input norm error, y max output norm error,
    // z max linear displacement, w max angular step.
    mr_float4 diagnostics;
} NumiTemporalConeIntegrationStatus;

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
static_assert(sizeof(NumiTemporalConeAssemblyHeader) == 64);
static_assert(sizeof(NumiTemporalConeAssemblyContactSpan) == 16);
static_assert(sizeof(NumiTemporalConeAssemblyTerm) == 16);
static_assert(sizeof(NumiTemporalConeAssemblyStatus) == 32);
static_assert(sizeof(NumiTemporalConeRigidHeader) == 64);
static_assert(sizeof(NumiTemporalConeRigidBody) == 80);
static_assert(sizeof(NumiTemporalConeRigidContact) == 128);
static_assert(sizeof(NumiTemporalConeRigidStatus) == 32);
static_assert(sizeof(NumiTemporalConeIntegrationHeader) == 48);
static_assert(sizeof(NumiTemporalConeRigidPose) == 32);
static_assert(sizeof(NumiTemporalConeIntegrationStatus) == 32);
static_assert(sizeof(NumiTemporalConeIslandContact) == 48);
static_assert(sizeof(NumiTemporalConeIslandStatus) == 48);
#endif

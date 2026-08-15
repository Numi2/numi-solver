#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_BRAIDED_BAG_ABI_VERSION 3u
#define NUMI_BRAIDED_BAG_RING_SIZE 8u
#define NUMI_BRAIDED_BAG_LEVEL_COUNT 7u
#define NUMI_BRAIDED_BAG_NODE_COUNT \
    (NUMI_BRAIDED_BAG_RING_SIZE * NUMI_BRAIDED_BAG_LEVEL_COUNT)
#define NUMI_BRAIDED_BAG_EDGE_COUNT \
    (2u * NUMI_BRAIDED_BAG_RING_SIZE * \
     (NUMI_BRAIDED_BAG_LEVEL_COUNT - 1u) + \
     NUMI_BRAIDED_BAG_RING_SIZE / 2u)
#define NUMI_BRAIDED_BAG_BALL_COUNT 6u
#define NUMI_BRAIDED_BAG_BALL_PAIR_COUNT \
    ((NUMI_BRAIDED_BAG_BALL_COUNT * \
      (NUMI_BRAIDED_BAG_BALL_COUNT - 1u)) / 2u)
#define NUMI_BRAIDED_BAG_CONTACT_COUNT \
    (2u * NUMI_BRAIDED_BAG_BALL_COUNT + \
     NUMI_BRAIDED_BAG_BALL_PAIR_COUNT)
// Exact fixed structural superset: all braid contacts may share strand
// nodes; every other off-diagonal block exists only when two slots share a
// ball. For six balls this is 309 blocks rather than the dense 729.
#define NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT 309u
#define NUMI_BRAIDED_BAG_PARTICLE_COUNT \
    (NUMI_BRAIDED_BAG_NODE_COUNT + NUMI_BRAIDED_BAG_BALL_COUNT)
#define NUMI_BRAIDED_BAG_INVALID_PARTICLE 0xffffffffu

enum NumiBraidedBagPath : mr_u32 {
    NUMI_BRAIDED_BAG_PATH_DENSE = 0u,
    NUMI_BRAIDED_BAG_PATH_STREAMED = 1u,
};

typedef struct MR_ALIGN16 NumiBraidedBagConfig {
    // x ABI, y environments, z minimum solver iterations,
    // w maximum solver iterations.
    mr_uint4 control;
    // x timestep, y gravity z, z node drag, w ball drag.
    mr_float4 timing;
    // x stretch stiffness, y axial damping, z strand radius, w friction.
    mr_float4 braidMaterial;
    // x penetration stabilization, y contact CFM, z woven-base height,
    // w maximum depenetration speed.
    mr_float4 contact;
    // x absolute KKT tolerance, y relative KKT tolerance,
    // z relaxation, w reserved.
    mr_float4 solver;
    // x authored bag radius, y bottom height, z mouth height,
    // w escape padding.
    mr_float4 bounds;
} NumiBraidedBagConfig;

typedef struct MR_ALIGN16 NumiBraidedBagNode {
    // xyz position, w inverse mass. Anchored mouth nodes use zero inverse mass.
    mr_float4 positionAndInverseMass;
    // xyz velocity, w reserved.
    mr_float4 velocity;
} NumiBraidedBagNode;

typedef struct MR_ALIGN16 NumiBraidedBagBall {
    // xyz center, w inverse mass.
    mr_float4 positionAndInverseMass;
    // xyz linear velocity, w radius.
    mr_float4 velocityAndRadius;
} NumiBraidedBagBall;

typedef struct MR_ALIGN16 NumiBraidedBagEdge {
    // x first node, y second node, zw reserved.
    mr_uint4 nodes;
    // x rest length, yzw reserved.
    mr_float4 rest;
} NumiBraidedBagEdge;

typedef struct MR_ALIGN16 NumiBraidedBagContact {
    // Up to three unified particles: braid nodes [0,56), balls [56,62).
    mr_uint4 particles;
    // Signed Jacobian weights corresponding to xyz particles; w count.
    mr_float4 weights;
    // xyz normal, w signed surface separation.
    mr_float4 normalAndSeparation;
    // xyz deterministic tangent u, w reserved.
    mr_float4 tangentU;
    // xyz deterministic tangent v, w reserved.
    mr_float4 tangentV;
    // x contact class, y stable feature, zw reserved.
    mr_uint4 identity;
} NumiBraidedBagContact;

typedef struct MR_ALIGN16 NumiBraidedBagStatus {
    // x failed transactional steps, y completed steps,
    // z maximum solver iterations, w escaped-ball mask.
    mr_uint4 control;
    // x maximum penetration, y maximum relative strand stretch,
    // z maximum normalized KKT residual, w maximum cone violation.
    mr_float4 solverMetrics;
    // x maximum ball radial center distance, y minimum ball-center height,
    // z maximum ball speed, w maximum braid-node speed.
    mr_float4 physicalMetrics;
    // x maximum normalized cone complementarity/VI residual,
    // y maximum positive contact objective,
    // z maximum Delassus infinity-norm row sum,
    // w maximum rigorous condition-number upper bound from CFM.
    mr_float4 certificateMetrics;
    // x minimum active contacts, y maximum active contacts,
    // z accumulated active contacts, w maximum active CSR blocks.
    mr_uint4 topologyMetrics;
} NumiBraidedBagStatus;

#ifndef __METAL_VERSION__
static_assert(sizeof(NumiBraidedBagConfig) == 96);
static_assert(sizeof(NumiBraidedBagNode) == 32);
static_assert(sizeof(NumiBraidedBagBall) == 32);
static_assert(sizeof(NumiBraidedBagEdge) == 32);
static_assert(sizeof(NumiBraidedBagContact) == 96);
static_assert(sizeof(NumiBraidedBagStatus) == 80);
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/engine_types.h"
#include "numi/temporal_cone_island.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

#ifndef NUMI_ARTICULATED_CHAIN_DOF
#define NUMI_ARTICULATED_CHAIN_DOF 2u
#endif
#ifndef NUMI_ARTICULATED_ARMATURE_SCALE
#define NUMI_ARTICULATED_ARMATURE_SCALE 1.0
#endif
#ifndef NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE
#define NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE 0
#endif

constexpr std::uint32_t kDofs = NUMI_ARTICULATED_CHAIN_DOF;
constexpr std::uint32_t kContacts = kDofs;
constexpr std::uint32_t kBodies = kDofs + 1u;
static_assert(kDofs > 0u);
static_assert(kDofs <= NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF);
constexpr double kGeometryScale = kDofs <= 8u
    ? 1.0
    : 8.0 / static_cast<double>(kDofs);
constexpr double kArmatureScale = NUMI_ARTICULATED_ARMATURE_SCALE;
constexpr bool kExpectConditionFailure =
    NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE != 0;

double linkHalfLength(std::size_t link) {
    return kGeometryScale * (
        0.24 + 0.015 * static_cast<double>(link % 5u)
    );
}

double linkMass(std::size_t link) {
    return 1.0 + 0.08 * static_cast<double>(link % 7u);
}

double linkInertiaY(std::size_t link) {
    const double length = 2.0 * linkHalfLength(link);
    return linkMass(link) * length * length / 12.0;
}

double linkArmature(std::size_t link) {
    return kArmatureScale * (
        0.02 + 0.005 * static_cast<double>(link % 3u)
    );
}

mr_float4 f4(float x, float y, float z, float w = 0.0f) {
    return {x, y, z, w};
}

mr_uint4 u4(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t w
) {
    return {x, y, z, w};
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRBodyPropertiesGPU body(
    std::uint32_t parent,
    std::uint32_t inbound,
    float mass,
    float ix,
    float iy,
    float iz
) {
    MRBodyPropertiesGPU value = {};
    value.articulationIndex = 0u;
    value.parentBody = parent;
    value.inboundJoint = inbound;
    value.motionType = MR_MOTION_DYNAMIC;
    value.massAndInverseMass = f4(mass, 1.0f / mass, 0.0f);
    value.inertiaRow0 = f4(ix, 0.0f, 0.0f);
    value.inertiaRow1 = f4(0.0f, iy, 0.0f);
    value.inertiaRow2 = f4(0.0f, 0.0f, iz);
    value.inverseInertiaRow0 = f4(1.0f / ix, 0.0f, 0.0f);
    value.inverseInertiaRow1 = f4(0.0f, 1.0f / iy, 0.0f);
    value.inverseInertiaRow2 = f4(0.0f, 0.0f, 1.0f / iz);
    value.dampingAndSpeedLimits = f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);
    return value;
}

struct Model {
    MRWorldGPU world = {};
    MRArticulationGPU articulation = {};
    std::vector<MRJointDescriptorGPU> joints;
    std::vector<MRDofPropertiesGPU> dofs;
    std::vector<MRBodyPropertiesGPU> bodies;
};

Model makeModel() {
    Model model;
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = kBodies;
    model.world.articulationCount = 1u;
    model.world.jointCount = kDofs;
    model.world.nq = kDofs;
    model.world.nv = kDofs;
    model.world.pairCapacity = kContacts;
    model.world.contactCapacity = kContacts;
    model.world.constraintCapacity = kContacts;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_TEMPORAL_CONE;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0f, 0.0f, -9.81f, 1.0f / 240.0f);
    model.world.solverScales = f4(1.0e-6f, 1.0e-7f, 2.0f, 1.0e-4f);

    model.articulation.rootBody = 0u;
    model.articulation.rootType = MR_ROOT_FIXED;
    model.articulation.firstBody = 0u;
    model.articulation.bodyCount = kBodies;
    model.articulation.firstJoint = 0u;
    model.articulation.jointCount = kDofs;
    model.articulation.nq = kDofs;
    model.articulation.nv = kDofs;

    model.joints.resize(kDofs);
    model.dofs.resize(kDofs);
    model.bodies.resize(kBodies);
    model.bodies[0] = body(MR_INVALID_INDEX, MR_INVALID_INDEX, 1.0f,
        0.2f, 0.25f, 0.3f);
    for (std::uint32_t jointIndex = 0u; jointIndex < kDofs; ++jointIndex) {
        const double inertiaY = linkInertiaY(jointIndex);
        model.bodies[jointIndex + 1u] = body(
            jointIndex,
            jointIndex,
            static_cast<float>(linkMass(jointIndex)),
            static_cast<float>(0.85 * inertiaY),
            static_cast<float>(inertiaY),
            static_cast<float>(1.15 * inertiaY)
        );
        auto& joint = model.joints[jointIndex];
        joint.parentBody = jointIndex;
        joint.childBody = jointIndex + 1u;
        joint.jointType = MR_JOINT_REVOLUTE;
        joint.qOffset = jointIndex;
        joint.nq = 1u;
        joint.vOffset = jointIndex;
        joint.nv = 1u;
        joint.axis0 = f4(0.0f, 1.0f, 0.0f);
        joint.parentAnchor = jointIndex == 0u
            ? f4(0.0f, 0.0f, 0.0f)
            : f4(
                0.0f,
                0.0f,
                static_cast<float>(linkHalfLength(jointIndex - 1u))
            );
        joint.childAnchor = f4(
            0.0f,
            0.0f,
            static_cast<float>(-linkHalfLength(jointIndex))
        );
        joint.parentRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        joint.childRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        auto& dof = model.dofs[jointIndex];
        dof.articulationIndex = 0u;
        dof.jointIndex = jointIndex;
        dof.qIndex = jointIndex;
        dof.vIndex = jointIndex;
        dof.localDof = 0u;
        dof.drive = f4(
            0.0f,
            0.0f,
            static_cast<float>(linkArmature(jointIndex)),
            0.0f
        );
    }
    return model;
}

NumiTemporalConeRigidLaw law(float gap, float restitution) {
    constexpr float timestep = 1.0f / 240.0f;
    NumiTemporalConeRigidLaw value = {};
    value.stiffnessAndRestitution = f4(18000.0f, 900.0f, 900.0f, restitution);
    value.dampingAndImpactThreshold = f4(260.0f, 40.0f, 40.0f, 0.05f);
    value.stabilization = f4(gap, 0.0005f, 1.5f, timestep);
    return value;
}

NumiTemporalConeArticulatedContact contact(std::uint32_t point) {
    NumiTemporalConeArticulatedContact value = {};
    value.control = u4(point, 0u, 0u, 0u);
    value.normalAndFrictionU = f4(0.0f, 0.0f, 1.0f, 0.65f);
    value.tangentUAndFrictionV = f4(1.0f, 0.0f, 0.0f, 0.45f);
    value.tangentVAndMaximumNormal = f4(0.0f, 1.0f, 0.0f, 0.0f);
    return value;
}

struct Batch {
    Model model;
    MRArticulatedOperatorDispatchGPU operatorDispatch = {};
    std::vector<float> q;
    std::vector<float> velocities;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<NumiTemporalConeArticulatedHeader> articulatedHeaders;
    std::vector<NumiTemporalConeAssemblyHeader> assemblyHeaders;
    std::vector<NumiTemporalConeArticulatedContact> contacts;
    std::vector<NumiTemporalConeRigidLaw> laws;
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columns;
    std::size_t validProblems = 0u;
};

Batch makeBatch(std::size_t problemCount) {
    Batch batch;
    batch.model = makeModel();
    batch.validProblems = problemCount - 3u;
    batch.operatorDispatch.articulationIndex = 0u;
    batch.operatorDispatch.environmentCount = static_cast<mr_u32>(problemCount);
    batch.operatorDispatch.pointCount = kContacts;
    batch.operatorDispatch.flags = MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR;
    batch.operatorDispatch.qStride = kDofs;
    batch.operatorDispatch.pointStride = kContacts;
    batch.operatorDispatch.bodyPoseStride = kBodies;
    batch.operatorDispatch.pointWorldStride = kContacts;
    batch.operatorDispatch.massMatrixStride = kDofs * kDofs;
    batch.operatorDispatch.pointJacobianStride = kContacts * 3u * kDofs;
    batch.operatorDispatch.generalizedStride = kDofs;
    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        for (std::uint32_t dof = 0u; dof < kDofs; ++dof) {
            const double phase =
                0.173 * static_cast<double>(problem) +
                0.619 * static_cast<double>(dof);
            batch.q.push_back(static_cast<float>(0.32 * std::sin(phase)));
            batch.velocities.push_back(static_cast<float>(
                0.45 * std::cos(0.71 * phase) +
                0.08 * std::sin(0.37 * phase)
            ));
        }
        if (problem + 3u == problemCount) {
            batch.q[kDofs * problem] =
                std::numeric_limits<float>::quiet_NaN();
        }
        for (std::uint32_t pointIndex = 0u;
             pointIndex < kContacts;
             ++pointIndex) {
            MRArticulatedPointImpulseGPU query = {};
            query.bodyIndex = pointIndex + 1u;
            query.localPoint = f4(
                0.0f,
                0.0f,
                static_cast<float>(linkHalfLength(pointIndex))
            );
            batch.points.push_back(query);
            batch.contacts.push_back(contact(pointIndex));
            batch.laws.push_back(law(
                -0.0015f - 0.00002f * static_cast<float>(pointIndex % 11u),
                0.15f + 0.02f * static_cast<float>(pointIndex % 5u)
            ));
        }
        if (problem + 2u == problemCount) {
            batch.contacts[kContacts * problem].tangentVAndMaximumNormal =
                f4(1.0f, 0.0f, 0.0f);
        } else if (problem + 1u == problemCount) {
            batch.laws[kContacts * problem].stiffnessAndRestitution.w = 1.5f;
        }

        NumiTemporalConeArticulatedHeader articulated = {};
        articulated.control = u4(
            NUMI_TEMPORAL_CONE_ARTICULATED_ABI_VERSION,
            kDofs,
            kContacts,
            static_cast<std::uint32_t>(1000u + problem)
        );
        articulated.inputRanges = u4(
            static_cast<std::uint32_t>(problem * kDofs * kDofs),
            static_cast<std::uint32_t>(problem * kContacts * 3u * kDofs),
            static_cast<std::uint32_t>(problem * kDofs),
            static_cast<std::uint32_t>(problem * kContacts)
        );
        articulated.responseRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts * 3u * kDofs),
            static_cast<std::uint32_t>(problem * kContacts * 3u * kDofs)
        );
        articulated.solverRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kDofs),
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts * 9u)
        );
        articulated.operatorRanges = u4(static_cast<std::uint32_t>(problem), 0u, 0u, 0u);
        batch.articulatedHeaders.push_back(articulated);

        const std::uint32_t rowBase = static_cast<std::uint32_t>(batch.rowOffsets.size());
        const std::uint32_t blockBase = static_cast<std::uint32_t>(batch.columns.size());
        for (std::uint32_t row = 0u; row <= kContacts; ++row) {
            batch.rowOffsets.push_back(row * kContacts);
        }
        for (std::uint32_t row = 0u; row < kContacts; ++row) {
            for (std::uint32_t column = 0u; column < kContacts; ++column) {
                batch.columns.push_back(column);
            }
        }
        const std::uint32_t blockCount = kContacts * kContacts;
        NumiTemporalConeAssemblyHeader assembly = {};
        assembly.control = u4(
            NUMI_TEMPORAL_CONE_ASSEMBLY_ABI_VERSION,
            kContacts,
            4u,
            NUMI_TEMPORAL_CONE_ISLAND_MAX_ITERATIONS
        );
        assembly.outputRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            rowBase,
            blockBase,
            blockCount
        );
        assembly.inputRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts * 9u),
            0u,
            0u
        );
        assembly.tolerances = f4(2.0e-6f, 2.0e-6f, 1.0f, 0.0f);
        batch.assemblyHeaders.push_back(assembly);
    }
    return batch;
}

struct Result {
    std::vector<MRArticulatedBodyPoseGPU> poses;
    std::vector<MRArticulatedPointWorldGPU> pointWorld;
    std::vector<float> factors;
    std::vector<float> pointJacobians;
    std::vector<MRArticulatedOperatorStatusGPU> operatorStatuses;
    std::vector<float> kinematicPointJacobians;
    std::vector<MRArticulatedOperatorStatusGPU> kinematicStatuses;
    std::vector<float> preparedContactJacobians;
    std::vector<NumiTemporalConeArticulatedStatus> preparationStatuses;
    std::vector<NumiTemporalConeAssemblyContactSpan> spans;
    std::vector<NumiTemporalConeAssemblyTerm> terms;
    std::vector<float> jacobians;
    std::vector<float> responses;
    // Inverse-ABA output is packed [problem][contact axis][DoF].
    std::vector<float> inverseResponses;
    std::vector<MRInverseMassStatusGPU> inverseStatuses;
    std::vector<NumiTemporalConeIslandContact> solverContacts;
    std::vector<float> regularization;
    std::vector<NumiTemporalConeArticulatedStatus> responseStatuses;
    std::vector<NumiTemporalConeStreamHeader> streamHeaders;
    std::vector<float> blocks;
    std::vector<NumiTemporalConeAssemblyStatus> assemblyStatuses;
    std::vector<mr_float4> impulses;
    std::vector<NumiTemporalConeIslandStatus> solverStatuses;
    std::vector<float> outputVelocities;
    std::vector<NumiTemporalConeArticulatedStatus> publishStatuses;
    double seconds = 0.0;
    std::array<double, 5> stageSeconds = {};
    double kinematicsSeconds = 0.0;
    double preparationSeconds = 0.0;
    double inverseSeconds = 0.0;
};

std::size_t operatorThreadgroupBytes(std::size_t bodies, std::size_t dofs) {
    const auto aligned = [](std::size_t value) { return (value + 15u) & ~std::size_t{15u}; };
    std::size_t bytes = 0u;
    const auto append = [&](std::size_t value) { bytes = aligned(bytes); bytes += value; };
    append(16u * bodies);
    append(16u * bodies);
    append(16u * bodies);
    append(16u * bodies);
    append(sizeof(std::uint32_t) * bodies);
    append(sizeof(std::uint32_t) * bodies);
    append(sizeof(std::uint8_t) * bodies);
    append(sizeof(float) * dofs * dofs);
    append(sizeof(float) * dofs);
    append(sizeof(float) * dofs);
    append(sizeof(float) * dofs);
    return aligned(bytes);
}

template <typename T>
bool exactVector(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

Result runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const std::array<id<MTLComputePipelineState>, 7>& pipelines,
    const Batch& batch
) {
    const std::uint32_t problemCount =
        static_cast<std::uint32_t>(batch.articulatedHeaders.size());
    const auto makeBytes = [&](const auto& values) -> id<MTLBuffer> {
        return [device newBufferWithBytes:values.data()
                                  length:values.size() * sizeof(values.front())
                                 options:MTLResourceStorageModeShared];
    };
    const auto makeOne = [&](const auto& value) -> id<MTLBuffer> {
        return [device newBufferWithBytes:&value
                                  length:sizeof(value)
                                 options:MTLResourceStorageModeShared];
    };
    const auto output = [&](std::size_t bytes) -> id<MTLBuffer> {
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        if (buffer != nil) std::memset(buffer.contents, 0, buffer.length);
        return buffer;
    };
    const auto& model = batch.model;
    id<MTLBuffer> worldBuffer = makeOne(model.world);
    id<MTLBuffer> articulationBuffer = makeOne(model.articulation);
    id<MTLBuffer> jointBuffer = makeBytes(model.joints);
    id<MTLBuffer> dofBuffer = makeBytes(model.dofs);
    id<MTLBuffer> bodyBuffer = makeBytes(model.bodies);
    id<MTLBuffer> operatorDispatchBuffer = makeOne(batch.operatorDispatch);
    id<MTLBuffer> qBuffer = makeBytes(batch.q);
    id<MTLBuffer> pointBuffer = makeBytes(batch.points);
    id<MTLBuffer> poseBuffer = output(problemCount * kBodies * sizeof(MRArticulatedBodyPoseGPU));
    id<MTLBuffer> pointWorldBuffer = output(problemCount * kContacts * sizeof(MRArticulatedPointWorldGPU));
    id<MTLBuffer> factorBuffer = output(problemCount * kDofs * kDofs * sizeof(float));
    id<MTLBuffer> pointJacobianBuffer = output(problemCount * kContacts * 3u * kDofs * sizeof(float));
    id<MTLBuffer> generalizedBuffer = output(problemCount * kDofs * sizeof(float));
    id<MTLBuffer> operatorDeltaBuffer = output(problemCount * kDofs * sizeof(float));
    id<MTLBuffer> operatorStatusBuffer = output(problemCount * sizeof(MRArticulatedOperatorStatusGPU));
    MRArticulatedOperatorDispatchGPU kinematicDispatch = batch.operatorDispatch;
    kinematicDispatch.flags =
        MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY;
    id<MTLBuffer> kinematicDispatchBuffer = makeOne(kinematicDispatch);
    id<MTLBuffer> kinematicPoseBuffer = output(
        problemCount * kBodies * sizeof(MRArticulatedBodyPoseGPU)
    );
    id<MTLBuffer> kinematicPointWorldBuffer = output(
        problemCount * kContacts * sizeof(MRArticulatedPointWorldGPU)
    );
    id<MTLBuffer> kinematicMassBuffer = output(
        problemCount * kDofs * kDofs * sizeof(float)
    );
    id<MTLBuffer> kinematicPointJacobianBuffer = output(
        problemCount * kContacts * 3u * kDofs * sizeof(float)
    );
    id<MTLBuffer> kinematicGeneralizedBuffer = output(
        problemCount * kDofs * sizeof(float)
    );
    id<MTLBuffer> kinematicDeltaBuffer = output(
        problemCount * kDofs * sizeof(float)
    );
    id<MTLBuffer> kinematicStatusBuffer = output(
        problemCount * sizeof(MRArticulatedOperatorStatusGPU)
    );
    id<MTLBuffer> articulatedHeaderBuffer = makeBytes(batch.articulatedHeaders);
    id<MTLBuffer> assemblyHeaderBuffer = makeBytes(batch.assemblyHeaders);
    id<MTLBuffer> velocityBuffer = makeBytes(batch.velocities);
    id<MTLBuffer> contactBuffer = makeBytes(batch.contacts);
    id<MTLBuffer> lawBuffer = makeBytes(batch.laws);
    id<MTLBuffer> spanBuffer = output(problemCount * kContacts * sizeof(NumiTemporalConeAssemblyContactSpan));
    id<MTLBuffer> termBuffer = output(problemCount * kContacts * sizeof(NumiTemporalConeAssemblyTerm));
    id<MTLBuffer> jacobianBuffer = output(problemCount * kContacts * 3u * kDofs * sizeof(float));
    id<MTLBuffer> preparedJacobianBuffer = output(
        problemCount * kContacts * 3u * kDofs * sizeof(float)
    );
    id<MTLBuffer> preparationStatusBuffer = output(
        problemCount * sizeof(NumiTemporalConeArticulatedStatus)
    );
    id<MTLBuffer> responseBuffer = output(problemCount * kContacts * 3u * kDofs * sizeof(float));
    id<MTLBuffer> inverseResponseBuffer = output(
        problemCount * kContacts * 3u * kDofs * sizeof(float)
    );
    id<MTLBuffer> inverseStatusBuffer = output(
        problemCount * sizeof(MRInverseMassStatusGPU)
    );
    id<MTLBuffer> inverseContactStatusBuffer = output(
        problemCount * sizeof(MRMetalWorldContactStatusGPU)
    );
    id<MTLBuffer> solverContactBuffer = output(problemCount * kContacts * sizeof(NumiTemporalConeIslandContact));
    id<MTLBuffer> regularizationBuffer = output(problemCount * kContacts * 9u * sizeof(float));
    id<MTLBuffer> responseStatusBuffer = output(problemCount * sizeof(NumiTemporalConeArticulatedStatus));
    id<MTLBuffer> rowBuffer = makeBytes(batch.rowOffsets);
    id<MTLBuffer> columnBuffer = makeBytes(batch.columns);
    id<MTLBuffer> blockBuffer = output(batch.columns.size() * 9u * sizeof(float));
    id<MTLBuffer> streamHeaderBuffer = output(problemCount * sizeof(NumiTemporalConeStreamHeader));
    id<MTLBuffer> assemblyStatusBuffer = output(problemCount * sizeof(NumiTemporalConeAssemblyStatus));
    id<MTLBuffer> impulseBuffer = output(problemCount * kContacts * sizeof(mr_float4));
    id<MTLBuffer> solverStatusBuffer = output(problemCount * sizeof(NumiTemporalConeIslandStatus));
    id<MTLBuffer> outputVelocityBuffer = output(problemCount * kDofs * sizeof(float));
    id<MTLBuffer> publishStatusBuffer = output(problemCount * sizeof(NumiTemporalConeArticulatedStatus));
    require(worldBuffer && articulationBuffer && jointBuffer && dofBuffer &&
        bodyBuffer && operatorDispatchBuffer && qBuffer && pointBuffer &&
        poseBuffer && pointWorldBuffer && factorBuffer && pointJacobianBuffer &&
        generalizedBuffer && operatorDeltaBuffer && operatorStatusBuffer &&
        kinematicDispatchBuffer && kinematicPoseBuffer &&
        kinematicPointWorldBuffer && kinematicMassBuffer &&
        kinematicPointJacobianBuffer && kinematicGeneralizedBuffer &&
        kinematicDeltaBuffer && kinematicStatusBuffer &&
        articulatedHeaderBuffer && assemblyHeaderBuffer && velocityBuffer &&
        contactBuffer && lawBuffer && spanBuffer && termBuffer && jacobianBuffer &&
        preparedJacobianBuffer && preparationStatusBuffer &&
        responseBuffer && inverseResponseBuffer && inverseStatusBuffer &&
        inverseContactStatusBuffer && solverContactBuffer && regularizationBuffer &&
        responseStatusBuffer && rowBuffer && columnBuffer && blockBuffer &&
        streamHeaderBuffer && assemblyStatusBuffer && impulseBuffer &&
        solverStatusBuffer && outputVelocityBuffer && publishStatusBuffer,
        "failed to allocate articulated qualification buffers");

    const auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> op = [command computeCommandEncoder];
    [op setComputePipelineState:pipelines[0]];
    const std::array<id<MTLBuffer>, 15> operatorBuffers = {
        worldBuffer, articulationBuffer, jointBuffer, dofBuffer, bodyBuffer,
        operatorDispatchBuffer, qBuffer, pointBuffer, poseBuffer,
        pointWorldBuffer, factorBuffer, pointJacobianBuffer,
        generalizedBuffer, operatorDeltaBuffer, operatorStatusBuffer
    };
    const std::array<id<MTLBuffer>, 15> kinematicBuffers = {
        worldBuffer, articulationBuffer, jointBuffer, dofBuffer, bodyBuffer,
        kinematicDispatchBuffer, qBuffer, pointBuffer, kinematicPoseBuffer,
        kinematicPointWorldBuffer, kinematicMassBuffer,
        kinematicPointJacobianBuffer, kinematicGeneralizedBuffer,
        kinematicDeltaBuffer, kinematicStatusBuffer
    };
    for (std::size_t index = 0u; index < operatorBuffers.size(); ++index) {
        [op setBuffer:operatorBuffers[index] offset:0 atIndex:index];
    }
    [op setThreadgroupMemoryLength:operatorThreadgroupBytes(kBodies, kDofs) atIndex:0];
    [op dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [op endEncoding];

    id<MTLComputeCommandEncoder> response = [command computeCommandEncoder];
    [response setComputePipelineState:pipelines[1]];
    const std::array<id<MTLBuffer>, 14> responseBuffers = {
        articulatedHeaderBuffer, operatorStatusBuffer, factorBuffer,
        pointJacobianBuffer, velocityBuffer, contactBuffer, lawBuffer,
        spanBuffer, termBuffer, jacobianBuffer, responseBuffer,
        solverContactBuffer, regularizationBuffer, responseStatusBuffer
    };
    for (std::size_t index = 0u; index < responseBuffers.size(); ++index) {
        [response setBuffer:responseBuffers[index] offset:0 atIndex:index];
    }
    [response setBytes:&problemCount length:sizeof(problemCount) atIndex:14];
    const mr_uint4 inputCaps = u4(
        problemCount * kDofs * kDofs,
        problemCount * kContacts * 3u * kDofs,
        problemCount * kDofs,
        problemCount * kContacts
    );
    const mr_uint4 responseCaps = u4(
        problemCount * kContacts,
        problemCount * kContacts,
        problemCount * kContacts * 3u * kDofs,
        problemCount * kContacts * 3u * kDofs
    );
    const mr_uint4 responseSolverCaps = u4(
        problemCount * kContacts,
        problemCount * kContacts,
        problemCount * kContacts * 9u,
        problemCount
    );
    [response setBytes:&inputCaps length:sizeof(inputCaps) atIndex:15];
    [response setBytes:&responseCaps length:sizeof(responseCaps) atIndex:16];
    [response setBytes:&responseSolverCaps length:sizeof(responseSolverCaps) atIndex:17];
    [response dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [response endEncoding];

    id<MTLComputeCommandEncoder> assembly = [command computeCommandEncoder];
    [assembly setComputePipelineState:pipelines[2]];
    const std::array<id<MTLBuffer>, 11> assemblyBuffers = {
        assemblyHeaderBuffer, spanBuffer, termBuffer, jacobianBuffer,
        responseBuffer, regularizationBuffer, rowBuffer, columnBuffer,
        blockBuffer, streamHeaderBuffer, assemblyStatusBuffer
    };
    for (std::size_t index = 0u; index < assemblyBuffers.size(); ++index) {
        [assembly setBuffer:assemblyBuffers[index] offset:0 atIndex:index];
    }
    [assembly setBytes:&problemCount length:sizeof(problemCount) atIndex:11];
    const mr_uint4 assemblyInputs = responseCaps;
    const mr_uint4 assemblyOutputs = u4(
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columns.size()),
        problemCount * kContacts * 9u,
        static_cast<std::uint32_t>(batch.columns.size() * 9u)
    );
    [assembly setBytes:&assemblyInputs length:sizeof(assemblyInputs) atIndex:12];
    [assembly setBytes:&assemblyOutputs length:sizeof(assemblyOutputs) atIndex:13];
    [assembly dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [assembly endEncoding];

    id<MTLComputeCommandEncoder> solver = [command computeCommandEncoder];
    [solver setComputePipelineState:pipelines[3]];
    const std::array<id<MTLBuffer>, 7> solverBuffers = {
        streamHeaderBuffer, rowBuffer, columnBuffer, blockBuffer,
        solverContactBuffer, impulseBuffer, solverStatusBuffer
    };
    for (std::size_t index = 0u; index < solverBuffers.size(); ++index) {
        [solver setBuffer:solverBuffers[index] offset:0 atIndex:index];
    }
    [solver setBytes:&problemCount length:sizeof(problemCount) atIndex:7];
    const mr_uint4 solverCaps = u4(
        problemCount * kContacts,
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columns.size()),
        problemCount * kContacts
    );
    [solver setBytes:&solverCaps length:sizeof(solverCaps) atIndex:8];
    [solver dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [solver endEncoding];

    id<MTLComputeCommandEncoder> publish = [command computeCommandEncoder];
    [publish setComputePipelineState:pipelines[4]];
    const std::array<id<MTLBuffer>, 8> publishBuffers = {
        articulatedHeaderBuffer, velocityBuffer, responseBuffer, impulseBuffer,
        responseStatusBuffer, solverStatusBuffer, outputVelocityBuffer,
        publishStatusBuffer
    };
    for (std::size_t index = 0u; index < publishBuffers.size(); ++index) {
        [publish setBuffer:publishBuffers[index] offset:0 atIndex:index];
    }
    [publish setBytes:&problemCount length:sizeof(problemCount) atIndex:8];
    const mr_uint4 publishCaps = u4(
        problemCount * kDofs,
        problemCount * kContacts * 3u * kDofs,
        problemCount * kContacts,
        problemCount * kDofs
    );
    [publish setBytes:&publishCaps length:sizeof(publishCaps) atIndex:9];
    [publish dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
               threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [publish endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted && command.error == nil,
        "Metal articulated chain failed: " + errorText(command.error));
    const auto end = std::chrono::steady_clock::now();
    const double gpuSeconds = command.GPUEndTime > command.GPUStartTime
        ? command.GPUEndTime - command.GPUStartTime
        : 0.0;

    const auto measureStage = [&](const auto& encode) {
        const auto wallStart = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> stageCommand = [queue commandBuffer];
        id<MTLComputeCommandEncoder> stage =
            [stageCommand computeCommandEncoder];
        encode(stage);
        [stage endEncoding];
        [stageCommand commit];
        [stageCommand waitUntilCompleted];
        require(stageCommand.status == MTLCommandBufferStatusCompleted &&
                stageCommand.error == nil,
            "Metal articulated stage profile failed: " +
                errorText(stageCommand.error));
        const auto wallEnd = std::chrono::steady_clock::now();
        return stageCommand.GPUEndTime > stageCommand.GPUStartTime
            ? stageCommand.GPUEndTime - stageCommand.GPUStartTime
            : std::chrono::duration<double>(wallEnd - wallStart).count();
    };
    std::array<double, 5> stageSeconds = {};
    stageSeconds[0] = measureStage([&](id<MTLComputeCommandEncoder> stage) {
        [stage setComputePipelineState:pipelines[0]];
        for (std::size_t index = 0u; index < operatorBuffers.size(); ++index) {
            [stage setBuffer:operatorBuffers[index] offset:0 atIndex:index];
        }
        [stage setThreadgroupMemoryLength:operatorThreadgroupBytes(kBodies, kDofs)
                                  atIndex:0];
        [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    });
    stageSeconds[1] = measureStage([&](id<MTLComputeCommandEncoder> stage) {
        [stage setComputePipelineState:pipelines[1]];
        for (std::size_t index = 0u; index < responseBuffers.size(); ++index) {
            [stage setBuffer:responseBuffers[index] offset:0 atIndex:index];
        }
        [stage setBytes:&problemCount length:sizeof(problemCount) atIndex:14];
        [stage setBytes:&inputCaps length:sizeof(inputCaps) atIndex:15];
        [stage setBytes:&responseCaps length:sizeof(responseCaps) atIndex:16];
        [stage setBytes:&responseSolverCaps
                  length:sizeof(responseSolverCaps)
                 atIndex:17];
        [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    });
    stageSeconds[2] = measureStage([&](id<MTLComputeCommandEncoder> stage) {
        [stage setComputePipelineState:pipelines[2]];
        for (std::size_t index = 0u; index < assemblyBuffers.size(); ++index) {
            [stage setBuffer:assemblyBuffers[index] offset:0 atIndex:index];
        }
        [stage setBytes:&problemCount length:sizeof(problemCount) atIndex:11];
        [stage setBytes:&assemblyInputs length:sizeof(assemblyInputs) atIndex:12];
        [stage setBytes:&assemblyOutputs length:sizeof(assemblyOutputs) atIndex:13];
        [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    });
    stageSeconds[3] = measureStage([&](id<MTLComputeCommandEncoder> stage) {
        [stage setComputePipelineState:pipelines[3]];
        for (std::size_t index = 0u; index < solverBuffers.size(); ++index) {
            [stage setBuffer:solverBuffers[index] offset:0 atIndex:index];
        }
        [stage setBytes:&problemCount length:sizeof(problemCount) atIndex:7];
        [stage setBytes:&solverCaps length:sizeof(solverCaps) atIndex:8];
        [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    });
    stageSeconds[4] = measureStage([&](id<MTLComputeCommandEncoder> stage) {
        [stage setComputePipelineState:pipelines[4]];
        for (std::size_t index = 0u; index < publishBuffers.size(); ++index) {
            [stage setBuffer:publishBuffers[index] offset:0 atIndex:index];
        }
        [stage setBytes:&problemCount length:sizeof(problemCount) atIndex:8];
        [stage setBytes:&publishCaps length:sizeof(publishCaps) atIndex:9];
        [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    });
    const double kinematicsSeconds = measureStage(
        [&](id<MTLComputeCommandEncoder> stage) {
            [stage setComputePipelineState:pipelines[0]];
            for (std::size_t index = 0u; index < kinematicBuffers.size(); ++index) {
                [stage setBuffer:kinematicBuffers[index] offset:0 atIndex:index];
            }
            [stage setThreadgroupMemoryLength:operatorThreadgroupBytes(kBodies, kDofs)
                                      atIndex:0];
            [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        }
    );
    const mr_uint4 preparationInputCaps = u4(
        problemCount * kContacts * 3u * kDofs,
        problemCount * kContacts,
        problemCount,
        0u
    );
    const std::uint32_t preparationOutputCapacity =
        problemCount * kContacts * 3u * kDofs;
    const double preparationSeconds = measureStage(
        [&](id<MTLComputeCommandEncoder> stage) {
            [stage setComputePipelineState:pipelines[6]];
            [stage setBuffer:articulatedHeaderBuffer offset:0 atIndex:0];
            [stage setBuffer:kinematicStatusBuffer offset:0 atIndex:1];
            [stage setBuffer:kinematicPointJacobianBuffer offset:0 atIndex:2];
            [stage setBuffer:contactBuffer offset:0 atIndex:3];
            [stage setBuffer:preparedJacobianBuffer offset:0 atIndex:4];
            [stage setBuffer:preparationStatusBuffer offset:0 atIndex:5];
            [stage setBuffer:inverseContactStatusBuffer offset:0 atIndex:6];
            [stage setBytes:&problemCount length:sizeof(problemCount) atIndex:7];
            [stage setBytes:&preparationInputCaps
                      length:sizeof(preparationInputCaps)
                     atIndex:8];
            [stage setBytes:&preparationOutputCapacity
                      length:sizeof(preparationOutputCapacity)
                     atIndex:9];
            [stage dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        }
    );

    MRInverseMassDispatchGPU inverseDispatch = {};
    inverseDispatch.articulationIndex = 0u;
    inverseDispatch.environmentCount = problemCount;
    inverseDispatch.rhsCount = 3u * kContacts;
    inverseDispatch.qStride = kDofs;
    inverseDispatch.rhsEnvironmentStride = 3u * kContacts * kDofs;
    inverseDispatch.rhsVectorStride = kDofs;
    inverseDispatch.outputEnvironmentStride = 3u * kContacts * kDofs;
    inverseDispatch.outputVectorStride = kDofs;
    const auto inverseStart = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> inverseCommand = [queue commandBuffer];
    id<MTLComputeCommandEncoder> inverse = [inverseCommand computeCommandEncoder];
    [inverse setComputePipelineState:pipelines[5]];
    const std::array<id<MTLBuffer>, 5> inverseModelBuffers = {
        worldBuffer, articulationBuffer, jointBuffer, dofBuffer, bodyBuffer
    };
    for (std::size_t index = 0u; index < inverseModelBuffers.size(); ++index) {
        [inverse setBuffer:inverseModelBuffers[index] offset:0 atIndex:index];
    }
    [inverse setBytes:&inverseDispatch length:sizeof(inverseDispatch) atIndex:5];
    [inverse setBuffer:qBuffer offset:0 atIndex:6];
    [inverse setBuffer:preparedJacobianBuffer offset:0 atIndex:7];
    [inverse setBuffer:inverseResponseBuffer offset:0 atIndex:8];
    [inverse setBuffer:inverseStatusBuffer offset:0 atIndex:9];
    [inverse setBuffer:inverseContactStatusBuffer offset:0 atIndex:12];
    [inverse dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [inverse endEncoding];
    [inverseCommand commit];
    [inverseCommand waitUntilCompleted];
    require(inverseCommand.status == MTLCommandBufferStatusCompleted &&
            inverseCommand.error == nil,
        "Metal inverse-ABA action failed: " + errorText(inverseCommand.error));
    const auto inverseEnd = std::chrono::steady_clock::now();
    const double inverseGpuSeconds =
        inverseCommand.GPUEndTime > inverseCommand.GPUStartTime
        ? inverseCommand.GPUEndTime - inverseCommand.GPUStartTime
        : 0.0;

    Result result;
    const auto copy = [](auto& destination, id<MTLBuffer> source) {
        using Value = typename std::decay_t<decltype(destination)>::value_type;
        const auto* begin = static_cast<const Value*>(source.contents);
        destination.assign(begin, begin + source.length / sizeof(Value));
    };
    copy(result.poses, poseBuffer);
    copy(result.pointWorld, pointWorldBuffer);
    copy(result.factors, factorBuffer);
    copy(result.pointJacobians, pointJacobianBuffer);
    copy(result.operatorStatuses, operatorStatusBuffer);
    copy(result.kinematicPointJacobians, kinematicPointJacobianBuffer);
    copy(result.kinematicStatuses, kinematicStatusBuffer);
    copy(result.preparedContactJacobians, preparedJacobianBuffer);
    copy(result.preparationStatuses, preparationStatusBuffer);
    copy(result.spans, spanBuffer);
    copy(result.terms, termBuffer);
    copy(result.jacobians, jacobianBuffer);
    copy(result.responses, responseBuffer);
    copy(result.inverseResponses, inverseResponseBuffer);
    copy(result.inverseStatuses, inverseStatusBuffer);
    copy(result.solverContacts, solverContactBuffer);
    copy(result.regularization, regularizationBuffer);
    copy(result.responseStatuses, responseStatusBuffer);
    copy(result.streamHeaders, streamHeaderBuffer);
    copy(result.blocks, blockBuffer);
    copy(result.assemblyStatuses, assemblyStatusBuffer);
    copy(result.impulses, impulseBuffer);
    copy(result.solverStatuses, solverStatusBuffer);
    copy(result.outputVelocities, outputVelocityBuffer);
    copy(result.publishStatuses, publishStatusBuffer);
    result.seconds = gpuSeconds > 0.0
        ? gpuSeconds
        : std::chrono::duration<double>(end - start).count();
    result.stageSeconds = stageSeconds;
    result.kinematicsSeconds = kinematicsSeconds;
    result.preparationSeconds = preparationSeconds;
    result.inverseSeconds = inverseGpuSeconds > 0.0
        ? inverseGpuSeconds
        : std::chrono::duration<double>(inverseEnd - inverseStart).count();
    return result;
}

using Vec2 = std::array<double, 2>;

Vec2 point(double angle, double length) {
    return {length * std::sin(angle), length * std::cos(angle)};
}

Vec2 add(Vec2 a, Vec2 b) { return {a[0] + b[0], a[1] + b[1]}; }
Vec2 crossY(Vec2 radius) { return {radius[1], -radius[0]}; }
double dot(Vec2 a, Vec2 b) { return a[0] * b[0] + a[1] * b[1]; }

struct Oracle {
    std::vector<double> mass;
    std::vector<Vec2> pointPositions;
    // Contact-major, then generalized-coordinate column.
    std::vector<Vec2> pointJacobians;
};

Oracle oracle(const float* q) {
    Oracle value;
    value.mass.assign(kDofs * kDofs, 0.0);
    value.pointPositions.resize(kContacts);
    value.pointJacobians.assign(kContacts * kDofs, Vec2{0.0, 0.0});
    std::vector<Vec2> jointPositions(kDofs);
    std::vector<Vec2> bodyPositions(kDofs);
    Vec2 joint = {0.0, 0.0};
    double angle = 0.0;
    for (std::size_t link = 0u; link < kDofs; ++link) {
        jointPositions[link] = joint;
        angle += q[link];
        bodyPositions[link] = add(
            joint,
            point(angle, linkHalfLength(link))
        );
        joint = add(
            joint,
            point(angle, 2.0 * linkHalfLength(link))
        );
        value.pointPositions[link] = joint;
        for (std::size_t dof = 0u; dof <= link; ++dof) {
            value.pointJacobians[link * kDofs + dof] = crossY({
                joint[0] - jointPositions[dof][0],
                joint[1] - jointPositions[dof][1]
            });
        }
    }
    for (std::size_t link = 0u; link < kDofs; ++link) {
        std::vector<Vec2> bodyJacobian(kDofs, Vec2{0.0, 0.0});
        for (std::size_t dof = 0u; dof <= link; ++dof) {
            bodyJacobian[dof] = crossY({
                bodyPositions[link][0] - jointPositions[dof][0],
                bodyPositions[link][1] - jointPositions[dof][1]
            });
        }
        for (std::size_t row = 0u; row <= link; ++row) {
            for (std::size_t column = 0u; column <= link; ++column) {
                value.mass[row * kDofs + column] +=
                    linkMass(link) * dot(
                        bodyJacobian[row], bodyJacobian[column]
                    ) + linkInertiaY(link);
            }
        }
    }
    for (std::size_t dof = 0u; dof < kDofs; ++dof) {
        value.mass[dof * kDofs + dof] += linkArmature(dof);
    }
    return value;
}

std::vector<double> factorSPD(const std::vector<double>& matrix) {
    require(matrix.size() == kDofs * kDofs, "invalid FP64 mass dimensions");
    std::vector<double> factor(kDofs * kDofs, 0.0);
    for (std::size_t row = 0u; row < kDofs; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * kDofs + column];
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -= factor[row * kDofs + inner] *
                    factor[column * kDofs + inner];
            }
            if (row == column) {
                require(value > 1.0e-14 && std::isfinite(value),
                    "FP64 articulated mass matrix is not SPD");
                factor[row * kDofs + row] = std::sqrt(value);
            } else {
                factor[row * kDofs + column] =
                    value / factor[column * kDofs + column];
            }
        }
    }
    return factor;
}

std::vector<double> solveFactor(
    const std::vector<double>& factor,
    const std::vector<double>& rightHandSide
) {
    require(rightHandSide.size() == kDofs, "invalid FP64 response dimensions");
    std::vector<double> intermediate(kDofs, 0.0);
    std::vector<double> solution(kDofs, 0.0);
    for (std::size_t row = 0u; row < kDofs; ++row) {
        double value = rightHandSide[row];
        for (std::size_t column = 0u; column < row; ++column) {
            value -= factor[row * kDofs + column] * intermediate[column];
        }
        intermediate[row] = value / factor[row * kDofs + row];
    }
    for (std::size_t reverse = 0u; reverse < kDofs; ++reverse) {
        const std::size_t row = kDofs - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u; column < kDofs; ++column) {
            value -= factor[column * kDofs + row] * solution[column];
        }
        solution[row] = value / factor[row * kDofs + row];
    }
    return solution;
}

int run(int argc, const char* const* argv) {
    std::size_t problemCount = 256u;
    std::uint32_t replayCount = 3u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--islands" && argument + 1 < argc) {
            problemCount = std::stoull(argv[++argument]);
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(std::stoul(argv[++argument]));
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-articulated [--islands N] [--replays N] [--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 8u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    const std::uint64_t maximumElementsPerProblem = std::max({
        std::uint64_t{kDofs} * kDofs,
        std::uint64_t{kContacts} * 3u * kDofs,
        std::uint64_t{kContacts} * kContacts * 9u,
        std::uint64_t{kContacts} * 9u
    });
    require(problemCount <=
            std::numeric_limits<std::uint32_t>::max() /
                maximumElementsPerProblem,
        "island count exceeds articulated ABI");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(device != nil && queue != nil, "no Apple Metal command queue is available");
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    require(library != nil, "failed to load metallib: " + errorText(error));
    const std::array<NSString*, 7> names = {
        @"mr_articulated_operator",
        @"numi_temporal_cone_articulated_response",
        @"numi_temporal_cone_stream_assemble",
        @"numi_temporal_cone_stream_solve",
        @"numi_temporal_cone_articulated_publish",
        @"numi_articulated_inverse_aba_stream",
        @"numi_temporal_cone_articulated_prepare_jacobians"
    };
    std::array<id<MTLComputePipelineState>, 7> pipelines = {};
    for (std::size_t index = 0u; index < pipelines.size(); ++index) {
        id<MTLFunction> function = [library newFunctionWithName:names[index]];
        pipelines[index] = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipelines[index] != nil && pipelines[index].threadExecutionWidth == 32u,
            "failed to create SIMD32 articulated pipeline: " + errorText(error));
    }
    const Batch batch = makeBatch(problemCount);
    (void)runGPU(device, queue, pipelines, batch);
    std::vector<Result> replays;
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipelines, batch));
    }
    const Result& result = replays.front();
    bool deterministic = true;
    for (std::size_t replay = 1u; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
            exactVector(result.poses, replays[replay].poses) &&
            exactVector(result.pointWorld, replays[replay].pointWorld) &&
            exactVector(result.factors, replays[replay].factors) &&
            exactVector(result.pointJacobians, replays[replay].pointJacobians) &&
            exactVector(result.operatorStatuses, replays[replay].operatorStatuses) &&
            exactVector(result.kinematicPointJacobians,
                replays[replay].kinematicPointJacobians) &&
            exactVector(result.kinematicStatuses,
                replays[replay].kinematicStatuses) &&
            exactVector(result.preparedContactJacobians,
                replays[replay].preparedContactJacobians) &&
            exactVector(result.preparationStatuses,
                replays[replay].preparationStatuses) &&
            exactVector(result.spans, replays[replay].spans) &&
            exactVector(result.terms, replays[replay].terms) &&
            exactVector(result.jacobians, replays[replay].jacobians) &&
            exactVector(result.responses, replays[replay].responses) &&
            exactVector(result.inverseResponses, replays[replay].inverseResponses) &&
            exactVector(result.inverseStatuses, replays[replay].inverseStatuses) &&
            exactVector(result.solverContacts, replays[replay].solverContacts) &&
            exactVector(result.regularization, replays[replay].regularization) &&
            exactVector(result.responseStatuses, replays[replay].responseStatuses) &&
            exactVector(result.streamHeaders, replays[replay].streamHeaders) &&
            exactVector(result.blocks, replays[replay].blocks) &&
            exactVector(result.assemblyStatuses, replays[replay].assemblyStatuses) &&
            exactVector(result.impulses, replays[replay].impulses) &&
            exactVector(result.solverStatuses, replays[replay].solverStatuses) &&
            exactVector(result.outputVelocities, replays[replay].outputVelocities) &&
            exactVector(result.publishStatuses, replays[replay].publishStatuses);
    }

    if constexpr (kExpectConditionFailure) {
        bool rejected = deterministic;
        double maximumCondition = 0.0;
        double maximumDiagonalConditionUpper = 0.0;
        double maximumInverseResponseScaledError = 0.0;
        double maximumInverseBackwardError = 0.0;
        double maximumInversePivotRatioSquared = 0.0;
        for (std::size_t problem = 0u;
             problem < batch.validProblems;
             ++problem) {
            maximumCondition = std::max<double>(
                maximumCondition,
                result.responseStatuses[problem].conditioning.z
            );
            rejected = rejected &&
                result.operatorStatuses[problem].code ==
                    MR_ARTICULATED_OPERATOR_SUCCESS &&
                result.responseStatuses[problem].control.x ==
                    NUMI_TEMPORAL_CONE_ARTICULATED_CONDITIONING_FAILED &&
                result.publishStatuses[problem].control.x ==
                    NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE &&
                std::memcmp(
                    &result.outputVelocities[kDofs * problem],
                    &batch.velocities[kDofs * problem],
                    kDofs * sizeof(float)
                ) == 0;
            rejected = rejected &&
                result.preparationStatuses[problem].control.x ==
                    NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS &&
                result.inverseStatuses[problem].code ==
                    MR_INVERSE_MASS_SUCCESS;
            const Oracle ref = oracle(batch.q.data() + problem * kDofs);
            const std::vector<double> cpuFactor = factorSPD(ref.mass);
            double matrixInfinity = 0.0;
            double inverseInfinity = 0.0;
            double maximumDiagonal = 0.0;
            for (std::size_t row = 0u; row < kDofs; ++row) {
                double rowSum = 0.0;
                for (std::size_t column = 0u; column < kDofs; ++column) {
                    rowSum += std::abs(ref.mass[row * kDofs + column]);
                }
                matrixInfinity = std::max(matrixInfinity, rowSum);
                maximumDiagonal = std::max(
                    maximumDiagonal,
                    ref.mass[row * kDofs + row]
                );
                std::vector<double> unit(kDofs, 0.0);
                unit[row] = 1.0;
                const std::vector<double> inverseColumn = solveFactor(
                    cpuFactor, unit
                );
                double inverseRowSum = 0.0;
                for (double value : inverseColumn) {
                    inverseRowSum += std::abs(value);
                }
                inverseInfinity = std::max(inverseInfinity, inverseRowSum);
            }
            maximumDiagonalConditionUpper = std::max(
                maximumDiagonalConditionUpper,
                static_cast<double>(kDofs) * maximumDiagonal * inverseInfinity
            );
            const auto& inverseStatus = result.inverseStatuses[problem];
            const double pivotRatio = inverseStatus.diagnostics.y /
                inverseStatus.diagnostics.x;
            maximumInversePivotRatioSquared = std::max(
                maximumInversePivotRatioSquared,
                pivotRatio * pivotRatio
            );
            const std::size_t valueBase =
                problem * kContacts * 3u * kDofs;
            for (std::size_t contactIndex = 0u;
                 contactIndex < kContacts;
                 ++contactIndex) {
                std::array<std::vector<double>, 3> rhs = {
                    std::vector<double>(kDofs, 0.0),
                    std::vector<double>(kDofs, 0.0),
                    std::vector<double>(kDofs, 0.0)
                };
                for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                    const Vec2 column = ref.pointJacobians[
                        contactIndex * kDofs + dof
                    ];
                    rhs[0][dof] = column[1];
                    rhs[1][dof] = column[0];
                }
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    const std::vector<double> reference = solveFactor(
                        cpuFactor, rhs[axis]
                    );
                    double maximumRhs = 0.0;
                    double maximumSolution = 0.0;
                    double maximumResidual = 0.0;
                    for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                        const double actual = result.inverseResponses[
                            valueBase + (contactIndex * 3u + axis) *
                                kDofs + dof
                        ];
                        maximumInverseResponseScaledError = std::max(
                            maximumInverseResponseScaledError,
                            std::abs(actual - reference[dof]) /
                                std::max(1.0, std::abs(reference[dof]))
                        );
                        maximumRhs = std::max(
                            maximumRhs, std::abs(rhs[axis][dof])
                        );
                        maximumSolution = std::max(
                            maximumSolution, std::abs(actual)
                        );
                    }
                    for (std::size_t row = 0u; row < kDofs; ++row) {
                        double action = 0.0;
                        for (std::size_t column = 0u;
                             column < kDofs;
                             ++column) {
                            action += ref.mass[row * kDofs + column] *
                                result.inverseResponses[
                                    valueBase + (contactIndex * 3u + axis) *
                                        kDofs + column
                                ];
                        }
                        maximumResidual = std::max(
                            maximumResidual,
                            std::abs(action - rhs[axis][row])
                        );
                    }
                    maximumInverseBackwardError = std::max(
                        maximumInverseBackwardError,
                        maximumResidual /
                            (maximumRhs + matrixInfinity * maximumSolution +
                             1.0e-30)
                    );
                }
            }
        }
        require(rejected, "ill-conditioned articulated response was published");
        require(maximumInverseResponseScaledError < 3.0e-5,
            "inverse ABA lost accuracy on the conditioning adversary");
        require(maximumInverseBackwardError < 3.0e-6,
            "inverse ABA residual failed on the conditioning adversary");
        const double bestSeconds = std::min_element(
            replays.begin(),
            replays.end(),
            [](const Result& a, const Result& b) {
                return a.seconds < b.seconds;
            }
        )->seconds;
        std::cout << std::setprecision(9)
                  << "numi-solver-articulated-conditioning device=\""
                  << device.name.UTF8String << "\""
                  << " dofs=" << kDofs
                  << " islands=" << problemCount
                  << " rejected_valid=" << batch.validProblems
                  << " maximum_condition_infinity=" << maximumCondition
                  << " threshold="
                  << NUMI_TEMPORAL_CONE_ARTICULATED_MAX_CONDITION_INFINITY
                  << " diagonal_condition_upper_bound="
                  << maximumDiagonalConditionUpper
                  << " inverse_pivot_ratio_squared="
                  << maximumInversePivotRatioSquared
                  << " inverse_response_max_scaled_error="
                  << maximumInverseResponseScaledError
                  << " inverse_backward_error="
                  << maximumInverseBackwardError
                  << " deterministic=" << (deterministic ? "yes" : "no")
                  << " rollback=" << (rejected ? "yes" : "no")
                  << " gpu_seconds=" << bestSeconds
                  << '\n';
        return 0;
    }

    double maxMassError = 0.0;
    double maxMassScaledError = 0.0;
    double maxJacobianError = 0.0;
    double maxJacobianScaledError = 0.0;
    double maxKinematicJacobianDifference = 0.0;
    double maxPreparedJacobianDifference = 0.0;
    double maxFiniteDifferenceError = 0.0;
    double maxResponseError = 0.0;
    double maxResponseScaledError = 0.0;
    double maxInverseResponseError = 0.0;
    double maxInverseResponseScaledError = 0.0;
    double maxDenseInverseDifference = 0.0;
    double maxInverseBackwardError = 0.0;
    double maxBlockError = 0.0;
    double maxBlockScaledError = 0.0;
    double maxFreeVelocityError = 0.0;
    double maxPublicationError = 0.0;
    double maxEnergyBudgetViolation = 0.0;
    double maxFp64Residual = 0.0;
    double maxGpuBackwardError = 0.0;
    double maxConditionInfinity = 0.0;
    double maxDiagonalConditionUpperBound = 0.0;
    double maxConditionScaledError = 0.0;
    std::uint32_t maxIterations = 0u;
    std::size_t failedValid = 0u;
    std::size_t failedInverseValid = 0u;
    std::size_t failedKinematicValid = 0u;
    std::size_t failedPreparationValid = 0u;
    const std::size_t valuesPerProblem = kContacts * 3u * kDofs;
    const std::size_t blocksPerProblem = kContacts * kContacts;
    for (std::size_t problem = 0u; problem < batch.validProblems; ++problem) {
        const std::size_t dofBase = problem * kDofs;
        const std::size_t contactBase = problem * kContacts;
        const std::size_t factorBase = problem * kDofs * kDofs;
        const std::size_t valueBase = problem * valuesPerProblem;
        const std::size_t regularizationBase = problem * kContacts * 9u;
        const std::size_t blockBase = problem * blocksPerProblem * 9u;
        const Oracle ref = oracle(batch.q.data() + dofBase);
        const std::vector<double> cpuFactor = factorSPD(ref.mass);
        double cpuMatrixInfinity = 0.0;
        double cpuInverseInfinity = 0.0;
        double cpuMaximumDiagonal = 0.0;
        for (std::size_t row = 0u; row < kDofs; ++row) {
            double rowSum = 0.0;
            for (std::size_t column = 0u; column < kDofs; ++column) {
                rowSum += std::abs(ref.mass[row * kDofs + column]);
            }
            cpuMatrixInfinity = std::max(cpuMatrixInfinity, rowSum);
            cpuMaximumDiagonal = std::max(
                cpuMaximumDiagonal,
                ref.mass[row * kDofs + row]
            );
            std::vector<double> unit(kDofs, 0.0);
            unit[row] = 1.0;
            const std::vector<double> inverseColumn = solveFactor(
                cpuFactor, unit
            );
            double inverseRowSum = 0.0;
            for (double value : inverseColumn) {
                inverseRowSum += std::abs(value);
            }
            cpuInverseInfinity = std::max(
                cpuInverseInfinity,
                inverseRowSum
            );
        }
        const double cpuConditionInfinity =
            cpuMatrixInfinity * cpuInverseInfinity;
        maxDiagonalConditionUpperBound = std::max(
            maxDiagonalConditionUpperBound,
            static_cast<double>(kDofs) * cpuMaximumDiagonal *
                cpuInverseInfinity
        );
        const double gpuConditionInfinity =
            result.responseStatuses[problem].conditioning.z;
        if (result.kinematicStatuses[problem].code !=
                MR_ARTICULATED_OPERATOR_SUCCESS ||
            result.kinematicStatuses[problem].nv != kDofs) {
            ++failedKinematicValid;
        }
        if (result.preparationStatuses[problem].control.x !=
                NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS ||
            result.preparationStatuses[problem].control.w != kContacts) {
            ++failedPreparationValid;
        }
        maxConditionScaledError = std::max(
            maxConditionScaledError,
            std::abs(gpuConditionInfinity - cpuConditionInfinity) /
                std::max(1.0, cpuConditionInfinity)
        );
        for (std::size_t row = 0u; row < kDofs; ++row) {
            for (std::size_t column = 0u; column < kDofs; ++column) {
                double gpuMass = 0.0;
                const std::size_t innerCount = std::min(row, column) + 1u;
                for (std::size_t inner = 0u; inner < innerCount; ++inner) {
                    gpuMass += result.factors[
                        factorBase + row * kDofs + inner
                    ] * result.factors[
                        factorBase + column * kDofs + inner
                    ];
                }
                const double reference = ref.mass[row * kDofs + column];
                const double error = std::abs(gpuMass - reference);
                maxMassError = std::max(maxMassError, error);
                maxMassScaledError = std::max(
                    maxMassScaledError,
                    error / std::max(1.0, std::abs(reference))
                );
            }
        }
        for (std::size_t pointIndex = 0u;
             pointIndex < kContacts;
             ++pointIndex) {
            const std::size_t worldBase = valueBase +
                pointIndex * 3u * kDofs;
            for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                for (std::size_t component = 0u; component < 3u; ++component) {
                    maxKinematicJacobianDifference = std::max(
                        maxKinematicJacobianDifference,
                        static_cast<double>(std::abs(
                            result.pointJacobians[
                                worldBase + component * kDofs + dof
                            ] -
                            result.kinematicPointJacobians[
                                worldBase + component * kDofs + dof
                            ]
                        ))
                    );
                }
                const Vec2 reference = ref.pointJacobians[
                    pointIndex * kDofs + dof
                ];
                const Vec2 gpu = {
                    result.pointJacobians[worldBase + dof],
                    result.pointJacobians[worldBase + 2u * kDofs + dof]
                };
                for (std::size_t component = 0u; component < 2u; ++component) {
                    const double error = std::abs(gpu[component] - reference[component]);
                    maxJacobianError = std::max(maxJacobianError, error);
                    maxJacobianScaledError = std::max(
                        maxJacobianScaledError,
                        error / std::max(1.0, std::abs(reference[component]))
                    );
                }
            }
        }
        for (std::size_t value = 0u; value < valuesPerProblem; ++value) {
            maxPreparedJacobianDifference = std::max(
                maxPreparedJacobianDifference,
                static_cast<double>(std::abs(
                    result.jacobians[valueBase + value] -
                    result.preparedContactJacobians[valueBase + value]
                ))
            );
        }
        if (problem < 2u) {
            constexpr float h = 1.0e-3f;
            std::vector<float> qPlus(
                batch.q.begin() + static_cast<std::ptrdiff_t>(dofBase),
                batch.q.begin() + static_cast<std::ptrdiff_t>(dofBase + kDofs)
            );
            std::vector<float> qMinus = qPlus;
            for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                qPlus[dof] += h;
                qMinus[dof] -= h;
                const Oracle plus = oracle(qPlus.data());
                const Oracle minus = oracle(qMinus.data());
                qPlus[dof] -= h;
                qMinus[dof] += h;
                for (std::size_t pointIndex = dof;
                     pointIndex < kContacts;
                     ++pointIndex) {
                    const Vec2 finiteDifference = {
                        (plus.pointPositions[pointIndex][0] -
                            minus.pointPositions[pointIndex][0]) / (2.0 * h),
                        (plus.pointPositions[pointIndex][1] -
                            minus.pointPositions[pointIndex][1]) / (2.0 * h)
                    };
                    const std::size_t worldBase = valueBase +
                        pointIndex * 3u * kDofs;
                    maxFiniteDifferenceError = std::max(
                        maxFiniteDifferenceError,
                        std::max(
                            std::abs(result.pointJacobians[worldBase + dof] -
                                finiteDifference[0]),
                            std::abs(result.pointJacobians[
                                worldBase + 2u * kDofs + dof
                            ] - finiteDifference[1])
                        )
                    );
                }
            }
        }
        std::vector<double> initialVelocity(kDofs);
        for (std::size_t dof = 0u; dof < kDofs; ++dof) {
            initialVelocity[dof] = batch.velocities[dofBase + dof];
        }
        if (result.inverseStatuses[problem].code != MR_INVERSE_MASS_SUCCESS ||
            result.inverseStatuses[problem].rhsCount != 3u * kContacts ||
            result.inverseStatuses[problem].nv != kDofs) {
            ++failedInverseValid;
        }
        for (std::size_t contactIndex = 0u;
             contactIndex < kContacts;
             ++contactIndex) {
            std::array<std::vector<double>, 3> rhs = {
                std::vector<double>(kDofs, 0.0),
                std::vector<double>(kDofs, 0.0),
                std::vector<double>(kDofs, 0.0)
            };
            double rawNormal = 0.0;
            double rawTangent = 0.0;
            for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                const Vec2 column = ref.pointJacobians[
                    contactIndex * kDofs + dof
                ];
                rhs[0][dof] = column[1];
                rhs[1][dof] = column[0];
                rawNormal += column[1] * initialVelocity[dof];
                rawTangent += column[0] * initialVelocity[dof];
            }
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const std::vector<double> response = solveFactor(
                    cpuFactor, rhs[axis]
                );
                double inverseMaximumRightHandSide = 0.0;
                double inverseMaximumSolution = 0.0;
                double inverseMaximumResidual = 0.0;
                for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                    const double actual = result.responses[
                        valueBase + contactIndex * 3u * kDofs +
                            dof * 3u + axis
                    ];
                    const double inverseActual = result.inverseResponses[
                        valueBase + (contactIndex * 3u + axis) * kDofs + dof
                    ];
                    const double error = std::abs(actual - response[dof]);
                    const double inverseError = std::abs(
                        inverseActual - response[dof]
                    );
                    maxResponseError = std::max(maxResponseError, error);
                    maxResponseScaledError = std::max(
                        maxResponseScaledError,
                        error / std::max(1.0, std::abs(response[dof]))
                    );
                    maxInverseResponseError = std::max(
                        maxInverseResponseError,
                        inverseError
                    );
                    maxInverseResponseScaledError = std::max(
                        maxInverseResponseScaledError,
                        inverseError /
                            std::max(1.0, std::abs(response[dof]))
                    );
                    maxDenseInverseDifference = std::max(
                        maxDenseInverseDifference,
                        std::abs(inverseActual - actual)
                    );
                    inverseMaximumRightHandSide = std::max(
                        inverseMaximumRightHandSide,
                        std::abs(rhs[axis][dof])
                    );
                    inverseMaximumSolution = std::max(
                        inverseMaximumSolution,
                        std::abs(inverseActual)
                    );
                }
                for (std::size_t row = 0u; row < kDofs; ++row) {
                    double action = 0.0;
                    double inverseAction = 0.0;
                    for (std::size_t column = 0u;
                         column < kDofs;
                         ++column) {
                        action += ref.mass[row * kDofs + column] *
                            response[column];
                        inverseAction += ref.mass[row * kDofs + column] *
                            result.inverseResponses[
                                valueBase + (contactIndex * 3u + axis) *
                                    kDofs + column
                            ];
                    }
                    maxFp64Residual = std::max(
                        maxFp64Residual,
                        std::abs(action - rhs[axis][row])
                    );
                    inverseMaximumResidual = std::max(
                        inverseMaximumResidual,
                        std::abs(inverseAction - rhs[axis][row])
                    );
                }
                maxInverseBackwardError = std::max(
                    maxInverseBackwardError,
                    inverseMaximumResidual /
                        (inverseMaximumRightHandSide +
                         cpuMatrixInfinity * inverseMaximumSolution + 1.0e-30)
                );
            }
            const auto& solverContact = result.solverContacts[
                contactBase + contactIndex
            ];
            const auto& lawValue = batch.laws[contactBase + contactIndex];
            const double dt = lawValue.stabilization.w;
            const double denominator = lawValue.dampingAndImpactThreshold.x +
                dt * lawValue.stiffnessAndRestitution.x;
            const double penetration = std::min<double>(
                lawValue.stabilization.x + lawValue.stabilization.y, 0.0);
            const double recovery = std::min<double>(lawValue.stabilization.z,
                -lawValue.stiffnessAndRestitution.x * penetration / denominator);
            const double rebound = rawNormal < -lawValue.dampingAndImpactThreshold.w
                ? -lawValue.stiffnessAndRestitution.w * rawNormal : 0.0;
            const double target = std::max(recovery, rebound);
            maxFreeVelocityError = std::max(
                maxFreeVelocityError,
                std::max(
                    std::abs(solverContact.freeVelocityAndFrictionU.x -
                        (rawNormal - target)),
                    std::abs(solverContact.freeVelocityAndFrictionU.y -
                        rawTangent)
                )
            );
        }
        const bool checkEveryBlock = kDofs <= 8u || problem < 8u;
        if (checkEveryBlock) {
          for (std::size_t target = 0u; target < kContacts; ++target) {
            for (std::size_t source = 0u; source < kContacts; ++source) {
                for (std::size_t row = 0u; row < 3u; ++row) {
                    for (std::size_t column = 0u; column < 3u; ++column) {
                        double expected = 0.0;
                        for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                            expected += result.jacobians[
                                valueBase + target * 3u * kDofs +
                                    row * kDofs + dof
                            ] * result.responses[
                                valueBase + source * 3u * kDofs +
                                    dof * 3u + column
                            ];
                        }
                        if (target == source && row == column) {
                            expected += result.regularization[
                                regularizationBase + target * 9u + 4u * row
                            ];
                        }
                        const double actual = result.blocks[
                            blockBase + (target * kContacts + source) * 9u +
                                3u * row + column
                        ];
                        const double error = std::abs(actual - expected);
                        maxBlockError = std::max(maxBlockError, error);
                        maxBlockScaledError = std::max(
                            maxBlockScaledError,
                            error / std::max(1.0, std::abs(expected))
                        );
                    }
                }
            }
          }
        }
        std::vector<double> expectedVelocity = initialVelocity;
        for (std::size_t contactIndex = 0u;
             contactIndex < kContacts;
             ++contactIndex) {
            const auto impulse = result.impulses[contactBase + contactIndex];
            const std::size_t responseBase = valueBase +
                contactIndex * 3u * kDofs;
            for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                expectedVelocity[dof] +=
                    result.responses[responseBase + dof * 3u + 0u] * impulse.x +
                    result.responses[responseBase + dof * 3u + 1u] * impulse.y +
                    result.responses[responseBase + dof * 3u + 2u] * impulse.z;
            }
        }
        std::vector<double> finalVelocity(kDofs);
        for (std::size_t dof = 0u; dof < kDofs; ++dof) {
            finalVelocity[dof] = result.outputVelocities[dofBase + dof];
            maxPublicationError = std::max(
                maxPublicationError,
                std::abs(finalVelocity[dof] - expectedVelocity[dof])
            );
        }
        const auto kinetic = [&](const std::vector<double>& velocity) {
            double energy = 0.0;
            for (std::size_t row = 0u; row < kDofs; ++row) {
                for (std::size_t column = 0u; column < kDofs; ++column) {
                    energy += 0.5 * velocity[row] *
                        ref.mass[row * kDofs + column] * velocity[column];
                }
            }
            return energy;
        };
        const double deltaEnergy = kinetic(finalVelocity) - kinetic(initialVelocity);
        double budget = 0.0;
        for (std::size_t contactIndex = 0u;
             contactIndex < kContacts;
             ++contactIndex) {
            const auto impulse = result.impulses[contactBase + contactIndex];
            const auto solverContact = result.solverContacts[
                contactBase + contactIndex
            ];
            double rawNormal = 0.0;
            for (std::size_t dof = 0u; dof < kDofs; ++dof) {
                rawNormal += result.jacobians[
                    valueBase + contactIndex * 3u * kDofs + dof
                ] * initialVelocity[dof];
            }
            const double targetNormal =
                -solverContact.freeVelocityAndFrictionU.x + rawNormal;
            budget += impulse.x * targetNormal;
            budget -= 0.5 * (
                result.regularization[
                    regularizationBase + contactIndex * 9u + 0u
                ] * impulse.x * impulse.x +
                result.regularization[
                    regularizationBase + contactIndex * 9u + 4u
                ] * impulse.y * impulse.y +
                result.regularization[
                    regularizationBase + contactIndex * 9u + 8u
                ] * impulse.z * impulse.z
            );
        }
        maxEnergyBudgetViolation = std::max(maxEnergyBudgetViolation, deltaEnergy - budget);
        maxGpuBackwardError = std::max<double>(maxGpuBackwardError,
            result.responseStatuses[problem].diagnostics.y);
        maxConditionInfinity = std::max<double>(maxConditionInfinity,
            gpuConditionInfinity);
        maxIterations = std::max(maxIterations, result.solverStatuses[problem].control.y);
        const bool valid = result.operatorStatuses[problem].code == MR_ARTICULATED_OPERATOR_SUCCESS &&
            result.responseStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS &&
            result.assemblyStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
            result.solverStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            result.publishStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS;
        if (!valid) ++failedValid;
    }
    bool rollback = true;
    for (std::size_t problem = batch.validProblems; problem < problemCount; ++problem) {
        const bool operatorFailure = problem == batch.validProblems;
        const std::uint32_t expectedResponse = operatorFailure
            ? NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE
            : NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
        bool operatorPayloadUntouched = true;
        if (operatorFailure) {
            operatorPayloadUntouched =
                result.operatorStatuses[problem].code ==
                    MR_ARTICULATED_OPERATOR_NONFINITE_INPUT;
            for (std::size_t value = 0u;
                 value < kDofs * kDofs;
                 ++value) {
                operatorPayloadUntouched = operatorPayloadUntouched &&
                    result.factors[
                        problem * kDofs * kDofs + value
                    ] == 0.0f;
            }
            for (std::size_t value = 0u;
                 value < valuesPerProblem;
                 ++value) {
                operatorPayloadUntouched = operatorPayloadUntouched &&
                    result.pointJacobians[
                        problem * valuesPerProblem + value
                    ] == 0.0f;
            }
        } else {
            operatorPayloadUntouched =
                result.operatorStatuses[problem].code ==
                    MR_ARTICULATED_OPERATOR_SUCCESS;
        }
        rollback = rollback && operatorPayloadUntouched &&
            result.responseStatuses[problem].control.x == expectedResponse &&
            result.publishStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE &&
            std::memcmp(&result.outputVelocities[kDofs * problem],
                &batch.velocities[kDofs * problem],
                kDofs * sizeof(float)) == 0;
    }
    if (failedValid != 0u || !rollback) {
        const auto printStatus = [&](std::size_t problem) {
            std::cerr << "status problem=" << problem
                      << " operator=" << result.operatorStatuses[problem].code
                      << ":" << result.operatorStatuses[problem].failingIndex
                      << " response=" << result.responseStatuses[problem].control.x
                      << " assembly=" << result.assemblyStatuses[problem].control.x
                      << " solver=" << result.solverStatuses[problem].control.x
                      << " publish=" << result.publishStatuses[problem].control.x
                      << '\n';
        };
        printStatus(0u);
        for (std::size_t problem = batch.validProblems;
             problem < problemCount;
             ++problem) {
            printStatus(problem);
        }
    }
    if (failedInverseValid != 0u ||
        failedKinematicValid != 0u ||
        failedPreparationValid != 0u ||
        maxKinematicJacobianDifference != 0.0 ||
        maxPreparedJacobianDifference != 0.0 ||
        maxMassScaledError >= 2.0e-5 ||
        maxJacobianScaledError >= 2.0e-5 ||
        maxFiniteDifferenceError >= 2.0e-3 ||
        maxResponseScaledError >= 3.0e-5 ||
        maxInverseResponseScaledError >= 3.0e-5 ||
        maxInverseBackwardError >= 3.0e-6 ||
        maxBlockScaledError >= 3.0e-5 ||
        maxFreeVelocityError >= 2.0e-4 ||
        maxPublicationError >= 2.0e-4 ||
        maxEnergyBudgetViolation >= 2.0e-3 ||
        maxConditionScaledError >= 2.0e-3) {
        std::cerr << "oracle metrics mass=" << maxMassError
                  << "/" << maxMassScaledError
                  << " jacobian=" << maxJacobianError
                  << "/" << maxJacobianScaledError
                  << " kinematic_jacobian=" << maxKinematicJacobianDifference
                  << " prepared_jacobian=" << maxPreparedJacobianDifference
                  << " finite_difference=" << maxFiniteDifferenceError
                  << " response=" << maxResponseError
                  << "/" << maxResponseScaledError
                  << " inverse_response=" << maxInverseResponseError
                  << "/" << maxInverseResponseScaledError
                  << " dense_inverse=" << maxDenseInverseDifference
                  << " inverse_backward=" << maxInverseBackwardError
                  << " block=" << maxBlockError
                  << "/" << maxBlockScaledError
                  << " free=" << maxFreeVelocityError
                  << " publication=" << maxPublicationError
                  << " energy=" << maxEnergyBudgetViolation
                  << " condition_infinity=" << maxConditionInfinity
                  << " condition_error=" << maxConditionScaledError
                  << " gpu_backward=" << maxGpuBackwardError
                  << " failed_inverse_valid=" << failedInverseValid
                  << " failed_kinematic_valid=" << failedKinematicValid
                  << " failed_preparation_valid=" << failedPreparationValid
                  << '\n';
    }
    require(deterministic, "articulated chain is not byte deterministic");
    require(rollback, "articulated invalid-input rollback failed");
    require(failedValid == 0u, "valid articulated islands failed");
    require(failedInverseValid == 0u, "valid inverse-ABA actions failed");
    require(failedKinematicValid == 0u,
        "valid kinematics-only articulated operators failed");
    require(failedPreparationValid == 0u,
        "valid articulated Jacobian preparation failed");
    require(maxMassScaledError < 2.0e-5, "articulated mass oracle mismatch");
    require(maxJacobianScaledError < 2.0e-5, "articulated Jacobian oracle mismatch");
    require(maxKinematicJacobianDifference == 0.0,
        "kinematics-only Jacobian changed arithmetic");
    require(maxPreparedJacobianDifference == 0.0,
        "prepared contact Jacobian changed arithmetic");
    require(maxFiniteDifferenceError < 2.0e-3, "articulated finite-difference mismatch");
    require(maxResponseScaledError < 3.0e-5, "articulated response-column mismatch");
    require(maxInverseResponseScaledError < 3.0e-5,
        "inverse-ABA response-column mismatch");
    require(maxInverseBackwardError < 3.0e-6,
        "inverse-ABA backward residual mismatch");
    require(maxBlockScaledError < 3.0e-5, "articulated Delassus block mismatch");
    require(maxFreeVelocityError < 2.0e-4, "articulated free-velocity mismatch");
    require(maxPublicationError < 2.0e-4, "articulated publication mismatch");
    require(maxEnergyBudgetViolation < 2.0e-3, "articulated energy budget violated");
    require(maxConditionScaledError < 2.0e-3,
        "articulated condition estimate mismatch");

    const double bestSeconds = std::min_element(
        replays.begin(), replays.end(), [](const Result& a, const Result& b) {
            return a.seconds < b.seconds;
        })->seconds;
    const double bestInverseSeconds = std::min_element(
        replays.begin(), replays.end(), [](const Result& a, const Result& b) {
            return a.inverseSeconds < b.inverseSeconds;
        })->inverseSeconds;
    const double bestKinematicsSeconds = std::min_element(
        replays.begin(), replays.end(), [](const Result& a, const Result& b) {
            return a.kinematicsSeconds < b.kinematicsSeconds;
        })->kinematicsSeconds;
    const double bestPreparationSeconds = std::min_element(
        replays.begin(), replays.end(), [](const Result& a, const Result& b) {
            return a.preparationSeconds < b.preparationSeconds;
        })->preparationSeconds;
    std::array<double, 5> bestStageSeconds = replays.front().stageSeconds;
    for (const Result& replay : replays) {
        for (std::size_t stage = 0u; stage < bestStageSeconds.size(); ++stage) {
            bestStageSeconds[stage] = std::min(
                bestStageSeconds[stage], replay.stageSeconds[stage]
            );
        }
    }
    std::cout << std::setprecision(9)
              << "numi-solver-articulated device=\"" << device.name.UTF8String << "\""
              << " dofs=" << kDofs
              << " islands=" << problemCount
              << " valid=" << batch.validProblems
              << " contacts=" << batch.validProblems * kContacts
              << " solver_stages=5 solver_command_buffers=1"
              << " diagnostic_command_buffers=8"
              << " readbacks_between_solver_stages=0"
              << " deterministic=" << (deterministic ? "yes" : "no")
              << " rollback=" << (rollback ? "yes" : "no")
              << " failed_valid=" << failedValid << "\n"
              << "mass_max_abs_error=" << maxMassError
              << " mass_max_scaled_error=" << maxMassScaledError
              << " jacobian_max_abs_error=" << maxJacobianError
              << " jacobian_max_scaled_error=" << maxJacobianScaledError
              << " kinematic_jacobian_max_abs_difference="
              << maxKinematicJacobianDifference
              << " prepared_jacobian_max_abs_difference="
              << maxPreparedJacobianDifference
              << " finite_difference_max_abs_error=" << maxFiniteDifferenceError
              << " response_max_abs_error=" << maxResponseError
              << " response_max_scaled_error=" << maxResponseScaledError
              << " inverse_response_max_abs_error=" << maxInverseResponseError
              << " inverse_response_max_scaled_error=" << maxInverseResponseScaledError
              << " dense_inverse_max_abs_difference=" << maxDenseInverseDifference
              << " delassus_max_abs_error=" << maxBlockError << "\n"
              << "delassus_max_scaled_error=" << maxBlockScaledError
              << " "
              << "free_velocity_max_abs_error=" << maxFreeVelocityError
              << " publication_max_abs_error=" << maxPublicationError
              << " fp64_solve_residual=" << maxFp64Residual
              << " gpu_response_backward_error=" << maxGpuBackwardError
              << " inverse_aba_backward_error=" << maxInverseBackwardError << "\n"
              << "condition_infinity=" << maxConditionInfinity
              << " diagonal_condition_upper_bound="
              << maxDiagonalConditionUpperBound
              << " condition_max_scaled_error=" << maxConditionScaledError
              << " energy_budget_violation=" << std::max(0.0, maxEnergyBudgetViolation)
              << " max_iterations=" << maxIterations << "\n"
              << "gpu_seconds=" << bestSeconds
              << " islands_per_second=" << static_cast<double>(problemCount) / bestSeconds
              << " contacts_per_second=" << static_cast<double>(batch.validProblems * kContacts) / bestSeconds
              << " operator_threadgroup_bytes="
              << operatorThreadgroupBytes(kBodies, kDofs)
              << " factor_bytes="
              << problemCount * kDofs * kDofs * sizeof(float)
              << " jacobian_response_bytes="
              << 2u * problemCount * valuesPerProblem * sizeof(float)
              << " block_bytes="
              << problemCount * blocksPerProblem * 9u * sizeof(float)
              << " inverse_aba_seconds=" << bestInverseSeconds
              << " kinematics_only_seconds=" << bestKinematicsSeconds
              << " contact_prepare_seconds=" << bestPreparationSeconds
              << " inverse_aba_rhs_per_second="
              << static_cast<double>(batch.validProblems * kContacts * 3u) /
                 bestInverseSeconds
              << " stage_gpu_seconds="
              << bestStageSeconds[0] << ","
              << bestStageSeconds[1] << ","
              << bestStageSeconds[2] << ","
              << bestStageSeconds[3] << ","
              << bestStageSeconds[4]
              << "\n";
    return 0;
}

} // namespace

int main(int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "numi-solver-articulated: " << error.what() << '\n';
            return 1;
        }
    }
}

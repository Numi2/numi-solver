#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/temporal_cone_island.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

constexpr std::size_t kMaxContacts =
    NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
constexpr std::size_t kMaxRows = NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS;
constexpr std::size_t kMatrixElements =
    NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;
struct TermDraft {
    std::uint32_t owner = 0u;
    std::uint32_t dofCount = 0u;
    std::vector<float> jacobian;
};

struct AssemblyBatch {
    std::vector<NumiTemporalConeAssemblyHeader> headers;
    std::vector<NumiTemporalConeAssemblyContactSpan> spans;
    std::vector<NumiTemporalConeAssemblyTerm> terms;
    std::vector<float> jacobianValues;
    std::vector<float> responseValues;
    std::vector<float> regularizationValues;
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columnIndices;
    std::vector<NumiTemporalConeIslandContact> contacts;
    std::vector<float> denseMatrices;
};

struct GPUResult {
    std::vector<NumiTemporalConeStreamHeader> headers;
    std::vector<float> blockValues;
    std::vector<NumiTemporalConeAssemblyStatus> assemblyStatuses;
    std::vector<mr_float4> impulses;
    std::vector<NumiTemporalConeIslandStatus> solverStatuses;
    double seconds = 0.0;
};

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

mr_uint4 u4(
    const std::uint32_t x,
    const std::uint32_t y,
    const std::uint32_t z,
    const std::uint32_t w
) {
    return {x, y, z, w};
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

std::vector<std::pair<std::size_t, std::size_t>> makeEdges(
    const std::size_t problem,
    const std::size_t contactCount
) {
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    const auto add = [&](std::size_t first, std::size_t second) {
        if (first == second || first >= contactCount ||
            second >= contactCount) {
            return;
        }
        if (first > second) {
            std::swap(first, second);
        }
        edges.emplace_back(first, second);
    };
    const std::size_t mode = (problem / 6u) % 5u;
    if (mode == 0u) {
        for (std::size_t contact = 1u; contact < contactCount; ++contact) {
            add(contact - 1u, contact);
        }
    } else if (mode == 1u) {
        for (std::size_t contact = 1u; contact < contactCount; ++contact) {
            add(contact - 1u, contact);
        }
        add(0u, contactCount - 1u);
    } else if (mode == 2u) {
        for (std::size_t contact = 1u; contact < contactCount; ++contact) {
            add(0u, contact);
        }
    } else if (mode == 3u) {
        for (std::size_t contact = 0u; contact < contactCount; ++contact) {
            add(contact, contact + 1u);
            add(contact, contact + 2u);
        }
    } else {
        for (std::size_t contact = 1u; contact < contactCount; ++contact) {
            add(contact - 1u, contact);
            add(contact / 2u, contact);
        }
    }
    if (contactCount == kMaxContacts && problem % 41u == 35u) {
        for (std::size_t first = 0u; first < contactCount; ++first) {
            for (std::size_t second = first + 1u;
                 second < contactCount;
                 ++second) {
                add(first, second);
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

AssemblyBatch makeBatch(const std::size_t problemCount) {
    AssemblyBatch batch;
    batch.headers.reserve(problemCount);
    batch.denseMatrices.assign(problemCount * kMatrixElements, 0.0f);
    constexpr std::array<std::uint32_t, 6> counts{{1u, 2u, 4u, 8u, 16u, 32u}};

    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        const std::uint32_t contactCount = counts[problem % counts.size()];
        const std::size_t contactBase = batch.contacts.size();
        const std::size_t spanBase = batch.spans.size();
        const std::size_t rowOffsetBase = batch.rowOffsets.size();
        const std::size_t blockBase = batch.columnIndices.size();
        const std::size_t regularizationBase =
            batch.regularizationValues.size();
        std::vector<std::vector<TermDraft>> drafts(contactCount);
        std::vector<std::vector<bool>> connected(
            contactCount,
            std::vector<bool>(contactCount, false)
        );
        for (std::size_t contact = 0u; contact < contactCount; ++contact) {
            connected[contact][contact] = true;
        }

        if (problem == 1u && contactCount == 2u) {
            for (std::size_t contact = 0u; contact < 2u; ++contact) {
                TermDraft shared;
                shared.owner = 0u;
                shared.dofCount = 1u;
                shared.jacobian = {1.0f, 0.0f, 0.0f};
                drafts[contact].push_back(std::move(shared));
            }
            connected[0][1] = true;
            connected[1][0] = true;
        } else {
            for (std::size_t contact = 0u;
                 contact < contactCount;
                 ++contact) {
                TermDraft local;
                local.owner = static_cast<std::uint32_t>(contact);
                local.dofCount = problem % 37u == 0u && contact == 0u
                    ? NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_DOF_PER_TERM
                    : 3u;
                local.jacobian.assign(3u * local.dofCount, 0.0f);
                local.jacobian[0u] = std::sqrt(
                    0.8f + 0.01f * static_cast<float>((problem + contact) % 5u)
                );
                local.jacobian[local.dofCount + 1u] = std::sqrt(1.0f);
                local.jacobian[2u * local.dofCount + 2u] = std::sqrt(1.2f);
                drafts[contact].push_back(std::move(local));
            }
            const auto edges = makeEdges(problem, contactCount);
            for (std::size_t edgeIndex = 0u;
                 edgeIndex < edges.size();
                 ++edgeIndex) {
                const auto [first, second] = edges[edgeIndex];
                const std::uint32_t owner = static_cast<std::uint32_t>(
                    contactCount + edgeIndex
                );
                for (const std::size_t contact : {first, second}) {
                    TermDraft edge;
                    edge.owner = owner;
                    edge.dofCount = 2u;
                    edge.jacobian.resize(6u);
                    for (std::size_t axis = 0u; axis < 3u; ++axis) {
                        for (std::size_t dof = 0u; dof < 2u; ++dof) {
                            const double phase = static_cast<double>(
                                1u + problem + 3u * edgeIndex +
                                5u * contact + 7u * axis + dof
                            );
                            edge.jacobian[2u * axis + dof] =
                                0.13f * static_cast<float>(
                                    dof == 0u
                                    ? std::sin(0.23 * phase)
                                    : std::cos(0.19 * phase)
                                );
                        }
                    }
                    drafts[contact].push_back(std::move(edge));
                }
                connected[first][second] = true;
                connected[second][first] = true;
            }
        }

        for (std::size_t contact = 0u;
             contact < contactCount;
             ++contact) {
            auto& contactDrafts = drafts[contact];
            std::sort(
                contactDrafts.begin(),
                contactDrafts.end(),
                [](const TermDraft& lhs, const TermDraft& rhs) {
                    return lhs.owner < rhs.owner;
                }
            );
            if (contactDrafts.size() >
                NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_TERMS_PER_CONTACT) {
                throw std::runtime_error("generated contact exceeds term capacity");
            }
            NumiTemporalConeAssemblyContactSpan span{};
            span.ranges = u4(
                static_cast<std::uint32_t>(batch.terms.size()),
                static_cast<std::uint32_t>(contactDrafts.size()),
                0u,
                0u
            );
            batch.spans.push_back(span);
            for (const auto& draft : contactDrafts) {
                NumiTemporalConeAssemblyTerm term{};
                term.control = u4(
                    draft.owner,
                    draft.dofCount,
                    static_cast<std::uint32_t>(batch.jacobianValues.size()),
                    static_cast<std::uint32_t>(batch.responseValues.size())
                );
                batch.jacobianValues.insert(
                    batch.jacobianValues.end(),
                    draft.jacobian.begin(),
                    draft.jacobian.end()
                );
                for (std::size_t dof = 0u; dof < draft.dofCount; ++dof) {
                    for (std::size_t axis = 0u; axis < 3u; ++axis) {
                        batch.responseValues.push_back(
                            draft.jacobian[axis * draft.dofCount + dof]
                        );
                    }
                }
                batch.terms.push_back(term);
            }

            for (std::size_t row = 0u; row < 3u; ++row) {
                for (std::size_t column = 0u; column < 3u; ++column) {
                    batch.regularizationValues.push_back(
                        row == column ? (problem == 1u ? 1.0f : 0.02f) : 0.0f
                    );
                }
            }
            NumiTemporalConeIslandContact solverContact{};
            if (problem == 1u) {
                solverContact.freeVelocityAndFrictionU =
                    f4(-1.0f, 0.0f, 0.0f, 0.0f);
                solverContact.warmImpulseAndFrictionV =
                    f4(0.0f, 0.0f, 0.0f, 0.0f);
            } else {
                const float frictionU =
                    0.25f + 0.05f * static_cast<float>((problem + contact) % 7u);
                const float frictionV = problem % 3u == 0u
                    ? frictionU
                    : 0.2f + 0.04f * static_cast<float>(
                        (2u * problem + contact) % 9u
                    );
                solverContact.freeVelocityAndFrictionU = f4(
                    (problem + contact) % 11u == 0u ? 0.05f : -0.2f,
                    0.2f * std::sin(
                        0.17f * static_cast<float>(problem + contact + 1u)
                    ),
                    0.2f * std::cos(
                        0.13f * static_cast<float>(problem + 2u * contact + 1u)
                    ),
                    frictionU
                );
                solverContact.warmImpulseAndFrictionV = f4(
                    0.005f * static_cast<float>((problem + contact) % 5u),
                    0.0f,
                    0.0f,
                    frictionV
                );
            }
            solverContact.limits = f4(
                (problem + contact) % 29u == 0u ? 1.25f : 0.0f,
                0.0f,
                0.0f,
                0.0f
            );
            batch.contacts.push_back(solverContact);
        }

        for (std::size_t target = 0u;
             target < contactCount;
             ++target) {
            batch.rowOffsets.push_back(static_cast<std::uint32_t>(
                batch.columnIndices.size() - blockBase
            ));
            for (std::size_t source = 0u;
                 source < contactCount;
                 ++source) {
                if (connected[target][source]) {
                    batch.columnIndices.push_back(
                        static_cast<std::uint32_t>(source)
                    );
                }
            }
        }
        batch.rowOffsets.push_back(static_cast<std::uint32_t>(
            batch.columnIndices.size() - blockBase
        ));
        const std::size_t blockCount =
            batch.columnIndices.size() - blockBase;
        NumiTemporalConeAssemblyHeader header{};
        header.control = u4(
            NUMI_TEMPORAL_CONE_ASSEMBLY_ABI_VERSION,
            contactCount,
            4u,
            512u
        );
        header.outputRanges = u4(
            static_cast<std::uint32_t>(contactBase),
            static_cast<std::uint32_t>(rowOffsetBase),
            static_cast<std::uint32_t>(blockBase),
            static_cast<std::uint32_t>(blockCount)
        );
        header.inputRanges = u4(
            static_cast<std::uint32_t>(spanBase),
            static_cast<std::uint32_t>(regularizationBase),
            0u,
            0u
        );
        header.tolerances = f4(5.0e-7f, 1.0e-6f, 1.0f, 0.0f);
        batch.headers.push_back(header);
    }

    // Independent CPU assembly of the dense oracle from the packed factors.
    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        const auto& header = batch.headers[problem];
        const std::size_t matrixBase = problem * kMatrixElements;
        for (std::size_t target = 0u; target < header.control.y; ++target) {
            const auto& targetSpan = batch.spans[header.inputRanges.x + target];
            for (std::size_t source = 0u; source < header.control.y; ++source) {
                const auto& sourceSpan =
                    batch.spans[header.inputRanges.x + source];
                for (std::size_t targetAxis = 0u;
                     targetAxis < 3u;
                     ++targetAxis) {
                    for (std::size_t sourceAxis = 0u;
                         sourceAxis < 3u;
                         ++sourceAxis) {
                        float coefficient = target == source
                            ? batch.regularizationValues[
                                header.inputRanges.y + target * 9u +
                                3u * targetAxis + sourceAxis
                            ]
                            : 0.0f;
                        std::size_t targetTerm = 0u;
                        std::size_t sourceTerm = 0u;
                        while (targetTerm < targetSpan.ranges.y &&
                               sourceTerm < sourceSpan.ranges.y) {
                            const auto& lhs = batch.terms[
                                targetSpan.ranges.x + targetTerm
                            ];
                            const auto& rhs = batch.terms[
                                sourceSpan.ranges.x + sourceTerm
                            ];
                            if (lhs.control.x < rhs.control.x) {
                                ++targetTerm;
                                continue;
                            }
                            if (rhs.control.x < lhs.control.x) {
                                ++sourceTerm;
                                continue;
                            }
                            for (std::size_t dof = 0u;
                                 dof < lhs.control.y;
                                 ++dof) {
                                coefficient = std::fma(
                                    batch.jacobianValues[
                                        lhs.control.z +
                                        targetAxis * lhs.control.y + dof
                                    ],
                                    batch.responseValues[
                                        rhs.control.w + 3u * dof + sourceAxis
                                    ],
                                    coefficient
                                );
                            }
                            ++targetTerm;
                            ++sourceTerm;
                        }
                        batch.denseMatrices[
                            matrixBase +
                            (3u * target + targetAxis) * kMaxRows +
                            3u * source + sourceAxis
                        ] = coefficient;
                    }
                }
            }
        }
    }
    return batch;
}

double minimumCholeskyPivot(
    const AssemblyBatch& batch,
    const std::size_t problem
) {
    const std::size_t dimension = 3u * batch.headers[problem].control.y;
    const std::size_t matrixBase = problem * kMatrixElements;
    std::vector<double> lower(dimension * dimension, 0.0);
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double reduced = batch.denseMatrices[
                matrixBase + row * kMaxRows + column
            ];
            if (batch.denseMatrices[matrixBase + row * kMaxRows + column] !=
                batch.denseMatrices[matrixBase + column * kMaxRows + row]) {
                return -1.0;
            }
            for (std::size_t inner = 0u; inner < column; ++inner) {
                reduced -= lower[row * dimension + inner] *
                    lower[column * dimension + inner];
            }
            if (row == column) {
                if (!(reduced > 1.0e-10) || !std::isfinite(reduced)) {
                    return -1.0;
                }
                const double pivot = std::sqrt(reduced);
                lower[row * dimension + column] = pivot;
                minimum = std::min(minimum, pivot);
            } else {
                lower[row * dimension + column] = reduced /
                    lower[column * dimension + column];
            }
        }
    }
    return minimum;
}

GPUResult runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> assemblyPipeline,
    id<MTLComputePipelineState> solverPipeline,
    const AssemblyBatch& batch,
    const bool includeSolve = true
) {
    const std::size_t problemCount = batch.headers.size();
    const auto makeBytes = [&](const auto& values) -> id<MTLBuffer> {
        return [device
            newBufferWithBytes:values.data()
                       length:values.size() * sizeof(values.front())
                      options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> headerBuffer = makeBytes(batch.headers);
    id<MTLBuffer> spanBuffer = makeBytes(batch.spans);
    id<MTLBuffer> termBuffer = makeBytes(batch.terms);
    id<MTLBuffer> jacobianBuffer = makeBytes(batch.jacobianValues);
    id<MTLBuffer> responseBuffer = makeBytes(batch.responseValues);
    id<MTLBuffer> regularizationBuffer =
        makeBytes(batch.regularizationValues);
    id<MTLBuffer> rowOffsetBuffer = makeBytes(batch.rowOffsets);
    id<MTLBuffer> columnBuffer = makeBytes(batch.columnIndices);
    id<MTLBuffer> contactBuffer = makeBytes(batch.contacts);
    id<MTLBuffer> blockBuffer = [device
        newBufferWithLength:batch.columnIndices.size() * 9u * sizeof(float)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> streamHeaderBuffer = [device
        newBufferWithLength:problemCount *
            sizeof(NumiTemporalConeStreamHeader)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> assemblyStatusBuffer = [device
        newBufferWithLength:problemCount *
            sizeof(NumiTemporalConeAssemblyStatus)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> impulseBuffer = [device
        newBufferWithLength:batch.contacts.size() * sizeof(mr_float4)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> solverStatusBuffer = [device
        newBufferWithLength:problemCount *
            sizeof(NumiTemporalConeIslandStatus)
                    options:MTLResourceStorageModeShared];
    if (headerBuffer == nil || spanBuffer == nil || termBuffer == nil ||
        jacobianBuffer == nil || responseBuffer == nil ||
        regularizationBuffer == nil || rowOffsetBuffer == nil ||
        columnBuffer == nil || contactBuffer == nil || blockBuffer == nil ||
        streamHeaderBuffer == nil || assemblyStatusBuffer == nil ||
        impulseBuffer == nil || solverStatusBuffer == nil) {
        throw std::runtime_error("failed to allocate assembly buffers");
    }
    std::memset(blockBuffer.contents, 0, blockBuffer.length);
    std::memset(streamHeaderBuffer.contents, 0, streamHeaderBuffer.length);
    std::memset(assemblyStatusBuffer.contents, 0, assemblyStatusBuffer.length);
    std::memset(impulseBuffer.contents, 0, impulseBuffer.length);
    std::memset(solverStatusBuffer.contents, 0, solverStatusBuffer.length);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> assembly =
        [commandBuffer computeCommandEncoder];
    [assembly setComputePipelineState:assemblyPipeline];
    [assembly setBuffer:headerBuffer offset:0 atIndex:0];
    [assembly setBuffer:spanBuffer offset:0 atIndex:1];
    [assembly setBuffer:termBuffer offset:0 atIndex:2];
    [assembly setBuffer:jacobianBuffer offset:0 atIndex:3];
    [assembly setBuffer:responseBuffer offset:0 atIndex:4];
    [assembly setBuffer:regularizationBuffer offset:0 atIndex:5];
    [assembly setBuffer:rowOffsetBuffer offset:0 atIndex:6];
    [assembly setBuffer:columnBuffer offset:0 atIndex:7];
    [assembly setBuffer:blockBuffer offset:0 atIndex:8];
    [assembly setBuffer:streamHeaderBuffer offset:0 atIndex:9];
    [assembly setBuffer:assemblyStatusBuffer offset:0 atIndex:10];
    const std::uint32_t count = static_cast<std::uint32_t>(problemCount);
    [assembly setBytes:&count length:sizeof(count) atIndex:11];
    const mr_uint4 inputCapacities = u4(
        static_cast<std::uint32_t>(batch.spans.size()),
        static_cast<std::uint32_t>(batch.terms.size()),
        static_cast<std::uint32_t>(batch.jacobianValues.size()),
        static_cast<std::uint32_t>(batch.responseValues.size())
    );
    const mr_uint4 outputCapacities = u4(
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columnIndices.size()),
        static_cast<std::uint32_t>(batch.regularizationValues.size()),
        static_cast<std::uint32_t>(batch.columnIndices.size() * 9u)
    );
    [assembly setBytes:&inputCapacities
                length:sizeof(inputCapacities)
               atIndex:12];
    [assembly setBytes:&outputCapacities
                length:sizeof(outputCapacities)
               atIndex:13];
    [assembly dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [assembly endEncoding];

    if (includeSolve) {
        id<MTLComputeCommandEncoder> solver =
            [commandBuffer computeCommandEncoder];
        [solver setComputePipelineState:solverPipeline];
        [solver setBuffer:streamHeaderBuffer offset:0 atIndex:0];
        [solver setBuffer:rowOffsetBuffer offset:0 atIndex:1];
        [solver setBuffer:columnBuffer offset:0 atIndex:2];
        [solver setBuffer:blockBuffer offset:0 atIndex:3];
        [solver setBuffer:contactBuffer offset:0 atIndex:4];
        [solver setBuffer:impulseBuffer offset:0 atIndex:5];
        [solver setBuffer:solverStatusBuffer offset:0 atIndex:6];
        [solver setBytes:&count length:sizeof(count) atIndex:7];
        const mr_uint4 solverCapacities = u4(
            static_cast<std::uint32_t>(batch.contacts.size()),
            static_cast<std::uint32_t>(batch.rowOffsets.size()),
            static_cast<std::uint32_t>(batch.columnIndices.size()),
            static_cast<std::uint32_t>(batch.contacts.size())
        );
        [solver setBytes:&solverCapacities
                  length:sizeof(solverCapacities)
                 atIndex:8];
        [solver dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                   threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [solver endEncoding];
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        throw std::runtime_error(
            "Metal assembly/solve failed: " + errorText(commandBuffer.error)
        );
    }

    GPUResult result;
    const auto* outputHeaders =
        static_cast<const NumiTemporalConeStreamHeader*>(
            streamHeaderBuffer.contents
        );
    result.headers.assign(outputHeaders, outputHeaders + problemCount);
    const auto* blocks = static_cast<const float*>(blockBuffer.contents);
    result.blockValues.assign(
        blocks,
        blocks + batch.columnIndices.size() * 9u
    );
    const auto* assemblyStatuses =
        static_cast<const NumiTemporalConeAssemblyStatus*>(
            assemblyStatusBuffer.contents
        );
    result.assemblyStatuses.assign(
        assemblyStatuses,
        assemblyStatuses + problemCount
    );
    const auto* impulses = static_cast<const mr_float4*>(
        impulseBuffer.contents
    );
    result.impulses.assign(impulses, impulses + batch.contacts.size());
    const auto* solverStatuses =
        static_cast<const NumiTemporalConeIslandStatus*>(
            solverStatusBuffer.contents
        );
    result.solverStatuses.assign(
        solverStatuses,
        solverStatuses + problemCount
    );
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

int run(const int argc, const char* const* argv) {
    std::size_t problemCount = 256u;
    std::uint32_t replayCount = 3u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--islands" && argument + 1 < argc) {
            problemCount = std::stoull(argv[++argument]);
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-assembly [--islands N] "
                         "[--replays N] [--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 36u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    if (problemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("island count exceeds the ABI");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (device == nil || queue == nil) {
        throw std::runtime_error("no Apple Metal command queue is available");
    }
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        throw std::runtime_error("failed to load metallib: " + errorText(error));
    }
    id<MTLFunction> assemblyFunction = [library
        newFunctionWithName:@"numi_temporal_cone_stream_assemble"];
    id<MTLFunction> solverFunction = [library
        newFunctionWithName:@"numi_temporal_cone_stream_solve"];
    id<MTLComputePipelineState> assemblyPipeline = [device
        newComputePipelineStateWithFunction:assemblyFunction
                                      error:&error];
    id<MTLComputePipelineState> solverPipeline = [device
        newComputePipelineStateWithFunction:solverFunction
                                      error:&error];
    if (assemblyPipeline == nil || solverPipeline == nil ||
        assemblyPipeline.threadExecutionWidth != 32u ||
        solverPipeline.threadExecutionWidth != 32u) {
        throw std::runtime_error(
            "failed to create SIMD32 assembly pipelines: " + errorText(error)
        );
    }

    const AssemblyBatch batch = makeBatch(problemCount);
    // Pipeline creation does not guarantee the first command sees steady
    // instruction/data-cache state. Execute one unreported chain and one
    // unreported assembly-only command before timing deterministic replays.
    (void)runGPU(
        device,
        queue,
        assemblyPipeline,
        solverPipeline,
        batch
    );
    (void)runGPU(
        device,
        queue,
        assemblyPipeline,
        solverPipeline,
        batch,
        false
    );
    std::vector<GPUResult> replays;
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(
            device,
            queue,
            assemblyPipeline,
            solverPipeline,
            batch
        ));
    }
    double assemblyOnlySeconds = 0.0;
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        assemblyOnlySeconds += runGPU(
            device,
            queue,
            assemblyPipeline,
            solverPipeline,
            batch,
            false
        ).seconds;
    }
    bool deterministic = true;
    for (std::size_t replay = 1u; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
            std::memcmp(
                replays[0].headers.data(),
                replays[replay].headers.data(),
                replays[0].headers.size() *
                    sizeof(NumiTemporalConeStreamHeader)
            ) == 0 &&
            std::memcmp(
                replays[0].blockValues.data(),
                replays[replay].blockValues.data(),
                replays[0].blockValues.size() * sizeof(float)
            ) == 0 &&
            std::memcmp(
                replays[0].assemblyStatuses.data(),
                replays[replay].assemblyStatuses.data(),
                replays[0].assemblyStatuses.size() *
                    sizeof(NumiTemporalConeAssemblyStatus)
            ) == 0 &&
            std::memcmp(
                replays[0].impulses.data(),
                replays[replay].impulses.data(),
                replays[0].impulses.size() * sizeof(mr_float4)
            ) == 0 &&
            std::memcmp(
                replays[0].solverStatuses.data(),
                replays[replay].solverStatuses.data(),
                replays[0].solverStatuses.size() *
                    sizeof(NumiTemporalConeIslandStatus)
            ) == 0;
    }

    std::size_t failedIslands = 0u;
    double maximumAssemblyError = 0.0;
    double maximumSymmetryError = 0.0;
    double maximumKKTResidual = 0.0;
    double maximumConeViolation = 0.0;
    double maximumPositiveObjective = 0.0;
    double minimumPivot = std::numeric_limits<double>::infinity();
    std::uint64_t contacts = 0u;
    std::uint64_t contactIterations = 0u;
    std::vector<std::uint32_t> iterationCounts;
    iterationCounts.reserve(problemCount);
    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        const auto& input = batch.headers[problem];
        const auto& assembly = replays[0].assemblyStatuses[problem];
        const auto& solver = replays[0].solverStatuses[problem];
        bool valid =
            assembly.control.x == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
            solver.control.x == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            replays[0].headers[problem].control.x ==
                NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION;
        const double pivot = minimumCholeskyPivot(batch, problem);
        valid = valid && pivot > 0.0;
        if (pivot > 0.0) {
            minimumPivot = std::min(minimumPivot, pivot);
        }
        maximumSymmetryError = std::max(
            maximumSymmetryError,
            static_cast<double>(assembly.diagnostics.x)
        );
        maximumKKTResidual = std::max(
            maximumKKTResidual,
            static_cast<double>(solver.residuals.x)
        );
        maximumConeViolation = std::max(
            maximumConeViolation,
            static_cast<double>(solver.residuals.y)
        );
        maximumPositiveObjective = std::max(
            maximumPositiveObjective,
            std::max(static_cast<double>(solver.residuals.w), 0.0)
        );
        contacts += input.control.y;
        contactIterations +=
            static_cast<std::uint64_t>(input.control.y) * solver.control.y;
        iterationCounts.push_back(solver.control.y);
        const std::size_t matrixBase = problem * kMatrixElements;
        for (std::size_t target = 0u; target < input.control.y; ++target) {
            const std::size_t rowBegin = batch.rowOffsets[
                input.outputRanges.y + target
            ];
            const std::size_t rowEnd = batch.rowOffsets[
                input.outputRanges.y + target + 1u
            ];
            for (std::size_t relativeBlock = rowBegin;
                 relativeBlock < rowEnd;
                 ++relativeBlock) {
                const std::size_t block =
                    input.outputRanges.z + relativeBlock;
                const std::size_t source = batch.columnIndices[block];
                for (std::size_t row = 0u; row < 3u; ++row) {
                    for (std::size_t column = 0u; column < 3u; ++column) {
                        const double expected = batch.denseMatrices[
                            matrixBase +
                            (3u * target + row) * kMaxRows +
                            3u * source + column
                        ];
                        const double actual = replays[0].blockValues[
                            block * 9u + 3u * row + column
                        ];
                        maximumAssemblyError = std::max(
                            maximumAssemblyError,
                            std::abs(actual - expected) /
                                std::max(1.0, std::abs(expected))
                        );
                        valid = valid && std::isfinite(actual);
                    }
                }
            }
        }
        if (!valid) {
            if (failedIslands < 5u) {
                std::cerr
                    << "failed_island=" << problem
                    << " contacts=" << input.control.y
                    << " assembly_status=" << assembly.control.x
                    << " solver_status=" << solver.control.x
                    << " kkt=" << solver.residuals.x << '\n';
            }
            ++failedIslands;
        }
    }
    const std::size_t sharedBase = batch.headers[1].outputRanges.x;
    const bool sharedRigidOracle =
        std::abs(static_cast<double>(replays[0].impulses[sharedBase].x) -
                 1.0 / 3.0) <= 2.0e-6 &&
        std::abs(static_cast<double>(replays[0].impulses[sharedBase + 1u].x) -
                 1.0 / 3.0) <= 2.0e-6;
    std::sort(iterationCounts.begin(), iterationCounts.end());
    const auto iterationPercentile = [&](const std::size_t numerator) {
        const std::size_t rank = std::max<std::size_t>(
            1u,
            (numerator * iterationCounts.size() + 99u) / 100u
        );
        return iterationCounts[
            std::min(rank, iterationCounts.size()) - 1u
        ];
    };
    const std::uint32_t iterationP50 = iterationPercentile(50u);
    const std::uint32_t iterationP95 = iterationPercentile(95u);
    const std::uint32_t iterationP99 = iterationPercentile(99u);
    const std::uint32_t maximumIterations = iterationCounts.back();

    // Response-column asymmetry must invalidate the transactional header and
    // make the chained solver publish no physical iterate.
    AssemblyBatch asymmetric = makeBatch(36u);
    const auto& asymmetricSpan = asymmetric.spans[
        asymmetric.headers[1].inputRanges.x
    ];
    asymmetric.responseValues[
        asymmetric.terms[asymmetricSpan.ranges.x].control.w
    ] += 0.25f;
    const GPUResult asymmetricFirst = runGPU(
        device, queue, assemblyPipeline, solverPipeline, asymmetric
    );
    const GPUResult asymmetricReplay = runGPU(
        device, queue, assemblyPipeline, solverPipeline, asymmetric
    );
    const bool asymmetricRejected =
        asymmetricFirst.assemblyStatuses[1].control.x ==
            NUMI_TEMPORAL_CONE_ASSEMBLY_ASYMMETRIC_RESPONSE &&
        asymmetricFirst.headers[1].control.x == 0u &&
        asymmetricFirst.solverStatuses[1].control.x ==
            NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI;

    AssemblyBatch missing = makeBatch(36u);
    auto& missingHeader = missing.headers[1];
    const std::size_t missingRowBase = missingHeader.outputRanges.y;
    missing.rowOffsets[missingRowBase + 0u] = 0u;
    missing.rowOffsets[missingRowBase + 1u] = 1u;
    missing.rowOffsets[missingRowBase + 2u] = 2u;
    missing.columnIndices[missingHeader.outputRanges.z + 0u] = 0u;
    missing.columnIndices[missingHeader.outputRanges.z + 1u] = 1u;
    missingHeader.outputRanges.w = 2u;
    const GPUResult missingFirst = runGPU(
        device, queue, assemblyPipeline, solverPipeline, missing
    );
    const GPUResult missingReplay = runGPU(
        device, queue, assemblyPipeline, solverPipeline, missing
    );
    const bool missingRejected =
        missingFirst.assemblyStatuses[1].control.x ==
            NUMI_TEMPORAL_CONE_ASSEMBLY_MISSING_COUPLING &&
        missingFirst.headers[1].control.x == 0u &&
        missingFirst.solverStatuses[1].control.x ==
            NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI;
    const auto sameResult = [](const GPUResult& first,
                               const GPUResult& second) {
        return std::memcmp(
                   first.headers.data(),
                   second.headers.data(),
                   first.headers.size() *
                       sizeof(NumiTemporalConeStreamHeader)
               ) == 0 &&
            std::memcmp(
                first.blockValues.data(),
                second.blockValues.data(),
                first.blockValues.size() * sizeof(float)
            ) == 0 &&
            std::memcmp(
                first.assemblyStatuses.data(),
                second.assemblyStatuses.data(),
                first.assemblyStatuses.size() *
                    sizeof(NumiTemporalConeAssemblyStatus)
            ) == 0 &&
            std::memcmp(
                first.impulses.data(),
                second.impulses.data(),
                first.impulses.size() * sizeof(mr_float4)
            ) == 0 &&
            std::memcmp(
                first.solverStatuses.data(),
                second.solverStatuses.data(),
                first.solverStatuses.size() *
                    sizeof(NumiTemporalConeIslandStatus)
            ) == 0;
    };
    const bool deterministicFailures =
        sameResult(asymmetricFirst, asymmetricReplay) &&
        sameResult(missingFirst, missingReplay);
    const auto zeroFailureImpulses = [](const AssemblyBatch& failedBatch,
                                        const GPUResult& result) {
        const auto& failedHeader = failedBatch.headers[1];
        for (std::size_t contact = 0u;
             contact < failedHeader.control.y;
             ++contact) {
            const auto& impulse = result.impulses[
                failedHeader.outputRanges.x + contact
            ];
            if (impulse.x != 0.0f || impulse.y != 0.0f ||
                impulse.z != 0.0f || impulse.w != 0.0f) {
                return false;
            }
        }
        return true;
    };
    const bool failureRollback =
        zeroFailureImpulses(asymmetric, asymmetricFirst) &&
        zeroFailureImpulses(missing, missingFirst);

    double totalSeconds = 0.0;
    for (const auto& replay : replays) {
        totalSeconds += replay.seconds;
    }
    const double averageSeconds = totalSeconds / replays.size();
    const double averageAssemblySeconds =
        assemblyOnlySeconds / replayCount;
    const double islandsPerSecond = problemCount / averageSeconds;
    const double blocksPerSecond =
        batch.columnIndices.size() / averageSeconds;
    const double contactIterationsPerSecond =
        contactIterations / averageSeconds;
    const std::uint64_t factorBytes =
        batch.spans.size() * sizeof(NumiTemporalConeAssemblyContactSpan) +
        batch.terms.size() * sizeof(NumiTemporalConeAssemblyTerm) +
        (batch.jacobianValues.size() + batch.responseValues.size() +
         batch.regularizationValues.size()) * sizeof(float);
    const std::uint64_t streamedOperatorBytes =
        batch.rowOffsets.size() * sizeof(std::uint32_t) +
        batch.columnIndices.size() * sizeof(std::uint32_t) +
        batch.columnIndices.size() * 9u * sizeof(float);
    const std::uint64_t denseOperatorBytes =
        problemCount * kMatrixElements * sizeof(float);
    std::uint32_t maximumTerms = 0u;
    std::uint32_t maximumTermDOFs = 0u;
    for (const auto& span : batch.spans) {
        maximumTerms = std::max(maximumTerms, span.ranges.y);
    }
    for (const auto& term : batch.terms) {
        maximumTermDOFs = std::max(maximumTermDOFs, term.control.y);
    }
    std::uint32_t maximumIslandBlocks = 0u;
    std::uint32_t fullCapacityIslands = 0u;
    std::uint64_t factorFMAs = 0u;
    for (const auto& header : batch.headers) {
        maximumIslandBlocks = std::max(
            maximumIslandBlocks,
            header.outputRanges.w
        );
        if (header.outputRanges.w ==
            NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS) {
            ++fullCapacityIslands;
        }
        for (std::size_t target = 0u;
             target < header.control.y;
             ++target) {
            const auto& targetSpan = batch.spans[
                header.inputRanges.x + target
            ];
            const std::size_t rowBegin = batch.rowOffsets[
                header.outputRanges.y + target
            ];
            const std::size_t rowEnd = batch.rowOffsets[
                header.outputRanges.y + target + 1u
            ];
            for (std::size_t relativeBlock = rowBegin;
                 relativeBlock < rowEnd;
                 ++relativeBlock) {
                const std::size_t source = batch.columnIndices[
                    header.outputRanges.z + relativeBlock
                ];
                const auto& sourceSpan = batch.spans[
                    header.inputRanges.x + source
                ];
                std::size_t targetTerm = 0u;
                std::size_t sourceTerm = 0u;
                while (targetTerm < targetSpan.ranges.y &&
                       sourceTerm < sourceSpan.ranges.y) {
                    const auto& lhs = batch.terms[
                        targetSpan.ranges.x + targetTerm
                    ];
                    const auto& rhs = batch.terms[
                        sourceSpan.ranges.x + sourceTerm
                    ];
                    if (lhs.control.x < rhs.control.x) {
                        ++targetTerm;
                    } else if (rhs.control.x < lhs.control.x) {
                        ++sourceTerm;
                    } else {
                        factorFMAs += 9u * lhs.control.y;
                        ++targetTerm;
                        ++sourceTerm;
                    }
                }
            }
        }
    }
    const double factorFMAsPerSecond =
        averageAssemblySeconds > 0.0
        ? factorFMAs / averageAssemblySeconds
        : 0.0;
    const double assemblyBlocksPerSecond =
        averageAssemblySeconds > 0.0
        ? batch.columnIndices.size() / averageAssemblySeconds
        : 0.0;
    const double assemblyFraction = averageSeconds > 0.0
        ? averageAssemblySeconds / averageSeconds
        : 0.0;
    const double factorAndStreamToDense = denseOperatorBytes > 0u
        ? static_cast<double>(factorBytes + streamedOperatorBytes) /
            static_cast<double>(denseOperatorBytes)
        : 0.0;
    const bool passed =
        failedIslands == 0u &&
        maximumAssemblyError <= 2.0e-6 &&
        maximumSymmetryError <= 2.0e-6 &&
        maximumKKTResidual <= 2.0e-6 &&
        maximumConeViolation <= 2.0e-6 &&
        maximumPositiveObjective <= 2.0e-5 &&
        maximumTerms ==
            NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_TERMS_PER_CONTACT &&
        maximumTermDOFs ==
            NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_DOF_PER_TERM &&
        maximumIslandBlocks == NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS &&
        fullCapacityIslands > 0u &&
        deterministic &&
        sharedRigidOracle &&
        asymmetricRejected &&
        missingRejected &&
        deterministicFailures &&
        failureRollback;

    std::cout << std::fixed << std::setprecision(9)
              << "device=" << device.name.UTF8String << '\n'
              << "islands=" << problemCount
              << " contacts=" << contacts
              << " blocks=" << batch.columnIndices.size()
              << " terms=" << batch.terms.size()
              << " replays=" << replayCount
              << " failed_islands=" << failedIslands << '\n'
              << "max_assembly_error=" << maximumAssemblyError
              << " max_symmetry_error=" << maximumSymmetryError
              << " min_spd_pivot=" << minimumPivot
              << " max_kkt_residual=" << maximumKKTResidual
              << " max_cone_violation=" << maximumConeViolation
              << " max_positive_objective=" << maximumPositiveObjective
              << " max_iterations=" << maximumIterations
              << " iteration_p50=" << iterationP50
              << " iteration_p95=" << iterationP95
              << " iteration_p99=" << iterationP99
              << '\n'
              << "deterministic_replay="
              << (deterministic ? "true" : "false")
              << " shared_rigid_oracle="
              << (sharedRigidOracle ? "true" : "false")
              << " asymmetric_rejected="
              << (asymmetricRejected ? "true" : "false")
              << " missing_coupling_rejected="
              << (missingRejected ? "true" : "false")
              << " deterministic_failures="
              << (deterministicFailures ? "true" : "false")
              << " failure_rollback="
              << (failureRollback ? "true" : "false") << '\n'
              << "average_gpu_seconds=" << averageSeconds
              << " assembly_gpu_seconds=" << averageAssemblySeconds
              << " assembly_fraction=" << assemblyFraction
              << " islands_per_second=" << islandsPerSecond
              << " blocks_per_second=" << blocksPerSecond
              << " assembly_blocks_per_second="
              << assemblyBlocksPerSecond
              << " factor_fmas_per_second=" << factorFMAsPerSecond
              << " contact_iterations_per_second="
              << contactIterationsPerSecond
              << " factor_bytes=" << factorBytes
              << " streamed_operator_bytes=" << streamedOperatorBytes
              << " dense_operator_bytes=" << denseOperatorBytes
              << " factor_stream_to_dense=" << factorAndStreamToDense
              << " max_terms_per_contact=" << maximumTerms
              << " max_dofs_per_term=" << maximumTermDOFs
              << " max_island_blocks=" << maximumIslandBlocks
              << " full_capacity_islands=" << fullCapacityIslands << '\n'
              << "same_command_buffer=true cpu_readback_between_stages=false\n"
              << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "numi-solver-assembly: " << error.what() << '\n';
            return 2;
        }
    }
}

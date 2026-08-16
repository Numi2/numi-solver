cmake_minimum_required(VERSION 3.28)

project(
    NumiSolver
    VERSION 0.1.0
    DESCRIPTION "Apple Metal Temporal Cone contact solver"
    LANGUAGES CXX
)

if(NOT APPLE)
    message(FATAL_ERROR "Numi Solver requires macOS and the Apple Metal toolchain.")
endif()

find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
enable_language(OBJCXX)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

set(NUMI_SOLVER_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(NUMI_SOLVER_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiTemporalCone.air")
set(NUMI_SOLVER_BRAIDED_BAG_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiBraidedBag.air")
set(NUMI_SOLVER_ARTICULATED_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiArticulatedOperator.air")
set(NUMI_SOLVER_INVERSE_ABA_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiArticulatedInverseABA.air")
set(NUMI_SOLVER_METALLIB "${NUMI_SOLVER_SHADER_DIR}/NumiTemporalCone.metallib")
set(NUMI_SOLVER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalWorldContact.metal")
set(
    NUMI_SOLVER_BRAIDED_BAG_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/BraidedBag.metal"
)
set(
    NUMI_SOLVER_ARTICULATED_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedOperator.metal"
)
set(
    NUMI_SOLVER_INVERSE_ABA_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedInverseMass.metal"
)
set(
    NUMI_SOLVER_HEADERS
    include/metalrobo/constraint_ir_shared.h
    include/metalrobo/engine_types.h
    include/metalrobo/gpu_types.h
    include/metalrobo/parallel_aba_shared.h
    include/metalrobo/rod_gpu_shared.h
    include/metalrobo/unified_quality_shared.h
    include/numi/temporal_cone_island.h
    include/numi/temporal_cone_probe.h
    include/numi/braided_bag.h
)

add_custom_command(
    OUTPUT "${NUMI_SOLVER_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_SOLVER_SHADER_DIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_SOURCE}"
        -o "${NUMI_SOLVER_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_BRAIDED_BAG_SOURCE}"
        -o "${NUMI_SOLVER_BRAIDED_BAG_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_ARTICULATED_SOURCE}"
        -o "${NUMI_SOLVER_ARTICULATED_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -DMR_ARTICULATED_ABA_MAX_BODIES=33u
        -DMR_INVERSE_MASS_KERNEL_NAME=numi_articulated_inverse_aba_stream
        -DMR_INVERSE_MASS_STREAMING_RHS=1
        -c "${NUMI_SOLVER_INVERSE_ABA_SOURCE}"
        -o "${NUMI_SOLVER_INVERSE_ABA_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
        "${NUMI_SOLVER_AIR}"
        "${NUMI_SOLVER_BRAIDED_BAG_AIR}"
        "${NUMI_SOLVER_ARTICULATED_AIR}"
        "${NUMI_SOLVER_INVERSE_ABA_AIR}"
        -o "${NUMI_SOLVER_METALLIB}"
    DEPENDS
        "${NUMI_SOLVER_SOURCE}"
        "${NUMI_SOLVER_BRAIDED_BAG_SOURCE}"
        "${NUMI_SOLVER_ARTICULATED_SOURCE}"
        "${NUMI_SOLVER_INVERSE_ABA_SOURCE}"
        ${NUMI_SOLVER_HEADERS}
    COMMENT "Compiling the Temporal Cone solver metallib"
    VERBATIM
)

add_custom_target(numi_temporal_cone ALL DEPENDS "${NUMI_SOLVER_METALLIB}")

add_library(numi_temporal_cone_abi INTERFACE)
target_include_directories(
    numi_temporal_cone_abi
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

add_executable(numi-solver-math tools/temporal_cone_math.mm)
add_dependencies(numi-solver-math numi_temporal_cone)
target_link_libraries(
    numi-solver-math
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-math PRIVATE cxx_std_23)
target_compile_options(numi-solver-math PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_compile_definitions(
    numi-solver-math
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-islands tools/temporal_cone_islands.mm)
add_dependencies(numi-solver-islands numi_temporal_cone)
target_link_libraries(
    numi-solver-islands
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-islands PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-islands
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-islands
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-assembly tools/temporal_cone_assembly.mm)
add_dependencies(numi-solver-assembly numi_temporal_cone)
target_link_libraries(
    numi-solver-assembly
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-assembly PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-assembly
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-assembly
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-rigid tools/temporal_cone_rigid.mm)
add_dependencies(numi-solver-rigid numi_temporal_cone)
target_link_libraries(
    numi-solver-rigid
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-rigid PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-rigid
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-rigid
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-braided-bag tools/braided_bag.mm)
add_dependencies(numi-solver-braided-bag numi_temporal_cone)
target_link_libraries(
    numi-solver-braided-bag
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-braided-bag PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-braided-bag
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-braided-bag
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-cloth-bag tools/cloth_bag.cpp)
target_compile_features(numi-solver-cloth-bag PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-cloth-bag
    PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(numi-solver-articulated tools/temporal_cone_articulated.mm)
add_dependencies(numi-solver-articulated numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-articulated PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-articulated
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(
    numi-solver-articulated-capacity
    tools/temporal_cone_articulated.mm
)
add_dependencies(numi-solver-articulated-capacity numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated-capacity
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-articulated-capacity PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-articulated-capacity
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated-capacity
    PRIVATE
        NUMI_ARTICULATED_ARMATURE_SCALE=16.0
        NUMI_ARTICULATED_CHAIN_DOF=32u
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(
    numi-solver-articulated-conditioning
    tools/temporal_cone_articulated.mm
)
add_dependencies(numi-solver-articulated-conditioning numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated-conditioning
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-articulated-conditioning PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-articulated-conditioning
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated-conditioning
    PRIVATE
        NUMI_ARTICULATED_ARMATURE_SCALE=1.0
        NUMI_ARTICULATED_CHAIN_DOF=32u
        NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE=1
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(
    numi-solver-articulated-zero-armature
    tools/temporal_cone_articulated.mm
)
add_dependencies(numi-solver-articulated-zero-armature numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated-zero-armature
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(
    numi-solver-articulated-zero-armature PRIVATE cxx_std_23
)
target_compile_options(
    numi-solver-articulated-zero-armature
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated-zero-armature
    PRIVATE
        NUMI_ARTICULATED_ARMATURE_SCALE=0.0
        NUMI_ARTICULATED_CHAIN_DOF=32u
        NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE=1
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME temporal_cone.math
        COMMAND $<TARGET_FILE:numi-solver-math>
            --cases 4096
            --replays 2
            --iterations 16
    )
    set_tests_properties(
        temporal_cone.math
        PROPERTIES LABELS "metal;physics;math" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.islands
        COMMAND $<TARGET_FILE:numi-solver-islands>
            --islands 96
            --replays 2
    )
    set_tests_properties(
        temporal_cone.islands
        PROPERTIES LABELS "metal;physics;math;islands;streamed" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.assembly
        COMMAND $<TARGET_FILE:numi-solver-assembly>
            --islands 96
            --replays 2
    )
    set_tests_properties(
        temporal_cone.assembly
        PROPERTIES LABELS "metal;physics;assembly;streamed" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.rigid
        COMMAND $<TARGET_FILE:numi-solver-rigid>
            --islands 96
            --replays 2
    )
    add_test(
        NAME temporal_cone.braided_bag
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 16
            --steps 480
            --replays 2
    )
    set_tests_properties(
        temporal_cone.braided_bag
        PROPERTIES
            LABELS "metal;physics;braid;rigid;contact;streamed"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.braided_bag_freefall
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 1
            --steps 1
            --replays 2
    )
    set_tests_properties(
        temporal_cone.braided_bag_freefall
        PROPERTIES
            LABELS "metal;physics;braid;zero_contact;streamed"
            TIMEOUT 60
    )
    foreach(NUMI_BAG_PRESET IN ITEMS stiff-braid low-cfm high-friction)
        string(REPLACE "-" "_" NUMI_BAG_TEST_SUFFIX "${NUMI_BAG_PRESET}")
        add_test(
            NAME temporal_cone.braided_bag_${NUMI_BAG_TEST_SUFFIX}
            COMMAND $<TARGET_FILE:numi-solver-braided-bag>
                --environments 4
                --steps 480
                --replays 2
                --preset ${NUMI_BAG_PRESET}
        )
        set_tests_properties(
            temporal_cone.braided_bag_${NUMI_BAG_TEST_SUFFIX}
            PROPERTIES
                LABELS "metal;physics;braid;contact;conditioning;streamed"
                TIMEOUT 60
        )
    endforeach()
    add_test(
        NAME temporal_cone.braided_bag_refinement
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 1
            --steps 480
            --replays 2
            --refinement
    )
    set_tests_properties(
        temporal_cone.braided_bag_refinement
        PROPERTIES
            LABELS "metal;physics;braid;contact;refinement;streamed"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.braided_bag_long_horizon
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 4
            --steps 1920
            --replays 2
    )
    set_tests_properties(
        temporal_cone.braided_bag_long_horizon
        PROPERTIES
            LABELS "metal;physics;braid;contact;long_horizon;streamed"
            TIMEOUT 60
    )
    add_test(
        NAME cloth_bag.reference
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --steps 30
            --substeps 4
            --iterations 12
    )
    set_tests_properties(
        cloth_bag.reference
        PROPERTIES
            LABELS "physics;cloth;reference;contact;determinism"
            TIMEOUT 120
    )
    add_test(
        NAME cloth_bag.airborne_spin
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --scenario spin
            --steps 30
            --substeps 4
            --iterations 12
    )
    set_tests_properties(
        cloth_bag.airborne_spin
        PROPERTIES
            LABELS "physics;cloth;reference;contact;determinism;spin"
            TIMEOUT 120
    )
    set_tests_properties(
        temporal_cone.rigid
        PROPERTIES LABELS "metal;physics;rigid;assembly;streamed" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated
        COMMAND $<TARGET_FILE:numi-solver-articulated>
            --islands 96
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated
        PROPERTIES
            LABELS "metal;physics;articulated;assembly;streamed;inverse_aba"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated_capacity
        COMMAND $<TARGET_FILE:numi-solver-articulated-capacity>
            --islands 16
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated_capacity
        PROPERTIES
            LABELS "metal;physics;articulated;capacity;assembly;streamed;inverse_aba"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated_conditioning
        COMMAND $<TARGET_FILE:numi-solver-articulated-conditioning>
            --islands 8
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated_conditioning
        PROPERTIES
            LABELS "metal;physics;articulated;conditioning;rollback;inverse_aba"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated_zero_armature
        COMMAND $<TARGET_FILE:numi-solver-articulated-zero-armature>
            --islands 8
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated_zero_armature
        PROPERTIES
            LABELS "metal;physics;articulated;conditioning;rollback;inverse_aba"
            TIMEOUT 60
    )
endif()

install(FILES "${NUMI_SOLVER_METALLIB}" DESTINATION lib/numi-solver)
install(DIRECTORY include/ DESTINATION include)
install(
    TARGETS
        numi-solver-math
        numi-solver-islands
        numi-solver-assembly
        numi-solver-rigid
        numi-solver-braided-bag
        numi-solver-cloth-bag
        numi-solver-articulated
        numi-solver-articulated-capacity
        numi-solver-articulated-conditioning
        numi-solver-articulated-zero-armature
    RUNTIME DESTINATION bin
)

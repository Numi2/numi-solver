#include <metal_stdlib>
#include "numi/static_contact_geometry.h"
using namespace metal;
kernel void numi_static_geometry_cast(
    device const NumiStaticGeometryInput* inputs [[buffer(0)]],
    device NumiStaticGeometryOutput* outputs [[buffer(1)]],
    constant unsigned& count [[buffer(2)]],
    unsigned index [[thread_position_in_grid]]) {
    if(index>=count)return;
    const auto input=inputs[index];
    const auto hit=numiStaticSceneCast(input.start,input.end,input.radius);
    outputs[index]={hit.time,hit.normal,hit.feature,unsigned(hit.contact),unsigned(hit.valid),0};
}

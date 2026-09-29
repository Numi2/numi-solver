#include "numi/static_contact_geometry.h"
#include "numi/finite_bench_geometry.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <cstring>
#ifdef NUMI_STATIC_GEOMETRY_NATIVE
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace {
numi::bench::Vec3 geometryWide(NumiStaticVec3 p) { return {p.x,p.y,p.z}; }
struct Probe {
    unsigned cases{},face{},edge{},corner{},floor{},miss{},invalid{},mismatches{};
    double maximumImpactPositionError{},maximumNormalError{},maximumContactGap{};
    std::vector<NumiStaticGeometryInput> inputs;
    std::vector<unsigned> expectations;
    bool nativeReplayExact=false;
    double gpuSeconds=0;
    void check(NumiStaticVec3 start,NumiStaticVec3 end,float radius,unsigned expected=99) {
        inputs.push_back({start,end,radius});expectations.push_back(expected);
    }
    void validate(const NumiStaticGeometryInput input,const unsigned expected,const NumiStaticGeometryOutput actual) {
        const auto start=input.start,end=input.end;const float radius=input.radius;
        if(expected==100){invalid+=!actual.valid;mismatches+=actual.valid;return;}
        auto reference=numi::bench::cast({},geometryWide(start),geometryWide(end),radius);
        auto floorHit=numi::bench::castFloor(geometryWide(start),geometryWide(end),radius,-.75);
        if(floorHit.contact&&(!reference.contact||floorHit.time<reference.time))reference={true,floorHit.time,floorHit.normal,5};
        ++cases;
        if(!actual.valid||actual.contact!=reference.contact||
            (expected!=99&&(actual.contact?actual.feature:0)!=expected)) {
            ++mismatches;std::cerr<<"geometry_mismatch case="<<cases<<" native="<<actual.contact
                <<" feature="<<actual.feature<<" reference="<<reference.contact<<" expected="<<expected<<'\n';return;
        }
        if(!actual.contact){++miss;return;}
        face+=actual.feature==1;edge+=actual.feature==2;corner+=actual.feature==3;floor+=actual.feature==5;
        auto motion=geometryWide(end)-geometryWide(start);
        maximumImpactPositionError=std::max(maximumImpactPositionError,
            numi::bench::length(motion)*std::abs(double(actual.time)-reference.time));
        maximumNormalError=std::max(maximumNormalError,numi::bench::length(geometryWide(actual.normal)-reference.normal));
        auto point=geometryWide(start)+motion*actual.time;
        auto sample=numi::bench::sample({},point,radius);
        double gap=std::min(sample.gap,point.z+.75-radius);
        maximumContactGap=std::max(maximumContactGap,std::abs(gap));
    }
    void finish();
};
#ifdef NUMI_STATIC_GEOMETRY_NATIVE
void Probe::finish() {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(device==nil)throw std::runtime_error("native Metal device unavailable");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_STATIC_GEOMETRY_METALLIB] error:&error];
    if(library==nil)throw std::runtime_error("cannot load native static-geometry metallib");
    id<MTLFunction> function=[library newFunctionWithName:@"numi_static_geometry_cast"];
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    if(pipeline==nil)throw std::runtime_error("cannot create native static-geometry pipeline");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> source=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(inputs[0]) options:MTLResourceStorageModeShared];
    std::vector<NumiStaticGeometryOutput> first;
    for(unsigned replay=0;replay<2;++replay) {
        id<MTLBuffer> destination=[device newBufferWithLength:inputs.size()*sizeof(NumiStaticGeometryOutput) options:MTLResourceStorageModeShared];
        if(source==nil||destination==nil||queue==nil)throw std::runtime_error("native static-geometry buffer allocation failed");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(command==nil||encoder==nil)throw std::runtime_error("native static-geometry encoding failed");
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:destination offset:0 atIndex:1];const unsigned count=static_cast<unsigned>(inputs.size());
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        const auto width=std::min<NSUInteger>(256,pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("native static-geometry command failed");
        gpuSeconds+=command.GPUEndTime-command.GPUStartTime;
        const auto* results=static_cast<const NumiStaticGeometryOutput*>(destination.contents);
        if(replay==0)first.assign(results,results+count);
        else nativeReplayExact=std::memcmp(first.data(),results,count*sizeof(results[0]))==0;
    }
    for(std::size_t i=0;i<inputs.size();++i)validate(inputs[i],expectations[i],first[i]);
    std::cout<<"device="<<[device.name UTF8String]<<" native_bit_exact_replay="<<nativeReplayExact
        <<" native_replay_count=2 native_case_count="<<inputs.size()<<" gpu_seconds="<<gpuSeconds<<'\n';
}
#else
void Probe::finish() {
    for(std::size_t i=0;i<inputs.size();++i) {
        auto input=inputs[i];auto hit=numiStaticSceneCast(input.start,input.end,input.radius);
        validate(input,expectations[i],{hit.time,hit.normal,hit.feature,unsigned(hit.contact),unsigned(hit.valid),0});
    }
}
#endif
}
int runStaticGeometryProbe() {
    try {
        Probe p;auto box=numiStaticBench();
        if(numiStaticFloorHeight()!=-.75f)throw std::runtime_error("FP32 floor differs from production room floor");
        for(float radius:{.004f,.04f,.07f}) {
            auto radial=[&](NumiStaticVec3 anchor,NumiStaticVec3 normal,unsigned feature){
                float size=numiStaticLength(normal);normal=numiStaticScale(normal,1/size);
                p.check(numiStaticAdd(anchor,numiStaticScale(normal,radius+.04f)),
                    numiStaticAdd(anchor,numiStaticScale(normal,radius-.01f)),radius,feature);
            };
            for(unsigned axis=0;axis<3;++axis)for(int sign:{-1,1}) {
                NumiStaticVec3 anchor{0,0,-.04f};
                anchor=numiStaticAdd(anchor,numiStaticAxis(axis,
                    numiStaticComponent(sign>0?box.maximum:box.minimum,axis)-numiStaticComponent(anchor,axis)));
                radial(anchor,numiStaticAxis(axis,float(sign)),1);
            }
            for(unsigned axis=0;axis<3;++axis)for(int a:{-1,1})for(int b:{-1,1}) {
                unsigned first=(axis+1)%3,second=(axis+2)%3;NumiStaticVec3 anchor{0,0,-.04f};
                anchor=numiStaticAdd(anchor,numiStaticAxis(first,numiStaticComponent(a>0?box.maximum:box.minimum,first)-numiStaticComponent(anchor,first)));
                anchor=numiStaticAdd(anchor,numiStaticAxis(second,numiStaticComponent(b>0?box.maximum:box.minimum,second)-numiStaticComponent(anchor,second)));
                radial(anchor,numiStaticAdd(numiStaticAxis(first,float(a)),numiStaticAxis(second,float(b))),2);
            }
            for(int x:{-1,1})for(int y:{-1,1})for(int z:{-1,1})
                radial({x>0?box.maximum.x:box.minimum.x,y>0?box.maximum.y:box.minimum.y,z>0?box.maximum.z:box.minimum.z},
                    {float(x),float(y),float(z)},3);
            p.check({.75f+.9f*radius,.5f+.9f*radius,.2f},{.75f+.9f*radius,.5f+.9f*radius,-.2f},radius,0);
            p.check({1,0,-.5f},{1,0,-1.f},radius,5);
            p.check({0,0,radius},{0,0,radius-.00001f},radius,4);
            // A long sweep through a narrow edge exercises FP32 quadratic
            // cancellation, not just short trajectories close to the bench.
            p.check({-.01f,.5f+.5f*radius,8},{-.01f,.5f+.5f*radius,-1},radius,2);
        }
        std::mt19937 generator(94812);std::uniform_real_distribution<float> xy(-1.1f,1.1f),z(-1,1),r(.004f,.1f);
        for(unsigned i=0;i<2000;++i){
            float radius=r(generator);NumiStaticVec3 start{xy(generator),xy(generator),z(generator)},end{xy(generator),xy(generator),z(generator)};
            const auto sample=numi::bench::sample({},geometryWide(start),radius);
            const double independentGap=std::min(sample.gap,double(start.z)+.75-radius);
            if(independentGap>.0001)p.check(start,end,radius);
        }
        for(float radius:{0.f,-.1f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
            p.check({0,0,1},{0,0,-.1f},radius,100);
        }
        auto nan=std::numeric_limits<float>::quiet_NaN();
        for(auto start:std::array<NumiStaticVec3,3>{{{nan,0,1},{0,0,-.04f},{1,0,-1.4f}}}){
            p.check(start,{0,0,0},.004f,100);
        }
        p.finish();
        bool pass=!p.mismatches&&p.face&&p.edge&&p.corner&&p.floor&&p.miss&&p.invalid==7&&
            p.maximumImpactPositionError<2e-6&&p.maximumNormalError<.002&&p.maximumContactGap<2e-6;
#ifdef NUMI_STATIC_GEOMETRY_NATIVE
        pass=pass&&p.nativeReplayExact;
        std::cout<<"backend=Apple_Metal_FP32_geometry independent_reference=production_FP64\n"
#else
        std::cout<<"backend=CPU_shared_FP32_geometry independent_reference=production_FP64\n"
#endif
            <<"cases="<<p.cases<<" faces="<<p.face<<" rounded_edges="<<p.edge<<" rounded_vertices="<<p.corner
            <<" floor="<<p.floor<<" misses="<<p.miss<<" rejected_invalid="<<p.invalid<<" mismatches="<<p.mismatches<<'\n'
            <<"maximum_impact_position_error_m="<<p.maximumImpactPositionError<<" maximum_normal_error="<<p.maximumNormalError
            <<" maximum_contact_gap_m="<<p.maximumContactGap<<'\n'
#ifdef NUMI_STATIC_GEOMETRY_NATIVE
            <<"static_geometry_native_probe_pass="<<pass<<" full_scene=NOT_QUALIFIED\n";
#else
            <<"static_geometry_host_probe_pass="<<pass<<" native_Metal_execution=NOT_RUN full_scene=NOT_QUALIFIED\n";
#endif
        return pass?0:1;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}

int main() {
#ifdef NUMI_STATIC_GEOMETRY_NATIVE
    @autoreleasepool { return runStaticGeometryProbe(); }
#else
    return runStaticGeometryProbe();
#endif
}

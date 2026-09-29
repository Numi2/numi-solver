#pragma once

// Scalar FP32 geometry shared by the native Metal path and its host probe.
// The independent production FP64 implementation remains finite_bench_geometry.h.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
inline bool numiStaticFinite(float x) { return metal::isfinite(x); }
inline float numiStaticSqrt(float x) { return metal::sqrt(x); }
inline float numiStaticAbs(float x) { return metal::abs(x); }
inline float numiStaticFma(float x,float y,float z) { return metal::fma(x,y,z); }
#else
#include <cmath>
inline bool numiStaticFinite(float x) { return std::isfinite(x); }
inline float numiStaticSqrt(float x) { return std::sqrt(x); }
inline float numiStaticAbs(float x) { return std::abs(x); }
inline float numiStaticFma(float x,float y,float z) { return std::fma(x,y,z); }
#endif

struct NumiStaticVec3 { float x,y,z; };
inline float numiStaticComponent(NumiStaticVec3 a,unsigned axis) { return axis==0?a.x:axis==1?a.y:a.z; }
inline NumiStaticVec3 numiStaticAxis(unsigned axis,float sign) { return {axis==0?sign:0,axis==1?sign:0,axis==2?sign:0}; }
inline NumiStaticVec3 numiStaticAdd(NumiStaticVec3 a,NumiStaticVec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline NumiStaticVec3 numiStaticSub(NumiStaticVec3 a,NumiStaticVec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline NumiStaticVec3 numiStaticScale(NumiStaticVec3 a,float s) { return {a.x*s,a.y*s,a.z*s}; }
inline float numiStaticDot(NumiStaticVec3 a,NumiStaticVec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline bool numiStaticFinite3(NumiStaticVec3 a) { return numiStaticFinite(a.x)&&numiStaticFinite(a.y)&&numiStaticFinite(a.z); }
inline float numiStaticMin(float a,float b) { return a<b?a:b; }
inline float numiStaticMax(float a,float b) { return a>b?a:b; }
inline float numiStaticClamp(float a,float lo,float hi) { return numiStaticMin(hi,numiStaticMax(lo,a)); }
inline float numiStaticLength(NumiStaticVec3 a) { return numiStaticSqrt(numiStaticDot(a,a)); }
inline float numiStaticTolerance(NumiStaticVec3 a,NumiStaticVec3 b) {
    return 8.0f*1.1920928955078125e-7f*numiStaticMax(1.0f,numiStaticMax(numiStaticLength(a),numiStaticLength(b)));
}
struct NumiStaticBox { NumiStaticVec3 minimum,maximum; };
struct NumiStaticSample { float gap; NumiStaticVec3 normal; bool valid; };
struct NumiStaticHit { float time; NumiStaticVec3 normal; unsigned feature; bool contact,valid; };
inline NumiStaticBox numiStaticBench() { return {{-.75f,-.5f,-.08f},{.75f,.5f,0}}; }
inline float numiStaticFloorHeight() { return -.75f; }
inline bool numiStaticValid(NumiStaticBox box,NumiStaticVec3 point,float radius) {
    return numiStaticFinite3(box.minimum)&&numiStaticFinite3(box.maximum)&&numiStaticFinite3(point)&&
        numiStaticFinite(radius)&&radius>0&&box.maximum.x>box.minimum.x&&
        box.maximum.y>box.minimum.y&&box.maximum.z>box.minimum.z&&
        numiStaticFinite3(numiStaticSub(box.maximum,box.minimum));
}
inline NumiStaticSample numiStaticBoxSample(NumiStaticBox box,NumiStaticVec3 p,float radius) {
    if(!numiStaticValid(box,p,radius))return {0,{0,0,0},false};
    NumiStaticVec3 nearest{numiStaticClamp(p.x,box.minimum.x,box.maximum.x),
        numiStaticClamp(p.y,box.minimum.y,box.maximum.y),numiStaticClamp(p.z,box.minimum.z,box.maximum.z)};
    NumiStaticVec3 offset=numiStaticSub(p,nearest);float distance=numiStaticLength(offset);
    if(!numiStaticFinite(distance))return {0,{0,0,0},false};
    if(distance>0)return {distance-radius,numiStaticScale(offset,1/distance),true};
    float nearestFace=3.402823466e38f;NumiStaticVec3 normal{0,0,0};
    // Same deterministic tie rule as the independent FP64 box sample.
    for(int axis=2;axis>=0;--axis)for(int side=0;side<2;++side){
        float sign=side==0?1.0f:-1.0f;
        float gap=side==0?numiStaticComponent(box.maximum,axis)-numiStaticComponent(p,axis):
            numiStaticComponent(p,axis)-numiStaticComponent(box.minimum,axis);
        if(gap<nearestFace){nearestFace=gap;normal=numiStaticAxis(axis,sign);}
    }
    return {-nearestFace-radius,normal,true};
}
inline NumiStaticSample numiStaticSceneSample(NumiStaticVec3 p,float radius) {
    NumiStaticSample s=numiStaticBoxSample(numiStaticBench(),p,radius);
    float floorGap=p.z-(numiStaticFloorHeight()+radius);
    if(!numiStaticFinite(floorGap))s.valid=false;
    if(s.valid&&floorGap<s.gap)s={floorGap,{0,0,1},true};
    return s;
}
struct NumiStaticRoots { float first,second; bool valid; };
inline NumiStaticRoots numiStaticRadialRoots(NumiStaticVec3 offset,NumiStaticVec3 motion,float radius) {
    float a=numiStaticDot(motion,motion);
    if(!numiStaticFinite(a))return {2,2,false};
    if(!(a>1e-30f))return {2,2,true};
    float middle=-numiStaticDot(offset,motion)/a;
    NumiStaticVec3 closest{numiStaticFma(motion.x,middle,offset.x),
        numiStaticFma(motion.y,middle,offset.y),numiStaticFma(motion.z,middle,offset.z)};
    float radial=radius*radius-numiStaticDot(closest,closest);
    if(!numiStaticFinite(middle)||!numiStaticFinite(radial))return {2,2,false};
    if(radial<0)return {2,2,true};
    float halfWidth=numiStaticSqrt(radial/a);
    return {middle-halfWidth,middle+halfWidth,numiStaticFinite(halfWidth)};
}
inline NumiStaticHit numiStaticAccept(NumiStaticHit hit,float time,NumiStaticVec3 normal,
                                     unsigned feature,NumiStaticVec3 delta,float tolerance) {
    if(!numiStaticFinite(time)||!numiStaticFinite3(normal)){hit.valid=false;return hit;}
    if(time<0||time>1||numiStaticDot(delta,normal)>=-tolerance)return hit;
    if(!hit.contact||time<hit.time)return {time,normal,feature,true,hit.valid};
    return hit;
}
inline NumiStaticHit numiStaticBoxCast(NumiStaticBox box,NumiStaticVec3 start,NumiStaticVec3 end,float radius) {
    NumiStaticHit hit{1,{0,0,0},0,false,true};
    if(!numiStaticValid(box,start,radius)||!numiStaticValid(box,end,radius)){hit.valid=false;return hit;}
    NumiStaticVec3 delta=numiStaticSub(end,start);float tolerance=numiStaticTolerance(start,end);
    NumiStaticSample initial=numiStaticBoxSample(box,start,radius);
    if(!initial.valid||!numiStaticFinite3(delta)||!numiStaticFinite(tolerance)||initial.gap< -tolerance){hit.valid=false;return hit;}
    for(unsigned axis=0;axis<3;++axis)
        if(numiStaticMin(numiStaticComponent(start,axis),numiStaticComponent(end,axis))>numiStaticComponent(box.maximum,axis)+radius||
           numiStaticMax(numiStaticComponent(start,axis),numiStaticComponent(end,axis))<numiStaticComponent(box.minimum,axis)-radius)return hit;
    // Closing tolerance is relative to displacement, not world-coordinate ULP:
    // a small inward correction at existing support must still be blocked.
    float closing=1e-7f*numiStaticLength(delta);
    if(initial.gap<=tolerance&&numiStaticDot(delta,initial.normal)<-closing)
        return {0,initial.normal,4,true,true};
    for(unsigned axis=0;axis<3;++axis)for(int sign=-1;sign<=1;sign+=2){
        float motion=numiStaticComponent(delta,axis);
        if(motion*sign>=0)continue;
        float plane=numiStaticComponent(sign>0?box.maximum:box.minimum,axis)+sign*radius;
        float time=(plane-numiStaticComponent(start,axis))/motion;
        if(time<0||time>1)continue;
        NumiStaticVec3 p=numiStaticAdd(start,numiStaticScale(delta,time));bool face=true;
        for(unsigned other=0;other<3;++other)if(other!=axis)
            face=face&&numiStaticComponent(p,other)>=numiStaticComponent(box.minimum,other)-tolerance&&
                numiStaticComponent(p,other)<=numiStaticComponent(box.maximum,other)+tolerance;
        if(face)hit=numiStaticAccept(hit,time,numiStaticAxis(axis,float(sign)),1,delta,closing);
    }
    for(unsigned axis=0;axis<3;++axis){
        unsigned first=(axis+1)%3,second=(axis+2)%3;
        for(int s1=-1;s1<=1;s1+=2)for(int s2=-1;s2<=1;s2+=2){
            NumiStaticVec3 edge=numiStaticAdd(numiStaticScale(numiStaticAxis(first,1),numiStaticComponent(s1>0?box.maximum:box.minimum,first)),
                numiStaticScale(numiStaticAxis(second,1),numiStaticComponent(s2>0?box.maximum:box.minimum,second)));
            NumiStaticVec3 offset=numiStaticSub(start,edge);offset=numiStaticSub(offset,numiStaticAxis(axis,numiStaticComponent(offset,axis)));
            NumiStaticVec3 motion=numiStaticSub(delta,numiStaticAxis(axis,numiStaticComponent(delta,axis)));
            auto roots=numiStaticRadialRoots(offset,motion,radius);
            hit.valid=hit.valid&&roots.valid;
            for(unsigned root=0;root<2;++root){
                float time=root==0?roots.first:roots.second;if(time<0||time>1)continue;
                NumiStaticVec3 p=numiStaticAdd(start,numiStaticScale(delta,time)),normal=numiStaticSub(p,edge);
                normal=numiStaticSub(normal,numiStaticAxis(axis,numiStaticComponent(normal,axis)));
                if(numiStaticComponent(p,axis)<numiStaticComponent(box.minimum,axis)-tolerance||
                    numiStaticComponent(p,axis)>numiStaticComponent(box.maximum,axis)+tolerance||
                    numiStaticComponent(normal,first)*s1< -tolerance||numiStaticComponent(normal,second)*s2< -tolerance)continue;
                float size=numiStaticLength(normal);
                if(size>0)hit=numiStaticAccept(hit,time,numiStaticScale(normal,1/size),2,delta,closing);
            }
        }
    }
    for(int sx=-1;sx<=1;sx+=2)for(int sy=-1;sy<=1;sy+=2)for(int sz=-1;sz<=1;sz+=2){
        NumiStaticVec3 cornerPoint{sx>0?box.maximum.x:box.minimum.x,sy>0?box.maximum.y:box.minimum.y,sz>0?box.maximum.z:box.minimum.z};
        NumiStaticVec3 offset=numiStaticSub(start,cornerPoint);
        auto roots=numiStaticRadialRoots(offset,delta,radius);
        hit.valid=hit.valid&&roots.valid;
        for(unsigned root=0;root<2;++root){
            float time=root==0?roots.first:roots.second;if(time<0||time>1)continue;
            NumiStaticVec3 normal=numiStaticSub(numiStaticAdd(start,numiStaticScale(delta,time)),cornerPoint);
            if(normal.x*sx< -tolerance||normal.y*sy< -tolerance||normal.z*sz< -tolerance)continue;
            float size=numiStaticLength(normal);
            if(size>0)hit=numiStaticAccept(hit,time,numiStaticScale(normal,1/size),3,delta,closing);
        }
    }
    return hit;
}
inline NumiStaticHit numiStaticSceneCast(NumiStaticVec3 start,NumiStaticVec3 end,float radius) {
    auto hit=numiStaticBoxCast(numiStaticBench(),start,end,radius);
    float support=numiStaticFloorHeight()+radius;
    if(!numiStaticFinite(support)||start.z<support-numiStaticTolerance(start,end))hit.valid=false;
    if(end.z<support&&end.z<start.z){
        float time=numiStaticClamp((support-start.z)/(end.z-start.z),0,1);
        if(!hit.contact||time<hit.time)hit={time,{0,0,1},5,true,hit.valid};
    }
    return hit;
}

// Pointer-free probe wire records; scalar triples avoid float3 ABI padding.
struct NumiStaticGeometryInput { NumiStaticVec3 start,end; float radius; };
struct NumiStaticGeometryOutput {
    float time; NumiStaticVec3 normal;
    unsigned feature,contact,valid,reserved;
};
#ifndef __METAL_VERSION__
static_assert(sizeof(NumiStaticGeometryInput)==28);
static_assert(sizeof(NumiStaticGeometryOutput)==32);
#endif

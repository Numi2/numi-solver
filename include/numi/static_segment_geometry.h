#pragma once

#include "static_contact_geometry.h"

// Whole axial-segment geometry, shared by the CPU and Metal FP32 paths.
// Compile without fast-math: the outward-rounded supporting-plane certificate
// relies on IEEE nextafter and on separate, correctly rounded scalar operations.
// The signed sample minimizes the box SDF (minus radius) over every point of the
// segment, including interior face-envelope intersections. Endpoint tests alone
// do not establish capsule/box separation.
#ifdef __METAL_VERSION__
#define NUMI_SEGMENT_THREAD thread
inline float numiSegmentDown(float x) {
    // Enclose possible flush-to-zero results on Metal rather than relying on
    // preservation of subnormals by either the compiler or the GPU.
    if(x> -1.1754943508222875e-38f&&x<1.1754943508222875e-38f)return -1.1754943508222875e-38f;
    float y=metal::nextafter(x,-INFINITY);
    return y> -1.1754943508222875e-38f&&y<1.1754943508222875e-38f?-1.1754943508222875e-38f:y;
}
inline float numiSegmentUp(float x) {
    if(x> -1.1754943508222875e-38f&&x<1.1754943508222875e-38f)return 1.1754943508222875e-38f;
    float y=metal::nextafter(x,INFINITY);
    return y> -1.1754943508222875e-38f&&y<1.1754943508222875e-38f?1.1754943508222875e-38f:y;
}
#else
#define NUMI_SEGMENT_THREAD
inline float numiSegmentDown(float x) {
    if(x> -1.1754943508222875e-38f&&x<1.1754943508222875e-38f)return -1.1754943508222875e-38f;
    float y=std::nextafter(x,-INFINITY);
    return y> -1.1754943508222875e-38f&&y<1.1754943508222875e-38f?-1.1754943508222875e-38f:y;
}
inline float numiSegmentUp(float x) {
    if(x> -1.1754943508222875e-38f&&x<1.1754943508222875e-38f)return 1.1754943508222875e-38f;
    float y=std::nextafter(x,INFINITY);
    return y> -1.1754943508222875e-38f&&y<1.1754943508222875e-38f?1.1754943508222875e-38f:y;
}
#endif

struct NumiStaticSegmentSample {
    float gap, parameter, clearanceLowerBound;
    NumiStaticVec3 normal;
    unsigned feature;
    bool valid;
};

// Status distinguishes an exhausted conservative advancement from a proven
// clear motion. A caller must never treat valid=false as a collision-free step.
enum NumiStaticSegmentCastStatus : unsigned {
    NumiStaticSegmentClear = 0,
    NumiStaticSegmentContact = 1,
    NumiStaticSegmentInvalid = 2,
    NumiStaticSegmentIterationLimit = 3,
    NumiStaticSegmentInitialPenetration = 4,
    NumiStaticSegmentUnresolved = 5
};
struct NumiStaticSegmentHit {
    float time, parameter;
    NumiStaticVec3 normal;
    float gap;
    unsigned feature, iterations;
    bool contact, valid;
    unsigned status;
};
struct NumiStaticSegmentIntervalPoint { NumiStaticVec3 point, lower, upper; };

inline NumiStaticVec3 numiStaticSegmentPoint(NumiStaticVec3 a,NumiStaticVec3 delta,float u) {
    return {numiStaticFma(delta.x,u,a.x),numiStaticFma(delta.y,u,a.y),numiStaticFma(delta.z,u,a.z)};
}
inline unsigned numiStaticSegmentFeature(NumiStaticBox box,NumiStaticVec3 p) {
    unsigned outside=0;
    for(unsigned axis=0;axis<3;++axis)
        outside+=numiStaticComponent(p,axis)<numiStaticComponent(box.minimum,axis)||
                 numiStaticComponent(p,axis)>numiStaticComponent(box.maximum,axis);
    return outside?outside:4;
}
inline void numiStaticSegmentConsider(NumiStaticBox box,NumiStaticVec3 a,NumiStaticVec3 b,
                                     float radius,float u,NUMI_SEGMENT_THREAD NumiStaticSegmentSample& best) {
    if(!numiStaticFinite(u)){best.valid=false;return;}
    u=numiStaticClamp(u,0,1);
    auto p=u==0?a:u==1?b:numiStaticSegmentPoint(a,numiStaticSub(b,a),u);
    auto s=numiStaticBoxSample(box,p,radius);
    if(!s.valid){best.valid=false;return;}
    // Minimum parameter is the deterministic tie rule for flat minima.
    if(s.gap<best.gap||(s.gap==best.gap&&u<best.parameter)) {
        best.gap=s.gap;best.parameter=u;best.normal=s.normal;
        best.feature=numiStaticSegmentFeature(box,p);
    }
}

// Lower bound on dot(p - support(box,n),n) for a point interval. The corner is
// the exact supporting vertex of the represented FP32 box for represented n.
// Each operation is rounded outward; this remains a valid half-space bound even
// when the approximate closest-feature normal is not the exact minimizer normal.
inline float numiStaticSegmentLowerPlane(NumiStaticBox box,NumiStaticVec3 lower,
                                        NumiStaticVec3 upper,NumiStaticVec3 n) {
    float value=0;
    for(unsigned axis=0;axis<3;++axis) {
        float normal=numiStaticComponent(n,axis);
        if(normal==0)continue;
        float corner=numiStaticComponent(normal>0?box.maximum:box.minimum,axis);
        float delta=normal>0?numiSegmentDown(numiStaticComponent(lower,axis)-corner):
                             numiSegmentUp(numiStaticComponent(upper,axis)-corner);
        value=numiSegmentDown(value+numiSegmentDown(delta*normal));
    }
    return value;
}
inline float numiStaticSegmentUpperLength(NumiStaticVec3 n) {
    float sum=0;
    for(unsigned axis=0;axis<3;++axis) {
        float component=numiStaticComponent(n,axis);
        if(component!=0)sum=numiSegmentUp(sum+numiSegmentUp(component*component));
    }
    return numiSegmentUp(numiStaticSqrt(sum));
}
inline float numiStaticSegmentBoxClearanceBound(NumiStaticBox box,
    NumiStaticVec3 lowerA,NumiStaticVec3 upperA,NumiStaticVec3 lowerB,NumiStaticVec3 upperB,
    NumiStaticVec3 normal,float radius) {
    float magnitude=numiStaticSegmentUpperLength(normal);
    float projection=numiStaticMin(numiStaticSegmentLowerPlane(box,lowerA,upperA,normal),
                                  numiStaticSegmentLowerPlane(box,lowerB,upperB,normal));
    if(!numiStaticFinite(magnitude)||!(magnitude>0)||!numiStaticFinite(projection))return -INFINITY;
    float distance=projection>0?numiSegmentDown(projection/magnitude):0;
    return numiSegmentDown(distance-radius);
}
inline float numiStaticSegmentFloorClearanceBound(NumiStaticVec3 lowerA,NumiStaticVec3 lowerB,
                                                 float radius) {
    return numiSegmentDown(numiSegmentDown(numiStaticMin(lowerA.z,lowerB.z)-numiStaticFloorHeight())-radius);
}

inline NumiStaticSegmentSample numiStaticSegmentBoxSample(NumiStaticBox box,
                                                         NumiStaticVec3 a,NumiStaticVec3 b,float radius) {
    NumiStaticSegmentSample best{3.402823466e38f,0,-INFINITY,{0,0,0},0,true};
    if(!numiStaticValid(box,a,radius)||!numiStaticValid(box,b,radius)) {best.valid=false;return best;}
    auto delta=numiStaticSub(b,a);
    if(!numiStaticFinite3(delta)){best.valid=false;return best;}
    // At most six coordinate-boundary intersections, plus the two endpoints.
    float knots[8]={0,1,0,0,0,0,0,0};unsigned count=2;
    for(unsigned axis=0;axis<3;++axis) {
        float slope=numiStaticComponent(delta,axis);
        if(slope==0)continue;
        for(unsigned side=0;side<2;++side) {
            float boundary=numiStaticComponent(side?box.maximum:box.minimum,axis);
            float u=(boundary-numiStaticComponent(a,axis))/slope;
            if(!numiStaticFinite(u)){best.valid=false;return best;}
            if(u>0&&u<1) {
                unsigned i=count++;
                while(i>0&&knots[i-1]>u){knots[i]=knots[i-1];--i;}
                knots[i]=u;
            }
        }
    }
    for(unsigned i=0;i<count;++i)numiStaticSegmentConsider(box,a,b,radius,knots[i],best);
    for(unsigned interval=0;interval+1<count;++interval) {
        float left=knots[interval],right=knots[interval+1];
        if(!(right>left))continue;
        auto middle=numiStaticSegmentPoint(a,delta,left+(right-left)*.5f);
        float quadratic=0,linear=0;unsigned outside=0;
        for(unsigned axis=0;axis<3;++axis) {
            float p=numiStaticComponent(middle,axis);
            if(p<numiStaticComponent(box.minimum,axis)||p>numiStaticComponent(box.maximum,axis)) {
                float boundary=numiStaticComponent(p<numiStaticComponent(box.minimum,axis)?box.minimum:box.maximum,axis);
                float offset=numiStaticComponent(a,axis)-boundary,slope=numiStaticComponent(delta,axis);
                quadratic=numiStaticFma(slope,slope,quadratic);
                linear=numiStaticFma(offset,slope,linear);++outside;
            }
        }
        if(!numiStaticFinite(quadratic)||!numiStaticFinite(linear)){best.valid=false;return best;}
        if(outside) {
            if(quadratic>0)numiStaticSegmentConsider(box,a,b,radius,
                numiStaticClamp(-linear/quadratic,left,right),best);
        } else {
            // Inside, distance is -min(six affine face clearances). The maximum
            // clearance occurs at an interval endpoint or a pairwise crossing.
            float offset[6],slope[6];
            for(unsigned axis=0;axis<3;++axis) {
                offset[2*axis]=numiStaticComponent(a,axis)-numiStaticComponent(box.minimum,axis);
                slope[2*axis]=numiStaticComponent(delta,axis);
                offset[2*axis+1]=numiStaticComponent(box.maximum,axis)-numiStaticComponent(a,axis);
                slope[2*axis+1]=-numiStaticComponent(delta,axis);
            }
            for(unsigned i=0;i<6;++i)for(unsigned j=i+1;j<6;++j)if(slope[i]!=slope[j]) {
                float u=(offset[j]-offset[i])/(slope[i]-slope[j]);
                if(!numiStaticFinite(u)){best.valid=false;return best;}
                if(u>=left&&u<=right)numiStaticSegmentConsider(box,a,b,radius,u,best);
            }
        }
    }
    best.clearanceLowerBound=numiStaticSegmentBoxClearanceBound(box,a,a,b,b,best.normal,radius);
    best.valid=best.valid&&numiStaticFinite(best.gap)&&numiStaticFinite(best.parameter)&&
               numiStaticFinite3(best.normal)&&numiStaticFinite(best.clearanceLowerBound);
    return best;
}
inline NumiStaticSegmentSample numiStaticSegmentSceneSample(NumiStaticVec3 a,NumiStaticVec3 b,float radius) {
    auto s=numiStaticSegmentBoxSample(numiStaticBench(),a,b,radius);
    float floorGap=numiStaticMin(a.z,b.z)-(numiStaticFloorHeight()+radius);
    float floorBound=numiStaticSegmentFloorClearanceBound(a,b,radius);
    if(!numiStaticFinite(floorGap)||!numiStaticFinite(floorBound)){s.valid=false;return s;}
    s.clearanceLowerBound=numiStaticMin(s.clearanceLowerBound,floorBound);
    if(floorGap<s.gap) {s.gap=floorGap;s.parameter=b.z<a.z?1:0;s.normal={0,0,1};s.feature=5;}
    return s;
}

inline NumiStaticSegmentIntervalPoint numiStaticSegmentInterpolate(NumiStaticVec3 start,
                                                                  NumiStaticVec3 end,float t) {
    if(t==0)return {start,start,start};
    if(t==1)return {end,end,end};
    NumiStaticSegmentIntervalPoint out{};
    for(unsigned axis=0;axis<3;++axis) {
        float a=numiStaticComponent(start,axis),b=numiStaticComponent(end,axis);
        float lower=numiSegmentDown(a+numiSegmentDown(numiSegmentDown(b-a)*t));
        float upper=numiSegmentUp(a+numiSegmentUp(numiSegmentUp(b-a)*t));
        float point=numiStaticFma(b-a,t,a);
        if(axis==0){out.point.x=point;out.lower.x=lower;out.upper.x=upper;}
        if(axis==1){out.point.y=point;out.lower.y=lower;out.upper.y=upper;}
        if(axis==2){out.point.z=point;out.lower.z=lower;out.upper.z=upper;}
    }
    return out;
}
inline float numiStaticSegmentUpperMotion(NumiStaticVec3 start,NumiStaticVec3 end) {
    NumiStaticVec3 bound{};
    for(unsigned axis=0;axis<3;++axis) {
        float a=numiStaticComponent(start,axis),b=numiStaticComponent(end,axis);
        float component=numiStaticMax(numiStaticAbs(numiSegmentDown(b-a)),numiStaticAbs(numiSegmentUp(b-a)));
        if(axis==0)bound.x=component;if(axis==1)bound.y=component;if(axis==2)bound.z=component;
    }
    return numiStaticSegmentUpperLength(bound);
}
inline NumiStaticSegmentHit numiStaticSegmentSceneCast(NumiStaticVec3 a0,NumiStaticVec3 b0,
    NumiStaticVec3 a1,NumiStaticVec3 b1,float radius,float contactTolerance=1e-7f,unsigned maximumIterations=128) {
    NumiStaticSegmentHit hit{1,0,{0,0,0},0,0,0,false,false,NumiStaticSegmentInvalid};
    if(!numiStaticFinite(contactTolerance)||contactTolerance<0||maximumIterations==0||
       !numiStaticValid(numiStaticBench(),a1,radius)||!numiStaticValid(numiStaticBench(),b1,radius))return hit;
    auto initialBox=numiStaticSegmentBoxSample(numiStaticBench(),a0,b0,radius);
    auto initial=numiStaticSegmentSceneSample(a0,b0,radius);
    if(!initial.valid||!initialBox.valid)return hit;
    hit.parameter=initial.parameter;hit.normal=initial.normal;hit.gap=initial.gap;hit.feature=initial.feature;
    if(initial.gap< -contactTolerance){hit.time=0;hit.status=NumiStaticSegmentInitialPenetration;return hit;}
    // Every point of a moving segment lies in the convex hull of these four
    // endpoints. A single certified separating plane therefore proves this
    // collider cannot be crossed, including tangent/outward existing support.
    float boxHullBound=numiStaticMin(
        numiStaticSegmentBoxClearanceBound(numiStaticBench(),a0,a0,b0,b0,initialBox.normal,radius),
        numiStaticSegmentBoxClearanceBound(numiStaticBench(),a1,a1,b1,b1,initialBox.normal,radius));
    float floorHullBound=numiStaticMin(numiStaticSegmentFloorClearanceBound(a0,b0,radius),
                                      numiStaticSegmentFloorClearanceBound(a1,b1,radius));
    if(!numiStaticFinite(boxHullBound)||!numiStaticFinite(floorHullBound))return hit;
    bool boxClear=boxHullBound>=-contactTolerance,floorClear=floorHullBound>=-contactTolerance;
    if(boxClear&&floorClear){hit.valid=true;hit.status=NumiStaticSegmentClear;return hit;}
    float speed=numiStaticMax(numiStaticSegmentUpperMotion(a0,a1),numiStaticSegmentUpperMotion(b0,b1));
    if(!numiStaticFinite(speed))return hit;
    if(!(speed>0)){hit.status=NumiStaticSegmentUnresolved;return hit;}
    float time=0;
    for(unsigned iteration=0;iteration<maximumIterations;++iteration) {
        hit.iterations=iteration+1;
        auto a=numiStaticSegmentInterpolate(a0,a1,time),b=numiStaticSegmentInterpolate(b0,b1,time);
        if(!numiStaticFinite3(a.lower)||!numiStaticFinite3(a.upper)||!numiStaticFinite3(b.lower)||!numiStaticFinite3(b.upper))return hit;
        auto box=numiStaticSegmentBoxSample(numiStaticBench(),a.point,b.point,radius);
        if(!box.valid)return hit;
        float boxBound=numiStaticSegmentBoxClearanceBound(numiStaticBench(),a.lower,a.upper,b.lower,b.upper,box.normal,radius);
        float floorBound=numiStaticSegmentFloorClearanceBound(a.lower,b.lower,radius);
        if(!numiStaticFinite(boxBound)||!numiStaticFinite(floorBound))return hit;
        float bound=boxClear?floorBound:floorClear?boxBound:numiStaticMin(boxBound,floorBound);
        bool floor=!floorClear&&(boxClear||floorBound<boxBound);
        auto witness=floor?numiStaticSegmentSceneSample(a.point,b.point,radius):box;
        // For a floor witness, do not accidentally replace it with a box which
        // is globally certified clear at the caller's explicit tolerance.
        if(floor){witness.gap=numiStaticMin(a.point.z,b.point.z)-(numiStaticFloorHeight()+radius);
            witness.parameter=b.point.z<a.point.z?1:0;witness.normal={0,0,1};witness.feature=5;}
        hit.time=time;hit.parameter=witness.parameter;hit.normal=witness.normal;hit.gap=witness.gap;hit.feature=witness.feature;
        if(bound<=contactTolerance) {
            // A poor separating plane may be inconclusive despite distant
            // geometry. Publish failure, never a false collision-free result.
            if(witness.gap>contactTolerance+16*1.1920928955078125e-7f*
               numiStaticMax(1.0f,numiStaticMax(numiStaticLength(a.point),numiStaticLength(b.point)))) {
                hit.status=NumiStaticSegmentUnresolved;return hit;
            }
            hit.contact=true;hit.valid=true;hit.status=NumiStaticSegmentContact;return hit;
        }
        // The unsigned distance from any material point to the fixed collider
        // is 1-Lipschitz; Hausdorff segment motion is bounded by the larger
        // endpoint displacement. Outward rounding makes this an upper speed
        // and a lower clearance, so the advancement cannot overstep contact.
        float step=numiSegmentDown(bound/speed);
        float remaining=numiSegmentUp(1-time);
        if(step>=remaining){hit.time=1;hit.valid=true;hit.status=NumiStaticSegmentClear;return hit;}
        float next=numiSegmentDown(time+step);
        if(!(next>time)){hit.status=NumiStaticSegmentUnresolved;return hit;}
        time=next;
    }
    hit.status=NumiStaticSegmentIterationLimit;return hit;
}

// Velocity qualification is over the complete supported axial interval, rather
// than at an arbitrary tie-breaking distance witness. Return feature=0 and
// gap=FLT_MAX if there is no point with gap <= tolerance; valid still describes
// whether the query was well formed. Planar contact velocity is affine in u.
// Rounded-feature velocity is a quadratic divided by sqrt(quadratic); its
// stationary points are the roots of a cubic, isolated in bounded scalar loops.
struct NumiStaticSegmentCubicRoots {float roots[3];unsigned count;bool valid;};
inline float numiStaticSegmentPolynomial(float c0,float c1,float c2,float c3,float u) {
    return numiStaticFma(numiStaticFma(numiStaticFma(c3,u,c2),u,c1),u,c0);
}
inline NumiStaticSegmentCubicRoots numiStaticSegmentCubic(float c0,float c1,float c2,float c3,
                                                        float left,float right) {
    NumiStaticSegmentCubicRoots out{{0,0,0},0,true};
    float scale=numiStaticMax(numiStaticMax(numiStaticAbs(c0),numiStaticAbs(c1)),
                             numiStaticMax(numiStaticAbs(c2),numiStaticAbs(c3)));
    if(!numiStaticFinite(scale)){out.valid=false;return out;}
    if(!(scale>0))return out;
    c0/=scale;c1/=scale;c2/=scale;c3/=scale;
    float knots[4]={left,right,0,0};unsigned count=2;
    float aa=3*c3,bb=2*c2,cc=c1;
    float critical[2]={2,2};
    if(aa!=0) {
        float discriminant=numiStaticFma(-4*aa,cc,bb*bb);
        if(discriminant>=0) {
            float root=numiStaticSqrt(discriminant);
            float q=-.5f*(bb+(bb<0?-root:root));
            if(q!=0){critical[0]=q/aa;critical[1]=cc/q;}
            else critical[0]=-bb/(2*aa);
        }
    } else if(bb!=0)critical[0]=-cc/bb;
    for(unsigned j=0;j<2;++j)if(critical[j]>left&&critical[j]<right) {
        unsigned i=count++;
        while(i>0&&knots[i-1]>critical[j]){knots[i]=knots[i-1];--i;}
        knots[i]=critical[j];
    }
    for(unsigned interval=0;interval+1<count;++interval) {
        float lo=knots[interval],hi=knots[interval+1];
        float fLo=numiStaticSegmentPolynomial(c0,c1,c2,c3,lo);
        float fHi=numiStaticSegmentPolynomial(c0,c1,c2,c3,hi);
        if(!numiStaticFinite(fLo)||!numiStaticFinite(fHi)){out.valid=false;return out;}
        if(fLo==0&&out.count<3)out.roots[out.count++]=lo;
        if((fLo<0&&fHi>0)||(fLo>0&&fHi<0)) {
            for(unsigned iteration=0;iteration<32;++iteration) {
                float middle=lo+(hi-lo)*.5f;
                if(!(middle>lo&&middle<hi))break;
                float f=numiStaticSegmentPolynomial(c0,c1,c2,c3,middle);
                if((f<0&&fLo<0)||(f>0&&fLo>0)){lo=middle;fLo=f;}else hi=middle;
            }
            if(out.count<3)out.roots[out.count++]=lo+(hi-lo)*.5f;
        }
        if(interval+2==count&&fHi==0&&out.count<3)out.roots[out.count++]=knots[interval+1];
    }
    return out;
}
inline void numiStaticSegmentVelocityConsider(NumiStaticBox box,NumiStaticVec3 a,NumiStaticVec3 b,
    NumiStaticVec3 va,NumiStaticVec3 vb,float radius,float tolerance,float u,
    NUMI_SEGMENT_THREAD NumiStaticSegmentSample& best,NUMI_SEGMENT_THREAD float& bestClosing) {
    if(!numiStaticFinite(u)){best.valid=false;return;}
    u=numiStaticClamp(u,0,1);
    auto p=u==0?a:u==1?b:numiStaticSegmentPoint(a,numiStaticSub(b,a),u);
    auto s=numiStaticBoxSample(box,p,radius);
    if(!s.valid){best.valid=false;return;}
    auto velocity=numiStaticSegmentPoint(va,numiStaticSub(vb,va),u);
    float closing=numiStaticDot(velocity,s.normal);
    if(!numiStaticFinite(closing)){best.valid=false;return;}
    if(s.gap<=tolerance&&(closing<bestClosing||(closing==bestClosing&&u<best.parameter))) {
        bestClosing=closing;best.gap=s.gap;best.parameter=u;best.normal=s.normal;
        best.feature=numiStaticSegmentFeature(box,p);
    }
}
inline NumiStaticSegmentSample numiStaticSegmentBoxVelocitySample(NumiStaticBox box,
    NumiStaticVec3 a,NumiStaticVec3 b,NumiStaticVec3 va,NumiStaticVec3 vb,float radius,float tolerance) {
    NumiStaticSegmentSample best{3.402823466e38f,0,0,{0,0,1},0,true};float bestClosing=3.402823466e38f;
    if(!numiStaticValid(box,a,radius)||!numiStaticValid(box,b,radius)||!numiStaticFinite3(va)||!numiStaticFinite3(vb)||
       !numiStaticFinite(tolerance)||tolerance<0){best.valid=false;return best;}
    // A yarn axis inside the solid has a nonunique/intersecting contact
    // manifold; a single SDF normal is an escape witness, not a velocity
    // constraint. Require positional repair before publishing velocity contact.
    auto position=numiStaticSegmentBoxSample(box,a,b,radius);
    if(!position.valid||position.gap< -radius){best.valid=false;return best;}
    auto delta=numiStaticSub(b,a),dv=numiStaticSub(vb,va);
    if(!numiStaticFinite3(delta)||!numiStaticFinite3(dv)){best.valid=false;return best;}
    float knots[8]={0,1,0,0,0,0,0,0};unsigned count=2;
    for(unsigned axis=0;axis<3;++axis) {
        float slope=numiStaticComponent(delta,axis);
        if(slope==0)continue;
        for(unsigned side=0;side<2;++side) {
            float boundary=numiStaticComponent(side?box.maximum:box.minimum,axis);
            float u=(boundary-numiStaticComponent(a,axis))/slope;
            if(!numiStaticFinite(u)){best.valid=false;return best;}
            if(u>0&&u<1){unsigned i=count++;while(i>0&&knots[i-1]>u){knots[i]=knots[i-1];--i;}knots[i]=u;}
        }
    }
    for(unsigned i=0;i<count;++i) {
        numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,knots[i],best,bestClosing);
        float before=numiSegmentDown(knots[i]),after=numiSegmentUp(knots[i]);
        if(before>=0)numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,before,best,bestClosing);
        if(after<=1)numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,after,best,bestClosing);
    }
    for(unsigned interval=0;interval+1<count;++interval) {
        float left=knots[interval],right=knots[interval+1];if(!(right>left))continue;
        auto middle=numiStaticSegmentPoint(a,delta,left+(right-left)*.5f);
        NumiStaticVec3 offset{},slope{};unsigned outside=0;
        for(unsigned axis=0;axis<3;++axis) {
            float p=numiStaticComponent(middle,axis);
            if(p<numiStaticComponent(box.minimum,axis)||p>numiStaticComponent(box.maximum,axis)) {
                float boundary=numiStaticComponent(p<numiStaticComponent(box.minimum,axis)?box.minimum:box.maximum,axis);
                float o=numiStaticComponent(a,axis)-boundary,d=numiStaticComponent(delta,axis);
                if(axis==0){offset.x=o;slope.x=d;}if(axis==1){offset.y=o;slope.y=d;}if(axis==2){offset.z=o;slope.z=d;}
                ++outside;
            }
        }
        if(outside) {
            float q0=numiStaticDot(offset,offset),q1=2*numiStaticDot(offset,slope),q2=numiStaticDot(slope,slope);
            if(q2>0)numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,
                numiStaticClamp(-.5f*q1/q2,left,right),best,bestClosing);
            if(q2>0) {
                float minimum=numiStaticClamp(-.5f*q1/q2,left,right);
                auto p=numiStaticSegmentPoint(a,delta,minimum);
                auto support=numiStaticBoxSample(box,p,radius);
                if(!support.valid){best.valid=false;return best;}
                if(support.gap<=tolerance)for(unsigned side=0;side<2;++side) {
                    float outsideU=side?right:left,insideU=minimum;
                    auto edgePoint=outsideU==0?a:outsideU==1?b:numiStaticSegmentPoint(a,delta,outsideU);
                    auto edgeSample=numiStaticBoxSample(box,edgePoint,radius);
                    if(!edgeSample.valid){best.valid=false;return best;}
                    if(edgeSample.gap<=tolerance)insideU=outsideU;
                    else for(unsigned iteration=0;iteration<32;++iteration) {
                        float u=outsideU+(insideU-outsideU)*.5f;
                        if(u==outsideU||u==insideU)break;
                        auto candidate=numiStaticBoxSample(box,numiStaticSegmentPoint(a,delta,u),radius);
                        if(!candidate.valid){best.valid=false;return best;}
                        if(candidate.gap<=tolerance)insideU=u;else outsideU=u;
                    }
                    numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,insideU,best,bestClosing);
                }
            }
            float c0=numiStaticDot(va,offset),c1=numiStaticDot(dv,offset)+numiStaticDot(va,slope),c2=numiStaticDot(dv,slope);
            auto stationary=numiStaticSegmentCubic(
                c1*q0-.5f*c0*q1,2*c2*q0+.5f*c1*q1-c0*q2,1.5f*c2*q1,c2*q2,left,right);
            if(!stationary.valid){best.valid=false;return best;}
            for(unsigned i=0;i<stationary.count;++i)
                numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,stationary.roots[i],best,bestClosing);
        } else {
            float offsets[6],slopes[6];
            for(unsigned axis=0;axis<3;++axis) {
                offsets[2*axis]=numiStaticComponent(a,axis)-numiStaticComponent(box.minimum,axis);
                slopes[2*axis]=numiStaticComponent(delta,axis);
                offsets[2*axis+1]=numiStaticComponent(box.maximum,axis)-numiStaticComponent(a,axis);
                slopes[2*axis+1]=-numiStaticComponent(delta,axis);
            }
            for(unsigned i=0;i<6;++i)for(unsigned j=i+1;j<6;++j)if(slopes[i]!=slopes[j]) {
                float u=(offsets[j]-offsets[i])/(slopes[i]-slopes[j]);
                if(!numiStaticFinite(u)){best.valid=false;return best;}
                if(u>=left&&u<=right) {
                    numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,u,best,bestClosing);
                    float before=numiSegmentDown(u),after=numiSegmentUp(u);
                    if(before>=left)numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,before,best,bestClosing);
                    if(after<=right)numiStaticSegmentVelocityConsider(box,a,b,va,vb,radius,tolerance,after,best,bestClosing);
                }
            }
        }
    }
    if(best.feature)best.clearanceLowerBound=numiStaticSegmentBoxClearanceBound(box,a,a,b,b,best.normal,radius);
    best.valid=best.valid&&numiStaticFinite(best.gap)&&numiStaticFinite(best.clearanceLowerBound);
    return best;
}
inline NumiStaticSegmentSample numiStaticSegmentSceneVelocitySample(NumiStaticVec3 a,NumiStaticVec3 b,
    NumiStaticVec3 va,NumiStaticVec3 vb,float radius,float tolerance) {
    auto best=numiStaticSegmentBoxVelocitySample(numiStaticBench(),a,b,va,vb,radius,tolerance);
    if(!best.valid)return best;
    float bestClosing=best.feature?numiStaticDot(numiStaticSegmentPoint(va,numiStaticSub(vb,va),best.parameter),best.normal):3.402823466e38f;
    float candidates[3]={0,1,2};
    if(b.z!=a.z)candidates[2]=(numiStaticFloorHeight()+radius+tolerance-a.z)/(b.z-a.z);
    for(unsigned i=0;i<3;++i) {
        float u=candidates[i];if(!(u>=0&&u<=1))continue;
        auto p=u==0?a:u==1?b:numiStaticSegmentPoint(a,numiStaticSub(b,a),u);
        float gap=p.z-(numiStaticFloorHeight()+radius);
        float closing=numiStaticFma(vb.z-va.z,u,va.z);
        if(!numiStaticFinite(gap)||!numiStaticFinite(closing)){best.valid=false;return best;}
        if(gap<=tolerance&&(closing<bestClosing||(closing==bestClosing&&u<best.parameter))) {
            bestClosing=closing;best.gap=gap;best.parameter=u;best.normal={0,0,1};best.feature=5;
            best.clearanceLowerBound=numiStaticSegmentFloorClearanceBound(a,b,radius);
        }
    }
    return best;
}

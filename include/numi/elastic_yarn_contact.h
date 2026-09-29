#pragma once
#include "numi/static_contact_geometry.h"

// CPU-qualified five-owner normal-contact mechanics. No native ABI/GPU integration.
// Five independent dynamic owning nodes: yarn0,yarn1,triangle0..2.
struct NumiElasticYarnBody { NumiStaticVec3 position, velocity; float mass; };
struct NumiElasticYarnWitness {
    float yarnParameter, triangleWeights[3], distance, gap;
    NumiStaticVec3 yarnPoint, trianglePoint, normal;
    bool valid, normalDefined, axisCrossing;
};
// Invalid/rejected results retain all input velocities and zero accepted impulse.
// energyUpper describes the attempted candidate when status is EnergyRejected.
struct NumiElasticYarnResult { NumiStaticVec3 velocities[5]; float impulse,effectiveInverseMass,closingBefore,closingAfter,energyUpper;
    unsigned status; bool valid; };
enum { NumiElasticYarnInactive=0,NumiElasticYarnSolved=1,NumiElasticYarnSeparated=2,NumiElasticYarnInvalid=3,NumiElasticYarnAmbiguousNormal=4,NumiElasticYarnEnergyRejected=5 };
inline NumiStaticVec3 numiElasticYarnCross(NumiStaticVec3 a,NumiStaticVec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline NumiStaticVec3 numiElasticYarnLerp(NumiStaticVec3 a,NumiStaticVec3 b,float t){return {numiStaticFma(b.x-a.x,t,a.x),numiStaticFma(b.y-a.y,t,a.y),numiStaticFma(b.z-a.z,t,a.z)};}
inline float numiElasticYarnUp(float x){
#ifdef __METAL_VERSION__
    return metal::nextafter(x,INFINITY);
#else
    return std::nextafter(x,INFINITY);
#endif
}
inline float numiElasticYarnDown(float x){
#ifdef __METAL_VERSION__
    return metal::nextafter(x,-INFINITY);
#else
    return std::nextafter(x,-INFINITY);
#endif
}
inline void numiElasticYarnConsider(NumiStaticVec3 a,NumiStaticVec3 b,const NumiStaticVec3 t[3],float u,float v,float w,NumiElasticYarnWitness& best){
    if(!numiStaticFinite(u)||!numiStaticFinite(v)||!numiStaticFinite(w)||u<0||u>1||v<0||w<0||v+w>1)return;
    float weights[3]={1-(v+w),v,w};
    auto yp=numiElasticYarnLerp(a,b,u);auto tp=numiStaticAdd(numiStaticScale(t[0],weights[0]),numiStaticAdd(numiStaticScale(t[1],weights[1]),numiStaticScale(t[2],weights[2])));
    auto delta=numiStaticSub(yp,tp);float d2=numiStaticDot(delta,delta);
    if(!numiStaticFinite(d2))return;
    float previous=best.distance*best.distance;
    if(!best.valid||d2<previous||(d2==previous&&u<best.yarnParameter)){
        best.yarnParameter=u;for(unsigned i=0;i<3;++i)best.triangleWeights[i]=weights[i];
        best.yarnPoint=yp;best.trianglePoint=tp;best.distance=numiStaticSqrt(d2);best.valid=true;
    }
}
inline void numiElasticYarnPointTriangle(NumiStaticVec3 p,float u,NumiStaticVec3 a,NumiStaticVec3 b,const NumiStaticVec3 t[3],NumiElasticYarnWitness& best){
    auto ab=numiStaticSub(t[1],t[0]),ac=numiStaticSub(t[2],t[0]),ap=numiStaticSub(p,t[0]);
    float d1=numiStaticDot(ab,ap),d2=numiStaticDot(ac,ap);
    if(d1<=0&&d2<=0){numiElasticYarnConsider(a,b,t,u,0,0,best);return;}
    auto bp=numiStaticSub(p,t[1]);float d3=numiStaticDot(ab,bp),d4=numiStaticDot(ac,bp);
    if(d3>=0&&d4<=d3){numiElasticYarnConsider(a,b,t,u,1,0,best);return;}
    float vc=numiStaticFma(d1,d4,-d3*d2);
    if(vc<=0&&d1>=0&&d3<=0){numiElasticYarnConsider(a,b,t,u,d1/(d1-d3),0,best);return;}
    auto cp=numiStaticSub(p,t[2]);float d5=numiStaticDot(ab,cp),d6=numiStaticDot(ac,cp);
    if(d6>=0&&d5<=d6){numiElasticYarnConsider(a,b,t,u,0,1,best);return;}
    float vb=numiStaticFma(d5,d2,-d1*d6);
    if(vb<=0&&d2>=0&&d6<=0){numiElasticYarnConsider(a,b,t,u,0,d2/(d2-d6),best);return;}
    float va=numiStaticFma(d3,d6,-d5*d4);
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){float w=(d4-d3)/((d4-d3)+(d5-d6));numiElasticYarnConsider(a,b,t,u,1-w,w,best);return;}
    float denom=va+vb+vc;if(!(denom>0))return;
    numiElasticYarnConsider(a,b,t,u,vb/denom,vc/denom,best);
}
inline void numiElasticYarnSegmentEdge(NumiStaticVec3 a,NumiStaticVec3 b,const NumiStaticVec3 t[3],unsigned first,unsigned second,NumiElasticYarnWitness& best){
    auto d1=numiStaticSub(b,a),d2=numiStaticSub(t[second],t[first]),r=numiStaticSub(a,t[first]);
    float aa=numiStaticDot(d1,d1),ee=numiStaticDot(d2,d2),bb=numiStaticDot(d1,d2),cc=numiStaticDot(d1,r),ff=numiStaticDot(d2,r);
    float u=0,v=0;
    if(aa==0){v=numiStaticClamp(ff/ee,0,1);}
    else{
        auto cross=numiElasticYarnCross(d1,d2);float determinant=numiStaticDot(cross,cross);
        if(determinant>32*1.1920928955078125e-7f*1.1920928955078125e-7f*aa*ee)
            u=numiStaticClamp(numiStaticDot(numiElasticYarnCross(d2,r),cross)/determinant,0,1);
        v=(bb*u+ff)/ee;
        if(v<0){v=0;u=numiStaticClamp(-cc/aa,0,1);}
        else if(v>1){v=1;u=numiStaticClamp((bb-cc)/aa,0,1);}
    }
    float weights[3]={0,0,0};weights[first]=1-v;weights[second]=v;
    numiElasticYarnConsider(a,b,t,u,weights[1],weights[2],best);
}
inline NumiElasticYarnWitness numiElasticYarnClosestSegmentTriangle(NumiStaticVec3 a,NumiStaticVec3 b,const NumiStaticVec3 t[3],float radius){
    NumiElasticYarnWitness best{};
    if(!numiStaticFinite3(a)||!numiStaticFinite3(b)||!numiStaticFinite(radius)||!(radius>0))return best;
    for(unsigned i=0;i<3;++i)if(!numiStaticFinite3(t[i]))return best;
    auto e1=numiStaticSub(t[1],t[0]),e2=numiStaticSub(t[2],t[0]),n=numiElasticYarnCross(e1,e2);
    float n2=numiStaticDot(n,n),e2max=numiStaticMax(numiStaticDot(e1,e1),numiStaticDot(e2,e2));
    if(!numiStaticFinite(n2)||!(n2>128*1.1920928955078125e-7f*1.1920928955078125e-7f*e2max*e2max))return best;
    numiElasticYarnPointTriangle(a,0,a,b,t,best);numiElasticYarnPointTriangle(b,1,a,b,t,best);
    for(unsigned i=0;i<3;++i)numiElasticYarnSegmentEdge(a,b,t,i,(i+1)%3,best);
    auto d=numiStaticSub(b,a);float slope=numiStaticDot(n,d);
    if(slope!=0){
        float u=-numiStaticDot(n,numiStaticSub(a,t[0]))/slope;
        if(u>=0&&u<=1){
            auto p=numiStaticSub(numiElasticYarnLerp(a,b,u),t[0]);
            float v=numiStaticDot(numiElasticYarnCross(p,e2),n)/n2,w=numiStaticDot(numiElasticYarnCross(e1,p),n)/n2;
            if(v>=0&&w>=0&&v+w<=1){numiElasticYarnConsider(a,b,t,u,v,w,best);best.axisCrossing=true;}
        }
    }
    if(!best.valid)return best;
    best.gap=best.distance-radius;
    // A crossing/coincident axis has no unique unsigned-distance normal. It is
    // a positional/CCD failure state, not an ordinary velocity impact.
    float scale=numiStaticMax(1.0f,numiStaticMax(numiStaticLength(best.yarnPoint),numiStaticLength(best.trianglePoint)));
    best.normalDefined=!best.axisCrossing&&best.distance>8*1.1920928955078125e-7f*scale;
    if(best.normalDefined)best.normal=numiStaticScale(numiStaticSub(best.yarnPoint,best.trianglePoint),1/best.distance);
    return best;
}
inline float numiElasticYarnEnergyUpper(const NumiElasticYarnBody b[5],const NumiStaticVec3 v[5]){
    float total=0;bool changed=false;
    for(unsigned i=0;i<5;++i)for(unsigned k=0;k<3;++k){
        float x=numiStaticComponent(b[i].velocity,k),y=numiStaticComponent(v[i],k);if(x==y)continue;changed=true;
        float dl=numiElasticYarnDown(y-x),du=numiElasticYarnUp(y-x),sl=numiElasticYarnDown(y+x),su=numiElasticYarnUp(y+x);
        float product=numiStaticMax(numiStaticMax(dl*sl,dl*su),numiStaticMax(du*sl,du*su));
        float term=numiElasticYarnUp(numiElasticYarnUp(product)*b[i].mass);total=numiElasticYarnUp(total+term);
    }
    return changed?numiElasticYarnUp(total*.5f):0;
}
// An explicit material-point witness is useful when a parallel contact has a
// flat interval of closest points. This only verifies coordinate-domain and
// represented geometry validity; it DOES NOT prove closest-point/manifold
// membership. The caller must validate physical manifold membership before
// supplying these coordinates. The normal is rebuilt from owning positions.
inline NumiElasticYarnWitness numiElasticYarnWitnessAtCallerValidatedCoordinates(NumiStaticVec3 a,NumiStaticVec3 b,const NumiStaticVec3 t[3],float radius,float u,float v,float w){
    auto closest=numiElasticYarnClosestSegmentTriangle(a,b,t,radius);NumiElasticYarnWitness witness{};
    if(!closest.valid)return witness;
    numiElasticYarnConsider(a,b,t,u,v,w,witness);if(!witness.valid)return witness;
    witness.gap=witness.distance-radius;witness.axisCrossing=closest.axisCrossing;
    float scale=numiStaticMax(1.0f,numiStaticMax(numiStaticLength(witness.yarnPoint),numiStaticLength(witness.trianglePoint)));
    witness.normalDefined=!witness.axisCrossing&&witness.distance>8*1.1920928955078125e-7f*scale;
    if(witness.normalDefined)witness.normal=numiStaticScale(numiStaticSub(witness.yarnPoint,witness.trianglePoint),1/witness.distance);
    return witness;
}
namespace numiElasticYarnDetail {
inline NumiElasticYarnResult resolveWitnessNormal(const NumiElasticYarnBody bodies[5],const NumiElasticYarnWitness& witness,float contactTolerance=1e-7f){
    NumiElasticYarnResult out{};out.status=NumiElasticYarnInvalid;for(unsigned i=0;i<5;++i)out.velocities[i]=bodies[i].velocity;
    for(unsigned i=0;i<5;++i)if(!numiStaticFinite3(bodies[i].position)||!numiStaticFinite3(bodies[i].velocity)||
        !numiStaticFinite(bodies[i].mass)||!(bodies[i].mass>0))return out;
    if(!numiStaticFinite(contactTolerance)||contactTolerance<0)return out;
    if(!witness.valid)return out;
    if(witness.gap>contactTolerance){out.status=NumiElasticYarnSeparated;out.valid=true;return out;}
    if(!witness.normalDefined){out.status=NumiElasticYarnAmbiguousNormal;return out;}
    float weights[5]={1-witness.yarnParameter,witness.yarnParameter,-witness.triangleWeights[0],-witness.triangleWeights[1],-witness.triangleWeights[2]};
    NumiStaticVec3 gradients[5];
    for(unsigned i=0;i<5;++i){gradients[i]=numiStaticScale(witness.normal,weights[i]);
        out.closingBefore+=numiStaticDot(gradients[i],bodies[i].velocity);
        out.effectiveInverseMass+=numiStaticDot(gradients[i],gradients[i])/bodies[i].mass;}
    if(!numiStaticFinite(out.effectiveInverseMass)||!(out.effectiveInverseMass>0)||!numiStaticFinite(out.closingBefore))return out;
    if(out.closingBefore>=0){out.status=NumiElasticYarnInactive;out.valid=true;out.closingAfter=out.closingBefore;return out;}
    const float candidateImpulse=-out.closingBefore/out.effectiveInverseMass;
    if(!numiStaticFinite(candidateImpulse)||!(candidateImpulse>0))return out;
    NumiStaticVec3 candidateVelocities[5];float candidateClosing=0;
    for(unsigned i=0;i<5;++i){float response=candidateImpulse/bodies[i].mass;auto g=gradients[i],v=bodies[i].velocity;
        candidateVelocities[i]={numiStaticFma(g.x,response,v.x),numiStaticFma(g.y,response,v.y),numiStaticFma(g.z,response,v.z)};
        if(!numiStaticFinite3(candidateVelocities[i]))return out;
        candidateClosing+=numiStaticDot(g,candidateVelocities[i]);}
    out.energyUpper=numiElasticYarnEnergyUpper(bodies,candidateVelocities);
    if(!numiStaticFinite(out.energyUpper)||out.energyUpper>0){out.status=NumiElasticYarnEnergyRejected;return out;}
    if(!numiStaticFinite(candidateClosing))return out;
    out.impulse=candidateImpulse;out.closingAfter=candidateClosing;
    for(unsigned i=0;i<5;++i)out.velocities[i]=candidateVelocities[i];
    out.valid=true;out.status=NumiElasticYarnSolved;return out;
}
} // namespace numiElasticYarnDetail
inline NumiElasticYarnResult numiElasticYarnResolveNormal(const NumiElasticYarnBody bodies[5],float radius,float contactTolerance=1e-7f){
    NumiStaticVec3 t[3]={bodies[2].position,bodies[3].position,bodies[4].position};
    auto witness=numiElasticYarnClosestSegmentTriangle(bodies[0].position,bodies[1].position,t,radius);
    return numiElasticYarnDetail::resolveWitnessNormal(bodies,witness,contactTolerance);
}
inline NumiElasticYarnResult numiElasticYarnResolveCallerValidatedNormal(const NumiElasticYarnBody bodies[5],float radius,float u,float v,float w,float contactTolerance=1e-7f){
    NumiStaticVec3 t[3]={bodies[2].position,bodies[3].position,bodies[4].position};
    auto witness=numiElasticYarnWitnessAtCallerValidatedCoordinates(bodies[0].position,bodies[1].position,t,radius,u,v,w);
    return numiElasticYarnDetail::resolveWitnessNormal(bodies,witness,contactTolerance);
}
// Publishers must check this predicate before publishing state or impulse/work
// ledgers. Invalid/rejected state is unchanged, but diagnostics are not ledgers.
inline bool numiElasticYarnCanPublish(const NumiElasticYarnResult& result){
    return result.valid&&(result.status==NumiElasticYarnSolved||result.status==NumiElasticYarnInactive||result.status==NumiElasticYarnSeparated);
}

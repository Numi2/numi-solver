#include "numi/static_segment_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

// This qualification reference does not import the production FP64 bench helper
// or the segment feature-interval algorithm. Exterior distance uses the twelve
// physical box edges, six face intersections and endpoint-to-box distances.
// Interior penetration uses a one-dimensional shrunk-box feasibility problem.
namespace {
struct D3 {
    double x,y,z;
    double& operator[](unsigned i) { return i==0?x:i==1?y:z; }
    double operator[](unsigned i) const { return i==0?x:i==1?y:z; }
};
D3 wide(NumiStaticVec3 p){return {p.x,p.y,p.z};}
D3 operator+(D3 a,D3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
D3 operator-(D3 a,D3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
D3 operator*(D3 a,double b){return {a.x*b,a.y*b,a.z*b};}
double dot(D3 a,D3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
double length(D3 a){return std::hypot(a.x,a.y,a.z);}
struct DBox {D3 low,high;};
DBox wide(NumiStaticBox box){return {wide(box.minimum),wide(box.maximum)};}
struct Reference {double gap,u,unsignedGap;D3 normal;};
struct Closest {double distance,u;};
D3 clampBox(D3 p,DBox box) {
    for(unsigned i=0;i<3;++i)p[i]=std::clamp(p[i],box.low[i],box.high[i]);
    return p;
}
Closest edges(D3 a,D3 b,D3 c,D3 d) {
    auto first=b-a,second=d-c,r=a-c;
    double aa=dot(first,first),ee=dot(second,second),f=dot(second,r),s=0,t=0;
    if(aa<=1e-30&&ee<=1e-30)return {length(r),0};
    if(aa<=1e-30)t=std::clamp(f/ee,0.0,1.0);
    else {
        double c0=dot(first,r);
        if(ee<=1e-30)s=std::clamp(-c0/aa,0.0,1.0);
        else {
            double bb=dot(first,second),denominator=aa*ee-bb*bb;
            if(denominator>1e-28*aa*ee)s=std::clamp((bb*f-c0*ee)/denominator,0.0,1.0);
            t=(bb*s+f)/ee;
            if(t<0){t=0;s=std::clamp(-c0/aa,0.0,1.0);}
            if(t>1){t=1;s=std::clamp((bb-c0)/aa,0.0,1.0);}
        }
    }
    return {length(r+first*s-second*t),s};
}
bool interiorInterval(D3 a,D3 b,DBox box,double shrink,double& low,double& high) {
    low=0;high=1;auto d=b-a;
    for(unsigned i=0;i<3;++i) {
        double lo=box.low[i]+shrink,hi=box.high[i]-shrink;
        if(lo>hi)return false;
        if(d[i]==0){if(a[i]<lo||a[i]>hi)return false;continue;}
        double first=(lo-a[i])/d[i],second=(hi-a[i])/d[i];
        if(first>second)std::swap(first,second);
        low=std::max(low,first);high=std::min(high,second);
        if(low>high)return false;
    }
    return true;
}
Reference boxReference(D3 a,D3 b,double radius,DBox box) {
    double low=0,high=1;
    if(interiorInterval(a,b,box,0,low,high)) {
        double left=0,right=.5*std::min({box.high.x-box.low.x,box.high.y-box.low.y,box.high.z-box.low.z});
        for(unsigned i=0;i<80;++i) {
            double middle=(left+right)*.5,l,h;
            if(interiorInterval(a,b,box,middle,l,h))left=middle;else right=middle;
        }
        interiorInterval(a,b,box,left,low,high);double u=std::clamp(low,0.0,1.0);D3 p=a+(b-a)*u;
        double depth=std::numeric_limits<double>::infinity();D3 normal{};
        for(int axis=2;axis>=0;--axis)for(int side:{1,-1}) {
            double face=side>0?box.high[axis]-p[axis]:p[axis]-box.low[axis];
            if(face<depth){depth=face;normal={};normal[axis]=side;}
        }
        return {-left-radius,u,-radius,normal};
    }
    Closest best{length(a-clampBox(a,box)),0};
    double endpoint=length(b-clampBox(b,box));if(endpoint<best.distance)best={endpoint,1};
    for(unsigned axis=0;axis<3;++axis) {
        unsigned first=(axis+1)%3,second=(axis+2)%3;
        for(unsigned s=0;s<2;++s)for(unsigned t=0;t<2;++t) {
            D3 c{},d{};c[axis]=box.low[axis];d[axis]=box.high[axis];
            c[first]=d[first]=s?box.high[first]:box.low[first];
            c[second]=d[second]=t?box.high[second]:box.low[second];
            auto q=edges(a,b,c,d);if(q.distance<best.distance)best=q;
        }
    }
    D3 p=a+(b-a)*best.u,offset=p-clampBox(p,box);double norm=length(offset);
    return {best.distance-radius,best.u,best.distance-radius,norm>0?offset*(1/norm):D3{0,0,1}};
}
Reference sceneReference(D3 a,D3 b,double radius,DBox box) {
    auto r=boxReference(a,b,radius,box);
    double floor=std::min(a.z,b.z)+.75-radius;
    r.unsignedGap=std::min(r.unsignedGap,floor);
    if(floor<r.gap){r.gap=floor;r.u=b.z<a.z?1:0;r.normal={0,0,1};}
    return r;
}
double pointGap(D3 p,double radius,DBox box) {
    auto offset=p-clampBox(p,box);double distance=length(offset),gap;
    if(distance>0)gap=distance-radius;
    else gap=-std::min({p.x-box.low.x,box.high.x-p.x,p.y-box.low.y,box.high.y-p.y,p.z-box.low.z,box.high.z-p.z})-radius;
    return std::min(gap,p.z+.75-radius);
}
struct Test {
    unsigned samples{},sampleFailures{},certificates{},certificateFailures{},sweeps{},sweepFailures{},
        clear{},contact{},unresolved{},rejected{},adversarial{},negativeControls{},tangentDepartures{},velocityQueries{},velocityFailures{},velocityRejected{},floorVelocityRegressions{},floorVelocityNegativeControls{},deformingSweeps{},deformingOracleUnresolved{},deformingSweepUnresolved{},deformingMisses{};
    double maxGapError{},maxWitnessError{},maxLowerBoundExcess{},maxImpactGap{},maxImpactPositionError{},maxVelocitySuboptimality{};
    unsigned maximumIterations{};
    void require(bool condition,const char* what) {
        if(!condition)throw std::runtime_error(what);
    }
    void sample(NumiStaticVec3 a,NumiStaticVec3 b,float radius,NumiStaticBox box=numiStaticBench(),bool scene=true) {
        ++samples;auto actual=scene?numiStaticSegmentSceneSample(a,b,radius):numiStaticSegmentBoxSample(box,a,b,radius);
        auto reference=scene?sceneReference(wide(a),wide(b),radius,wide(box)):boxReference(wide(a),wide(b),radius,wide(box));
        double error=std::abs(actual.gap-reference.gap);
        double witness=(scene?pointGap(wide(a)+(wide(b)-wide(a))*actual.parameter,radius,wide(box)):
            boxReference(wide(a)+(wide(b)-wide(a))*actual.parameter,wide(a)+(wide(b)-wide(a))*actual.parameter,radius,wide(box)).gap)-reference.gap;
        maxGapError=std::max(maxGapError,error);maxWitnessError=std::max(maxWitnessError,witness);
        if(!actual.valid||actual.parameter<0||actual.parameter>1||error>1e-6||witness>1e-6||
           std::abs(numiStaticLength(actual.normal)-1)>2e-6) {
            ++sampleFailures;if(sampleFailures<=8)std::cerr<<"sample_failure="<<samples<<" gap="<<actual.gap<<" ref="<<reference.gap
                <<" u="<<actual.parameter<<" witness="<<witness<<" valid="<<actual.valid<<'\n';
        }
        ++certificates;double excess=actual.clearanceLowerBound-reference.unsignedGap;
        maxLowerBoundExcess=std::max(maxLowerBoundExcess,excess);
        if(excess>2e-13||!actual.valid){++certificateFailures;if(certificateFailures<=8)std::cerr<<"certificate_failure="<<samples<<" excess="<<excess<<'\n';}
    }
    void sweep(NumiStaticVec3 a0,NumiStaticVec3 b0,NumiStaticVec3 a1,NumiStaticVec3 b1,float radius,
               double referenceTime,bool expectedContact,bool expectResolved=true,bool exactTranslation=false) {
        ++sweeps;auto actual=numiStaticSegmentSceneCast(a0,b0,a1,b1,radius);
        maximumIterations=std::max(maximumIterations,actual.iterations);
        if(!actual.valid)++unresolved;else if(actual.contact)++contact;else ++clear;
        bool failure=actual.valid&&actual.contact!=expectedContact;
        if(expectResolved&&!actual.valid)failure=true;
        if(actual.valid&&actual.contact) {
            auto a=wide(a0)+(wide(a1)-wide(a0))*actual.time,b=wide(b0)+(wide(b1)-wide(b0))*actual.time;
            double gap=sceneReference(a,b,radius,wide(numiStaticBench())).gap;
            maxImpactGap=std::max(maxImpactGap,std::abs(gap));
            if(gap< -1e-6||actual.time>referenceTime+2e-6)failure=true;
            double position=std::max(length(wide(a1)-wide(a0)),length(wide(b1)-wide(b0)))*std::abs(actual.time-referenceTime);
            if(exactTranslation)maxImpactPositionError=std::max(maxImpactPositionError,position);
        }
        if(failure){++sweepFailures;if(sweepFailures<=8)std::cerr<<"sweep_failure="<<sweeps<<" time="<<actual.time<<" reference="<<referenceTime
            <<" expected="<<expectedContact<<" actual="<<actual.contact<<" valid="<<actual.valid<<" gap="<<actual.gap
            <<" iterations="<<actual.iterations<<" status="<<actual.status<<'\n';}
    }
};
// Independent scalar convex minimization and bisection for rigidly translating
// capsules. This never uses a FP32 separating plane or conservative advancement.
std::pair<bool,double> translatingReference(NumiStaticVec3 a0,NumiStaticVec3 b0,NumiStaticVec3 a1,NumiStaticVec3 b1,float radius) {
    auto a=wide(a0),b=wide(b0),da=wide(a1)-a,db=wide(b1)-b;auto box=wide(numiStaticBench());
    auto gap=[&](double t){return sceneReference(a+da*t,b+db*t,radius,box).gap;};
    double lo=0,hi=1;
    for(unsigned i=0;i<100;++i){double l=(2*lo+hi)/3,h=(lo+2*hi)/3;if(gap(l)<gap(h))hi=h;else lo=l;}
    double middle=(lo+hi)*.5;
    if(gap(middle)>1e-12&&gap(1)>1e-12)return {false,1};
    if(gap(0)<=0)return {true,0};
    lo=0;hi=gap(middle)<=0?middle:1;
    for(unsigned i=0;i<80;++i){double m=(lo+hi)*.5;if(gap(m)<=0)hi=m;else lo=m;}
    return {true,(lo+hi)*.5};
}
// Independent temporal interval oracle for endpoints with different motions.
// At each time the geometry reference remains physical-edge/face/LP based.
// Subdivision uses FP64 endpoint Lipschitz envelopes and an explicit terminal
// uncertainty; it does not import the FP32 sweep or its supporting-plane bound.
struct TimeOracle {bool contact,known;double low,high;};
TimeOracle deformingReference(D3 a0,D3 b0,D3 a1,D3 b1,double radius,double tolerance,unsigned& nodes) {
    auto da=a1-a0,db=b1-b0;double speed=std::max(length(da),length(db));auto box=wide(numiStaticBench());
    auto gap=[&](double t){return sceneReference(a0+da*t,b0+db*t,radius,box).gap+tolerance;};
    auto interval=[&](auto&& self,double lo,double hi,double left,double right,unsigned depth)->TimeOracle {
        if(++nodes>50000)return {false,false,lo,hi};
        if(left<=0)return {true,true,lo,lo};
        double bound=.5*(left+right-speed*(hi-lo));
        if(bound>1e-12)return {false,true,lo,hi};
        if(depth==30)return {right<=0,right<=0,lo,hi};
        double middle=(lo+hi)*.5,center=gap(middle);
        auto first=self(self,lo,middle,left,center,depth+1);
        if(first.contact)return first;
        auto second=self(self,middle,hi,center,right,depth+1);
        if(second.contact)return second;
        if(!first.known||!second.known)return {false,false,lo,hi};
        return {false,true,lo,hi};
    };
    return interval(interval,0,1,gap(0),gap(1),0);
}
}

int main() {
    try {
        Test test;auto box=numiStaticBench();
        for(float radius:{.0001f,.004f,.04f,.07f}) {
            for(unsigned axis=0;axis<3;++axis)for(int sign:{-1,1}) {
                NumiStaticVec3 a{0,0,-.04f},b=a;
                unsigned span=(axis+1)%3;
                float boundary=numiStaticComponent(sign>0?box.maximum:box.minimum,axis)+sign*(radius+.003f);
                if(axis==0)a.x=b.x=boundary;if(axis==1)a.y=b.y=boundary;if(axis==2)a.z=b.z=boundary;
                if(span==0){a.x=-.4f;b.x=.4f;}if(span==1){a.y=-.3f;b.y=.3f;}if(span==2){a.z=-.07f;b.z=-.01f;}
                test.sample(a,b,radius);test.sample(b,a,radius);
            }
            // Both endpoint spheres are clear, while the yarn interior crosses
            // a face, a corner-near chord, and the deepest interior envelope.
            for(auto endpoints:std::array<std::array<NumiStaticVec3,2>,5>{{
                {{{-1,0,-.04f},{1,0,-.04f}}},
                {{{-.9f,.49f,-.04f},{.9f,.49f,-.04f}}},
                {{{.78f,.47f,.002f},{.72f,.53f,.002f}}},
                {{{.76f,.48f,-.01f},{.73f,.52f,-.01f}}},
                {{{-.8f,-.6f,-.12f},{.8f,.6f,.04f}}}
            }}) {
                test.sample(endpoints[0],endpoints[1],radius);++test.adversarial;
                auto endpointGap=std::min(numiStaticSceneSample(endpoints[0],radius).gap,numiStaticSceneSample(endpoints[1],radius).gap);
                auto whole=numiStaticSegmentSceneSample(endpoints[0],endpoints[1],radius);
                if(endpointGap>0&&whole.gap<0)++test.negativeControls;
            }
            // Degenerate axial segments and floor witness weights are explicit.
            test.sample({1,0,-.75f+radius},{1,0,-.75f+radius},radius);
            test.sample({1,0,-.6f},{2,0,-.8f},radius);
            for(float offset:{-1e-7f,0.f,1e-7f,1e-5f}) {
                test.sample({.75f+radius+offset,-.6f,.01f},{.75f+radius+offset,.6f,-.09f},radius);
                test.sample({-.8f,.5f+radius+offset,-.04f},{.8f,.5f+radius+offset,-.04f},radius);
            }
            // Whole yarn hits the bench although both endpoint point casts miss.
            NumiStaticVec3 a0{-1,0,.2f},b0{1,0,.2f},a1{-1,0,-.2f},b1{1,0,-.2f};
            double time=(double(a0.z)-radius)/(double(a0.z)-a1.z);
            test.sweep(a0,b0,a1,b1,radius,time,true,true,true);
            test.require(!numiStaticSceneCast(a0,a1,radius).contact&&!numiStaticSceneCast(b0,b1,radius).contact,
                         "endpoint-only sweep negative control unexpectedly sees the interior impact");
            ++test.negativeControls;
            // Resting face motion must not produce a time-zero locking contact.
            NumiStaticVec3 a{-.3f,0,radius},b{.3f,0,radius};
            auto tangent=numiStaticSegmentSceneCast(a,b,{-0.2f,.2f,radius},{.4f,.2f,radius},radius);
            test.require(tangent.valid&&!tangent.contact,"tangent top support was locked");++test.tangentDepartures;
            auto outward=numiStaticSegmentSceneCast(a,b,{-0.2f,.2f,radius+.1f},{.4f,.2f,radius+.1f},radius);
            test.require(outward.valid&&!outward.contact,"outward top support was locked");++test.tangentDepartures;
            auto floorTangent=numiStaticSegmentSceneCast({1,0,-.75f+radius},{1.2f,0,-.75f+radius},
                {1.1f,.2f,-.75f+radius},{1.3f,.2f,-.75f+radius},radius);
            test.require(floorTangent.valid&&!floorTangent.contact,"tangent floor support was locked");++test.tangentDepartures;
        }
        std::mt19937 generator(730491);std::uniform_real_distribution<float> xy(-1.3f,1.3f),z(-.9f,.6f),r(.0001f,.08f),motion(-.8f,.8f);
        for(unsigned i=0;i<12000;++i) {
            NumiStaticVec3 a{xy(generator),xy(generator),z(generator)},b{xy(generator),xy(generator),z(generator)};
            test.sample(a,b,r(generator));
        }
        // Independent translations include both narrow edge/corner contacts and
        // misses of the expanded-AABB corner false-positive region.
        for(unsigned i=0;i<1500;++i) {
            float radius=r(generator);NumiStaticVec3 a0{xy(generator),xy(generator),z(generator)},b0{xy(generator),xy(generator),z(generator)};
            if(sceneReference(wide(a0),wide(b0),radius,wide(box)).gap<.0001)continue;
            NumiStaticVec3 delta{motion(generator),motion(generator),motion(generator)};
            auto a1=numiStaticAdd(a0,delta),b1=numiStaticAdd(b0,delta);
            auto ref=translatingReference(a0,b0,a1,b1,radius);
            test.sweep(a0,b0,a1,b1,radius,ref.second,ref.first,false,false);
        }
        for(unsigned i=0;i<1000;++i) {
            float radius=r(generator);
            NumiStaticVec3 a0{xy(generator),xy(generator),z(generator)},b0{xy(generator),xy(generator),z(generator)},
                a1{xy(generator),xy(generator),z(generator)},b1{xy(generator),xy(generator),z(generator)};
            if(sceneReference(wide(a0),wide(b0),radius,wide(box)).gap<.0001)continue;
            ++test.deformingSweeps;unsigned nodes=0;
            auto reference=deformingReference(wide(a0),wide(b0),wide(a1),wide(b1),radius,1e-7,nodes);
            auto actual=numiStaticSegmentSceneCast(a0,b0,a1,b1,radius);
            test.deformingOracleUnresolved+=!reference.known;test.deformingSweepUnresolved+=!actual.valid;
            bool missed=actual.valid&&!actual.contact&&reference.known&&reference.contact;
            if(actual.valid&&actual.contact) {
                double independentGap=sceneReference(wide(a0)+(wide(a1)-wide(a0))*actual.time,
                    wide(b0)+(wide(b1)-wide(b0))*actual.time,radius,wide(box)).gap;
                if(independentGap< -1e-6)missed=true;
            }
            if(missed){++test.deformingMisses;std::cerr<<"deforming_sweep_miss="<<i<<" status="<<actual.status
                <<" contact="<<actual.contact<<" valid="<<actual.valid<<" reference_time="<<reference.low<<'\n';}
        }
        // Varying endpoint motion can change the interior closest parameter.
        // A descending V is an analytic top-face interior hit at u=1/2.
        for(float radius:{.004f,.04f}) {
            NumiStaticVec3 a0{-.9f,0,.2f},b0{.9f,0,.2f},a1{-.9f,0,-.2f},b1{.9f,0,.1f};
            // First contact over x in [-.75,.75] is at the left face boundary.
            double u=(double(box.minimum.x)-a0.x)/(double(b0.x)-a0.x);
            double dz=(double(a1.z)-a0.z)*(1-u)+(double(b1.z)-b0.z)*u;
            double nominal=(double(a0.z)-radius)/(-dz);
            auto actual=numiStaticSegmentSceneCast(a0,b0,a1,b1,radius);
            test.require(actual.valid&&actual.contact&&actual.time<=nominal,"deforming interior sweep missed the top/edge impact");
            auto ref=sceneReference(wide(a0)+(wide(a1)-wide(a0))*actual.time,wide(b0)+(wide(b1)-wide(b0))*actual.time,radius,wide(box));
            test.require(ref.gap>=-1e-6&&ref.gap<2e-6,"deforming sweep impact is not an independently verified contact");++test.adversarial;
        }
        // A flat top contact interval crosses both bench edges. Distance
        // ties pick the left boundary, where upward velocity must not hide
        // downward motion near the right boundary of the same supported yarn.
        for(float direction:{-1.f,1.f}) {
            NumiStaticVec3 a{-1,0,.004f},b{1,0,.004f},va{0,0,direction},vb{0,0,-direction};
            auto distance=numiStaticSegmentSceneSample(a,b,.004f);
            auto velocity=numiStaticSegmentSceneVelocitySample(a,b,va,vb,.004f,1e-7f);
            double expectedU=direction>0?.875:.125;
            test.require(velocity.valid&&velocity.feature&&velocity.gap<=1e-7f&&
                std::abs(velocity.parameter-expectedU)<2e-5,"flat bridge contact hid closing interior velocity");
            double closing=va.z+(vb.z-va.z)*velocity.parameter;
            test.require(closing<-.7499,"bridge velocity sample did not select the most closing witness");
            if(direction>0)test.require(distance.parameter<.126,"distance tie negative control changed");
            ++test.velocityQueries;++test.negativeControls;
        }
        // Analytic floor support interval ends in the middle of the segment.
        auto floorVelocity=numiStaticSegmentSceneVelocitySample({1,0,-.746f},{1.2f,0,-.6f},
            {0,0,-1},{0,0,1},.004f,1e-7f);
        test.require(floorVelocity.valid&&floorVelocity.feature==5&&floorVelocity.parameter==0,
                     "floor contact velocity endpoint was missed");++test.velocityQueries;
        // Floor boundary regression: the old world-space root rounds beyond
        // tolerance and gets discarded, hiding a closing interior witness
        // behind an outward-moving supported endpoint. Check both orientations
        // and two contact tolerances without changing any production budget.
        for(bool reverse:{false,true})for(float requestedTolerance:{1e-7f,0.f}) {
            const float radius=.004f,support=-.75f+radius;
            NumiStaticVec3 a{1,0,support},b{1.2f,0,support+.0001f};
            NumiStaticVec3 va{0,0,.01f},vb{0,0,-20};
            float tolerance=requestedTolerance>0?requestedTolerance:numiStaticTolerance(a,b);
            const auto oldBoundary=(support+tolerance-a.z)/(b.z-a.z);
            const float oldGap=numiStaticFma(b.z-a.z,oldBoundary,a.z)-support;
            test.require(oldGap>tolerance,"floor root negative control did not reproduce its rounding failure");
            ++test.floorVelocityNegativeControls;
            if(reverse){std::swap(a,b);std::swap(va,vb);}
            auto actual=numiStaticSegmentSceneVelocitySample(a,b,va,vb,radius,tolerance);
            float closing=numiStaticFma(vb.z-va.z,actual.parameter,va.z);
            test.require(actual.valid&&actual.feature==5&&actual.gap<=tolerance&&closing<-.005f,
                         "floor boundary rounding hid closing interior velocity");
            // Independently sample represented geometry with ordinary FP64
            // interpolation and the authored floor. The repair must select a
            // real nearby floor witness, not invent a distant contact.
            auto point=wide(a)+(wide(b)-wide(a))*actual.parameter;
            test.require(point.z+.75-radius<=tolerance+1e-7,
                         "floor repaired boundary is outside the independent geometry error budget");
            ++test.velocityQueries;++test.floorVelocityRegressions;
        }
        // Seeded rounded-feature queries compare with dense independent FP64
        // feature sampling. These include nonconstant normals and arbitrary
        // endpoint velocities, rather than just planar velocity gradients.
        std::uniform_real_distribution<float> small(-.006f,.006f),velocityDistribution(-5.f,5.f);
        for(unsigned i=0;i<1000;++i) {
            NumiStaticVec3 a{.75f+.004f+small(generator),.5f+.004f+small(generator),.02f+small(generator)},
                           b{.75f+.004f+small(generator),.5f+.004f+small(generator),-.1f+small(generator)};
            float radius=.007f,tolerance=2e-6f;
            NumiStaticVec3 va{velocityDistribution(generator),velocityDistribution(generator),velocityDistribution(generator)},
                           vb{velocityDistribution(generator),velocityDistribution(generator),velocityDistribution(generator)};
            auto actual=numiStaticSegmentSceneVelocitySample(a,b,va,vb,radius,tolerance);
            if(boxReference(wide(a),wide(b),radius,wide(box)).gap< -radius) {
                ++test.velocityQueries;++test.velocityRejected;
                test.require(!actual.valid,"solid-interior velocity query invented a unique support normal");
                continue;
            }
            double referenceClosing=std::numeric_limits<double>::infinity();bool supported=false;
            for(unsigned step=0;step<=2000;++step) {
                double u=double(step)/2000;auto p=wide(a)+(wide(b)-wide(a))*u;
                auto reference=sceneReference(p,p,radius,wide(box));
                if(reference.gap<=tolerance) {
                    supported=true;double closing=dot(wide(va)+(wide(vb)-wide(va))*u,reference.normal);
                    referenceClosing=std::min(referenceClosing,closing);
                }
            }
            ++test.velocityQueries;
            bool failure=!actual.valid;
            if(actual.feature) {
                auto p=wide(a)+(wide(b)-wide(a))*actual.parameter;
                auto reference=sceneReference(p,p,radius,wide(box));
                double closing=dot(wide(va)+(wide(vb)-wide(va))*actual.parameter,reference.normal);
                double deficit=supported?closing-referenceClosing:0;
                test.maxVelocitySuboptimality=std::max(test.maxVelocitySuboptimality,deficit);
                if(reference.gap>tolerance+3e-7||deficit>.004)failure=true;
            } else if(supported)failure=true;
            if(failure){++test.velocityFailures;if(test.velocityFailures<=8)std::cerr<<"velocity_failure="<<i<<" actual_feature="<<actual.feature
                <<" u="<<actual.parameter<<" valid="<<actual.valid<<" gap="<<actual.gap<<" actual_closing="<<dot(wide(va)+(wide(vb)-wide(va))*actual.parameter,wide(actual.normal))<<" reference_closing="<<referenceClosing<<'\n';}
        }
        auto noVelocitySupport=numiStaticSegmentSceneVelocitySample({0,0,.1f},{.1f,0,.1f},
            {0,0,-1},{0,0,-1},.004f,1e-7f);
        test.require(noVelocitySupport.valid&&noVelocitySupport.feature==0&&noVelocitySupport.gap>1e-7f,
                     "unsupported velocity query created contact");++test.velocityQueries;
        // Exhaustion is observable, not a successful miss. This motion reaches
        // an edge where one conservative iteration cannot establish clearance.
        auto exhausted=numiStaticSegmentSceneCast({.78f,-.2f,.08f},{.78f,.2f,.08f},
            {.72f,-.2f,-.02f},{.72f,.2f,-.02f},.004f,1e-7f,1);
        test.require(!exhausted.valid&&exhausted.status==NumiStaticSegmentIterationLimit,"iteration limit was hidden");++test.rejected;
        for(float invalidRadius:{0.f,-1.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
            auto sample=numiStaticSegmentSceneSample({0,0,1},{.1f,0,1},invalidRadius);
            auto cast=numiStaticSegmentSceneCast({0,0,1},{.1f,0,1},{0,0,.5f},{.1f,0,.5f},invalidRadius);
            test.require(!sample.valid&&!cast.valid,"invalid radius accepted");++test.rejected;
        }
        auto nan=std::numeric_limits<float>::quiet_NaN();
        test.require(!numiStaticSegmentSceneSample({nan,0,1},{0,0,1},.004f).valid,"NaN endpoint accepted");++test.rejected;
        auto initial=numiStaticSegmentSceneCast({-1,0,-.04f},{1,0,-.04f},{-1,0,.1f},{1,0,.1f},.004f);
        test.require(!initial.valid&&initial.status==NumiStaticSegmentInitialPenetration,"interior initial penetration accepted");++test.rejected;
        bool pass=!test.sampleFailures&&!test.certificateFailures&&!test.sweepFailures&&!test.velocityFailures&&!test.deformingMisses&&test.negativeControls>=8;
        std::cout<<std::setprecision(17)
            <<"backend=CPU_shared_FP32_segment_geometry native_Metal_execution=NOT_RUN full_scene=NOT_QUALIFIED\n"
            <<"samples="<<test.samples<<" sample_failures="<<test.sampleFailures<<" signed_gap_max_error_m="<<test.maxGapError
            <<" witness_max_suboptimality_m="<<test.maxWitnessError<<'\n'
            <<"clearance_certificates="<<test.certificates<<" certificate_failures="<<test.certificateFailures
            <<" maximum_lower_bound_excess_m="<<test.maxLowerBoundExcess<<'\n'
            <<"sweeps="<<test.sweeps<<" sweep_failures="<<test.sweepFailures<<" certified_clear="<<test.clear
            <<" conservative_contacts="<<test.contact<<" explicit_unresolved="<<test.unresolved
            <<" maximum_iterations="<<test.maximumIterations<<" maximum_impact_gap_m="<<test.maxImpactGap
            <<" analytic_impact_position_max_error_m="<<test.maxImpactPositionError<<'\n'
            <<"adversarial_cases="<<test.adversarial<<" endpoint_only_negative_controls="<<test.negativeControls
            <<" tangent_outward_departures="<<test.tangentDepartures<<" rejected_invalid_or_exhausted="<<test.rejected<<'\n'
            <<"velocity_queries="<<test.velocityQueries<<" velocity_failures="<<test.velocityFailures<<" solid_interior_velocity_rejected="<<test.velocityRejected<<" velocity_max_suboptimality_m_per_s="<<test.maxVelocitySuboptimality<<'\n'
            <<"floor_velocity_regressions="<<test.floorVelocityRegressions
            <<" floor_nominal_root_negative_controls="<<test.floorVelocityNegativeControls<<'\n'
            <<"deforming_sweeps="<<test.deformingSweeps<<" deforming_misses="<<test.deformingMisses
            <<" deforming_FP64_oracle_unresolved="<<test.deformingOracleUnresolved<<" deforming_FP32_sweep_unresolved="<<test.deformingSweepUnresolved<<'\n'
            <<"static_segment_host_probe_pass="<<pass<<'\n';
        return pass?0:1;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}

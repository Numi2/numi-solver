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
#include <string>
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
        ++certificates;double excess=actual.clearanceLowerBound-reference.gap;
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
            if(gap< -1e-6||gap>2e-6||actual.time>referenceTime+2e-6)failure=true;
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
struct SweepRefinementQualification {
 unsigned controls{},failures{},clear{},contact{},unresolved{},oracleUnresolved{},intervalOracleUnresolved{},analyticClosures{},falseClear{},falseContact{},acceptanceFailures{},maxIterations{},negativeControls{},tangentControls{},tinyClosingControls{},cancellationControls{},signedToleranceControls{},signedToleranceExpectedRejections{},signedToleranceNegativeControls{},certificateChecks{},certificateFailures{},witnessChecks{},witnessFailures{};
 double maxContactGap{},minContactGap{},maxBoundExcess{},maxWitnessDeficit{};

 void require(bool v,const std::string& message){if(!v){++failures;std::cerr<<"qualification_failure="<<message<<'\n';}}
 void sweep(const std::string& name,NumiStaticVec3 a0,NumiStaticVec3 b0,NumiStaticVec3 a1,NumiStaticVec3 b1,float radius,bool mustClear=false,bool mustContact=false,double analyticMinimum=std::numeric_limits<double>::quiet_NaN(),float tolerance=1e-7f){
  ++controls;auto h=numiStaticSegmentSceneCast(a0,b0,a1,b1,radius,tolerance);
  clear+=h.valid&&!h.contact;contact+=h.valid&&h.contact;unresolved+=!h.valid;maxIterations=std::max(maxIterations,h.iterations);
  unsigned nodes=0;auto ref=deformingReference(wide(a0),wide(b0),wide(a1),wide(b1),radius,tolerance,nodes);
  intervalOracleUnresolved+=!ref.known;
  if(std::isfinite(analyticMinimum)){
    if(!ref.known)++analyticClosures;ref.known=true;ref.contact=analyticMinimum<=-double(tolerance);
  }else if(!ref.known&&std::min({double(a0.z),double(b0.z),double(a1.z),double(b1.z)})>=double(radius)-double(tolerance)){
    // Independent analytic top and floor half-spaces: no point of the complete
    // bilinear hull can penetrate either collider farther than contact tolerance.
    ++analyticClosures;ref.known=true;ref.contact=false;
  }
  if(!ref.known){
    // Independent FP64 supporting feature from the physical-edge/face/LP oracle.
    // This closes exact tangent and rounded-corner miss controls for which a
    // temporal Lipschitz subdivision cannot finish at depth 30.
    auto initial=sceneReference(wide(a0),wide(b0),radius,wide(numiStaticBench()));
    auto box=wide(numiStaticBench());double floor=INFINITY;bool boxSafe=false;
    for(auto point:{wide(a0),wide(b0),wide(a1),wide(b1)})floor=std::min(floor,point.z+.75-radius);
    for(auto normal:{initial.normal,D3{1,0,0},D3{-1,0,0},D3{0,1,0},D3{0,-1,0},D3{0,0,1},D3{0,0,-1}}){
      D3 corner{normal.x>0?box.high.x:box.low.x,normal.y>0?box.high.y:box.low.y,normal.z>0?box.high.z:box.low.z};
      double projection=INFINITY;
      for(auto point:{wide(a0),wide(b0),wide(a1),wide(b1)})projection=std::min(projection,dot(point-corner,normal));
      boxSafe=boxSafe||projection/length(normal)-radius>=-double(tolerance)+1e-12;
    }
    if(boxSafe&&floor>=-double(tolerance)+1e-12){++analyticClosures;ref.known=true;ref.contact=false;}
  }
  oracleUnresolved+=!ref.known;falseClear+=h.valid&&!h.contact&&ref.known&&ref.contact;
  double gap=0;
  if(h.valid&&h.contact){auto a=wide(a0)+(wide(a1)-wide(a0))*h.time,b=wide(b0)+(wide(b1)-wide(b0))*h.time;
   gap=sceneReference(a,b,radius,wide(numiStaticBench())).gap;
   maxContactGap=std::max(maxContactGap,gap);minContactGap=std::min(minContactGap,gap);
   acceptanceFailures+=gap>2e-6||gap< -2e-6;
   falseContact+=gap>2e-6;
  }
  require(!mustClear||(h.valid&&!h.contact),name+" failed required clear");
  require(!mustContact||(h.valid&&h.contact),name+" failed required contact");
  require(h.iterations<=128,name+" exceeded cap");

 }
};

SweepRefinementQualification sweepRefinementControls(){
 SweepRefinementQualification q;
 for(float r:{.0001f,.001f,.004f,.04f,.08f})for(float direction:{-1.f,1.f}){
  // Whole interior face contact, with clear endpoint point trajectories.
  NumiStaticVec3 a0{direction*-1,0,.2f},b0{direction*1,0,.2f},a1{direction*-1,0,-.2f},b1{direction*1,0,-.2f};
  q.sweep("interior-face",a0,b0,a1,b1,r,false,true);
  q.require(!numiStaticSceneCast(a0,a1,r).contact&&!numiStaticSceneCast(b0,b1,r).contact,"endpoint-only negative control");++q.negativeControls;
  q.sweep("top-supported-tangent",{-.3f,0,r},{.3f,0,r},{-.2f,direction*.2f,r},{.4f,direction*.2f,r},r,true);++q.tangentControls;
  q.sweep("top-supported-outward",{-.3f,0,r},{.3f,0,r},{-.2f,direction*.2f,r+.1f},{.4f,direction*.2f,r+.1f},r,true);++q.tangentControls;
  float floor=-.75f+r;
  q.sweep("floor-supported-tangent",{1,0,floor},{1.2f,0,floor},{1.1f,direction*.2f,floor},{1.3f,direction*.2f,floor},r,true);++q.tangentControls;
  q.sweep("floor-supported-outward",{1,0,floor},{1.2f,0,floor},{1.1f,direction*.2f,floor+.1f},{1.3f,direction*.2f,floor+.1f},r,true);++q.tangentControls;
  q.sweep("floor-endpoint-cross",{1,0,-.5f},{1.2f,0,-.6f},{1.1f,0,-.8f},{1.3f,0,-.7f},r,false,true);
  // Explicit time boundaries: very early events, near t=1 impact, and exactly
  // terminal support. A globally certified hull within contact tolerance may
  // end on support without claiming an inward crossing or a time-zero lock.
  for(float gap:{2e-7f,5e-6f})for(float displacement:{.1f,1000.f}){
   float start=r+gap,end=start-displacement;
   q.sweep("near-time-zero-impact",{direction*-.3f,0,start},{direction*.3f,0,start},
       {direction*-.3f,0,end},{direction*.3f,0,end},r,false,true);
  }
  q.sweep("near-terminal-impact",{-.3f,0,r+.1f},{.3f,0,r+.1f},
      {-.3f,direction*.2f,r-1e-6f},{.3f,direction*.2f,r-1e-6f},r,false,true);
  q.sweep("terminal-supported-boundary",{-.3f,0,r+.1f},{.3f,0,r+.1f},
      {-.3f,direction*.2f,r},{.3f,direction*.2f,r},r,true);
  // Both directions of a rounded vertical edge and rounded horizontal edge.
  // The exact represented geometry, not a nominal unrepresented sqrt(2), owns
  // touch/miss classification. +/-4um controls bound tiny grazing ambiguity.
  for(float offset:{-4e-6f,-2e-7f,0.f,2e-7f,4e-6f}){
   float d=r/std::sqrt(2.f)+offset;
   NumiStaticVec3 c{.75f+d,.5f+d,-.04f},t{direction*.15f,-direction*.15f,0};
   auto start=numiStaticSub(c,t),end=numiStaticAdd(c,t);
   auto cornerMinimum=[&](NumiStaticVec3 first,NumiStaticVec3 last,bool top){
    D3 p=wide(first),v=wide(last)-p,corner{.75,top?0.:.5,0};
    if(top){p.y=0;v.y=0;}else {p.z=0;v.z=0;}
    auto o=p-corner;double time=std::clamp(-dot(o,v)/dot(v,v),0.,1.);
    return length(o+v*time)-r;
   };
   double minimum=cornerMinimum(start,end,false);
   q.sweep("vertical-edge-graze",{start.x,start.y,-.07f},{start.x,start.y,-.01f},{end.x,end.y,-.07f},{end.x,end.y,-.01f},r,offset>3e-6f,offset< -3e-6f,minimum);++q.cancellationControls;
   c={.75f+d,0,d};t={direction*.15f,0,-direction*.15f};start=numiStaticSub(c,t);end=numiStaticAdd(c,t);
   minimum=cornerMinimum(start,end,true);
   q.sweep("top-edge-graze",{start.x,-.4f,start.z},{start.x,.4f,start.z},{end.x,-.4f,end.z},{end.x,.4f,end.z},r,offset>3e-6f,offset< -3e-6f,minimum);++q.cancellationControls;
  }
  // FP32 top-boundary sweeps where projected closing is tiny next to tangential
  // motion. A small normal closing speed must neither exhaust a distant clear
  // chord nor turn the two physical translation directions into different rules.
  for(float closing:{0.f,1e-8f,1e-7f,2e-6f}){
   float height=r+3e-6f;
   q.sweep("tiny-plane-closing",{-.3f,-.1f,height},{.3f,-.1f,height},
       {-.3f,direction*.4f,height-closing},{.3f,direction*.4f,height-closing},r,true);++q.tinyClosingControls;
  }
  // Expanded finite AABB reports a hit in this rounded-corner miss region.
  float cornerX=.75f+.8f*r,cornerY=.5f+.8f*r;
  q.require(cornerX<.75f+r&&cornerY<.5f+r,"expanded-AABB corner negative control");
  q.sweep("expanded-box-false-positive",{cornerX,cornerY,direction*.1f},{cornerX,cornerY,direction*.1f},
      {cornerX,cornerY,-direction*.1f},{cornerX,cornerY,-direction*.1f},r,true);++q.negativeControls;
  q.sweep("deforming-descending-bridge",{-1,0,.2f},{1,0,.2f},{-1,0,-.2f},{1,0,.1f},r,false,true);
  q.sweep("deforming-endpoint-exchange",{-1,-.7f,.3f},{1,.7f,.3f},{1,-.7f,-.1f},{-1,.7f,.2f},r);
  q.sweep("deforming-clear-outward",{-.3f,-.2f,r+4e-6f},{.3f,.2f,r+4e-6f},
       {-.6f,.3f,r+.2f},{.6f,-.3f,r+.1f},r,true);
 }
 // A zero/nonfinite normal is no plane. For a nonzero subnormal normal the
 // directed lower norm may be zero; negative projection must fail explicitly.
 for(auto normal:{NumiStaticVec3{0,0,0},NumiStaticVec3{std::numeric_limits<float>::quiet_NaN(),0,0},
                  NumiStaticVec3{std::numeric_limits<float>::infinity(),0,0}})
  q.require(!std::isfinite(numiStaticSegmentPlaneClearanceBound(0,normal,.004f)),"zero/nonfinite support plane admitted");
 q.require(!std::isfinite(numiStaticSegmentPlaneClearanceBound(-1e-20f,{1e-38f,0,0},.004f)),"zero lower norm negative division admitted");
 // Radius smaller than tolerance and large custom tolerances must use signed
 // support planes. The old unsigned projection clamp returned -radius and
 // certified these 4cm interior crossings clear. Reverse endpoint ordering too.
 for(float direction:{-1.f,1.f}){
  const double unspecified=std::numeric_limits<double>::quiet_NaN();
  auto control=[&](const char* name,float radius,float tolerance,float initial,float final,bool clear,bool contact,bool reject){
   NumiStaticVec3 a0{direction*-.1f,0,initial},b0{direction*.1f,0,initial},a1{direction*-.1f,0,final},b1{direction*.1f,0,final};
   q.sweep(name,a0,b0,a1,b1,radius,clear,contact,unspecified,tolerance);++q.signedToleranceControls;
   auto h=numiStaticSegmentSceneCast(a0,b0,a1,b1,radius,tolerance);
   if(reject){++q.signedToleranceExpectedRejections;q.require(!h.valid,"signed tolerance case fabricated accepted contact/clear");}
   if(final<-.03f&&initial>=-.01f){
    q.require(-double(radius)>=-double(tolerance),"old unsigned support admission negative control");++q.signedToleranceNegativeControls;
   }
  };
  control("sub-tolerance-radius-cross",5e-8f,1e-7f,.01f,-.04f,false,true,false);
  control("sub-tolerance-radius-initial-penetration",5e-8f,1e-7f,-.04f,.01f,false,false,true);
  control("custom-tolerance-inward-overlap",.004f,.015f,-.01f,-.04f,false,false,true);
  control("custom-tolerance-outward-overlap",.004f,.015f,-.01f,-.009f,true,false,false);
  control("custom-tolerance-through-bench",.004f,.015f,.1f,-.12f,false,true,false);
  // Box is globally clear at caller tolerance but has a 24mm solid overlap.
  // The floor impact must be rejected by the complete scene lower gate.
  NumiStaticVec3 a0{.73f,0,-.01f},b0{.73f,0,-.65f},a1=a0,b1{.73f,0,-.8f};
  if(direction<0){std::swap(a0,b0);std::swap(a1,b1);}
  q.sweep("ignored-box-overlap-at-floor-impact",a0,b0,a1,b1,.004f,false,false,unspecified,.05f);
  auto h=numiStaticSegmentSceneCast(a0,b0,a1,b1,.004f,.05f);
  q.require(!h.valid,"ignored globally tolerated box overlap escaped lower contact gate");++q.signedToleranceControls;++q.signedToleranceExpectedRejections;
 }
 // Hard cancellation: long thin yarns compared with the same fixed 2um witness
 // allowance. Failure is permitted and observable; an inaccurate clear is not.
 for(float length:{1.f,8.f,32.f,1024.f})for(float direction:{-1.f,1.f})for(float r:{.0001f,.004f}){
  q.sweep("long-interior-cross",{direction*-length,0,.1f},{direction*length,0,.1f},
       {direction*-length,0,-.1f},{direction*length,0,-.1f},r,false,length<=32);++q.cancellationControls;
 }
 // Independent exact FP64 half-space certificates. Any normal is legal: local
 // feature selection need not be accurate for these conservative bounds to hold.
 std::mt19937 rng(190629);std::uniform_real_distribution<float> coord(-3,3),unit(0,1),nv(-1,1);
 auto box=numiStaticBench();DBox db=wide(box);
 for(unsigned i=0;i<20000;++i){
  NumiStaticVec3 a0{coord(rng),coord(rng),coord(rng)},b0{coord(rng),coord(rng),coord(rng)},a1{coord(rng),coord(rng),coord(rng)},b1{coord(rng),coord(rng),coord(rng)},n{nv(rng),nv(rng),nv(rng)};
  float radius=.0001f+unit(rng)*.08f,t0=unit(rng),t1=unit(rng);if(t0>t1)std::swap(t0,t1);
  if(i%3==0){auto s=numiStaticSegmentSweepSample(a0,b0,a1,b1,radius,t0);n=numiStaticSegmentSweepFeaturePlane(s);}
  if(i%17==0)n={1,1e-12f,-1e-12f};
  auto nd=wide(n);double norm=length(nd),minimum=INFINITY;
  D3 corner{n.x>0?db.high.x:db.low.x,n.y>0?db.high.y:db.low.y,n.z>0?db.high.z:db.low.z};
  for(double t:{double(t0),double(t1)})for(auto endpoints:{std::pair{wide(a0),wide(a1)},std::pair{wide(b0),wide(b1)}}){
   auto point=endpoints.first+(endpoints.second-endpoints.first)*t;
   minimum=std::min(minimum,dot(point-corner,nd));
  }
  double reference=minimum/norm-radius;
  float bound=numiStaticSegmentSweepHullBound(a0,b0,a1,b1,n,radius,t0,t1);
  ++q.certificateChecks;double excess=double(bound)-reference;q.maxBoundExcess=std::max(q.maxBoundExcess,excess);q.certificateFailures+=excess>1e-13||!std::isfinite(bound);
  double floorReference=INFINITY;
  for(double t:{double(t0),double(t1)})for(auto endpoints:{std::pair{wide(a0),wide(a1)},std::pair{wide(b0),wide(b1)}})
   floorReference=std::min(floorReference,(endpoints.first+(endpoints.second-endpoints.first)*t).z+.75-radius);
  float floorBound=numiStaticSegmentSweepFloorHull(a0,b0,a1,b1,radius,t0,t1);
  ++q.certificateChecks;excess=double(floorBound)-floorReference;q.maxBoundExcess=std::max(q.maxBoundExcess,excess);q.certificateFailures+=excess>1e-13||!std::isfinite(floorBound);
  float t=unit(rng),u=unit(rng);auto a=wide(a0)+(wide(a1)-wide(a0))*t,b=wide(b0)+(wide(b1)-wide(b0))*t,p=a+(b-a)*u;
  double gap=length(p-clampBox(p,db))-radius;
  float upper=numiStaticSegmentSweepWitnessGapUpper(a0,b0,a1,b1,radius,t,u,false);
  ++q.witnessChecks;double deficit=gap-double(upper);q.maxWitnessDeficit=std::max(q.maxWitnessDeficit,deficit);q.witnessFailures+=deficit>1e-13||!std::isfinite(upper);
  upper=numiStaticSegmentSweepWitnessGapUpper(a0,b0,a1,b1,radius,t,u,true);gap=p.z+.75-radius;
  ++q.witnessChecks;deficit=gap-double(upper);q.maxWitnessDeficit=std::max(q.maxWitnessDeficit,deficit);q.witnessFailures+=deficit>1e-13||!std::isfinite(upper);
 }
 q.require(!q.oracleUnresolved&&!q.falseClear&&!q.falseContact&&!q.acceptanceFailures,"independent oracle/contact acceptance");
 q.require(q.unresolved==4+q.signedToleranceExpectedRejections,"extreme-length failure count changed");
 q.require(!q.certificateFailures&&!q.witnessFailures,"outward arithmetic certificate/witness"); std::cout<<std::setprecision(17)<<"sweep_refinement_controls="<<q.controls<<" control_failures="<<q.failures<<" clear="<<q.clear<<" contact="<<q.contact<<" explicit_unresolved="<<q.unresolved<<" oracle_unresolved="<<q.oracleUnresolved<<" interval_oracle_unresolved="<<q.intervalOracleUnresolved<<" analytic_oracle_closures="<<q.analyticClosures<<" false_clear="<<q.falseClear<<" false_contact="<<q.falseContact<<" contact_acceptance_failures="<<q.acceptanceFailures<<" maximum_contact_gap_m="<<q.maxContactGap<<" minimum_contact_gap_m="<<q.minContactGap<<" maximum_geometry_queries="<<q.maxIterations<<" endpoint_negative_controls="<<q.negativeControls<<" tangent_outward_controls="<<q.tangentControls<<" near_zero_closing_controls="<<q.tinyClosingControls<<" cancellation_grazing_controls="<<q.cancellationControls<<" signed_tolerance_controls="<<q.signedToleranceControls<<" signed_tolerance_expected_rejections="<<q.signedToleranceExpectedRejections<<" signed_tolerance_negative_controls="<<q.signedToleranceNegativeControls<<'\n'
 <<"sweep_refinement_outward_certificate_checks="<<q.certificateChecks<<" certificate_failures="<<q.certificateFailures<<" maximum_bound_excess_m="<<q.maxBoundExcess<<" witness_upper_bound_checks="<<q.witnessChecks<<" witness_upper_bound_failures="<<q.witnessFailures<<" maximum_witness_deficit_m="<<q.maxWitnessDeficit<<'\n';
 return q;
}

}

int main() {
    try {
        Test test;auto refinement=sweepRefinementControls();auto box=numiStaticBench();
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
        bool pass=!refinement.failures&&!test.unresolved&&!test.deformingSweepUnresolved&&!test.sampleFailures&&!test.certificateFailures&&!test.sweepFailures&&!test.velocityFailures&&!test.deformingMisses&&test.negativeControls>=8;
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

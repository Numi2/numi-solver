#include "numi/elastic_yarn_contact.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <sstream>
#include <vector>
using V=std::array<double,3>;
V wide(NumiStaticVec3 p){return {p.x,p.y,p.z};}
V add(V a,V b){for(int i=0;i<3;++i)a[i]+=b[i];return a;}
V sub(V a,V b){for(int i=0;i<3;++i)a[i]-=b[i];return a;}
V mul(V a,double x){for(auto& v:a)v*=x;return a;}
double dot(V a,V b){double s=0;for(int i=0;i<3;++i)s+=a[i]*b[i];return s;}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double norm(V a){return std::sqrt(dot(a,a));}
NumiStaticVec3 narrow(V p){return {float(p[0]),float(p[1]),float(p[2])};}
struct Ref {double distance=INFINITY,u=0,v=0,w=0;bool valid=false;};
// Independent constrained least squares in (segment u, triangle v,w).
// Enumerate all active subsets of the five domain inequalities and solve KKT;
// this does not call/duplicate the production closest-feature algorithms.
Ref oracle(NumiStaticVec3 aa,NumiStaticVec3 bb,const NumiStaticVec3 tt[3]){
    V a=wide(aa),b=wide(bb),t0=wide(tt[0]),t1=wide(tt[1]),t2=wide(tt[2]);
    std::array<V,3> A{sub(b,a),mul(sub(t1,t0),-1),mul(sub(t2,t0),-1)};V r=sub(a,t0);
    std::array<V,5> C{{{-1,0,0},{1,0,0},{0,-1,0},{0,0,-1},{0,1,1}}};
    std::array<double,5> bound{0,1,0,0,1};Ref best;
    for(unsigned mask=0;mask<32;++mask){int active=std::popcount(mask);if(active>3)continue;int n=3+active;double m[6][7]{};
        for(int i=0;i<3;++i){for(int j=0;j<3;++j)m[i][j]=dot(A[i],A[j]);m[i][n]=-dot(A[i],r);}
        int row=3;for(int c=0;c<5;++c)if(mask&(1u<<c)){for(int k=0;k<3;++k)m[row][k]=m[k][row]=C[c][k];m[row][n]=bound[c];++row;}
        bool good=true;
        for(int k=0;k<n;++k){int pivot=k;for(int row=k+1;row<n;++row)if(std::abs(m[row][k])>std::abs(m[pivot][k]))pivot=row;
            if(std::abs(m[pivot][k])<1e-17){good=false;break;}for(int j=k;j<=n;++j)std::swap(m[k][j],m[pivot][j]);
            double divisor=m[k][k];for(int j=k;j<=n;++j)m[k][j]/=divisor;
            for(int row=0;row<n;++row)if(row!=k){double x=m[row][k];for(int j=k;j<=n;++j)m[row][j]-=x*m[k][j];}}
        if(!good)continue;V z{m[0][n],m[1][n],m[2][n]};for(int c=0;c<5;++c)if(dot(C[c],z)>bound[c]+2e-9)good=false;
        if(!good)continue;V delta=r;for(int k=0;k<3;++k)delta=add(delta,mul(A[k],z[k]));double distance=norm(delta);
        if(!best.valid||distance<best.distance-1e-12||(std::abs(distance-best.distance)<=1e-12&&z[0]<best.u))best={distance,z[0],z[1],z[2],true};
    }
    return best;
}
long double energyDelta(const NumiElasticYarnBody b[5],const NumiStaticVec3 v[5]){long double result=0;
    for(int i=0;i<5;++i)for(int k=0;k<3;++k){long double x=numiStaticComponent(b[i].velocity,k),y=numiStaticComponent(v[i],k);result+=.5L*b[i].mass*(y-x)*(y+x);}return result;}
struct Test {int geometries=0,geometryFailures=0,geometryAmbiguous=0,impulses=0,impulseFailures=0,energyRejected=0,negativeControls=0,invalidControls=0,vertexMotions=0,allFiveMotions=0,explicitRows=0,statusControls=0;int statusCounts[6]{};
    double maxDistanceError=0,maxWitnessError=0,maxVelocityError=0,maxImpulseError=0,maxMomentumError=0,maxAngularError=0,maxClosing=0,maxEnergyGain=0,maxEnergyBoundExcess=0;
    void geometry(NumiStaticVec3 a,NumiStaticVec3 b,const NumiStaticVec3 t[3],float radius){++geometries;auto actual=numiElasticYarnClosestSegmentTriangle(a,b,t,radius);auto ref=oracle(a,b,t);
        if(!actual.valid||!ref.valid){++geometryFailures;return;}double error=std::abs(actual.distance-ref.distance);maxDistanceError=std::max(maxDistanceError,error);
        double distance=norm(sub(wide(actual.yarnPoint),wide(actual.trianglePoint)));maxWitnessError=std::max(maxWitnessError,std::abs(distance-actual.distance));
        geometryAmbiguous+=!actual.normalDefined;bool fail=error>2e-6||std::abs(distance-actual.distance)>2e-6||actual.yarnParameter<0||actual.yarnParameter>1;
        double sum=0;for(float w:actual.triangleWeights){sum+=w;fail|=w<0||w>1;}fail|=std::abs(sum-1)>2e-6;
        if(fail){++geometryFailures;if(geometryFailures<9)std::cerr<<"geometry_failure="<<geometries<<" error="<<error<<" actual="<<actual.distance<<" reference="<<ref.distance<<'\n';}}
    void impulse(NumiElasticYarnBody b[5],float radius,bool expectContact=true,const NumiElasticYarnWitness* given=nullptr){++impulses;auto solve=[&](){return given?numiElasticYarnResolveCallerValidatedNormal(b,radius,given->yarnParameter,given->triangleWeights[1],given->triangleWeights[2]):numiElasticYarnResolveNormal(b,radius);};auto actual=solve();auto replay=solve();bool exact=true;if(actual.status<6)++statusCounts[actual.status];
        exact&=actual.impulse==replay.impulse&&actual.effectiveInverseMass==replay.effectiveInverseMass&&actual.closingBefore==replay.closingBefore&&actual.closingAfter==replay.closingAfter&&actual.energyUpper==replay.energyUpper&&actual.status==replay.status&&actual.valid==replay.valid;
        for(int i=0;i<5;++i)exact&=std::memcmp(&actual.velocities[i],&replay.velocities[i],sizeof(NumiStaticVec3))==0;
        if(!actual.valid){energyRejected+=actual.status==NumiElasticYarnEnergyRejected;if(actual.status!=NumiElasticYarnEnergyRejected){++impulseFailures;std::cerr<<"impulse_invalid status="<<actual.status<<'\n';}return;}
        NumiStaticVec3 t[3]={b[2].position,b[3].position,b[4].position};auto witness=given?*given:numiElasticYarnClosestSegmentTriangle(b[0].position,b[1].position,t,radius);
        double weights[5]={1-double(witness.yarnParameter),double(witness.yarnParameter),-double(witness.triangleWeights[0]),-double(witness.triangleWeights[1]),-double(witness.triangleWeights[2])};
        V n=wide(witness.normal);double closing=0,denominator=0;for(int i=0;i<5;++i){closing+=weights[i]*dot(n,wide(b[i].velocity));denominator+=weights[i]*weights[i]*dot(n,n)/b[i].mass;}
        double J=std::max(0.,-closing/denominator);if(witness.gap>1e-7)J=0;
        maxImpulseError=std::max(maxImpulseError,std::abs(J-actual.impulse));V momentum{},angular{};double postClosing=0,velocityError=0;int motions=0;
        for(int i=0;i<5;++i){V expected=add(wide(b[i].velocity),mul(n,J*weights[i]/b[i].mass));velocityError=std::max(velocityError,norm(sub(wide(actual.velocities[i]),expected)));
            auto impulse=mul(sub(wide(actual.velocities[i]),wide(b[i].velocity)),b[i].mass);momentum=add(momentum,impulse);angular=add(angular,cross(wide(b[i].position),impulse));postClosing+=weights[i]*dot(n,wide(actual.velocities[i]));
            if(norm(impulse)>1e-12){++motions;if(i>=2)++vertexMotions;}}
        allFiveMotions+=motions==5;explicitRows+=given!=nullptr;
        maxVelocityError=std::max(maxVelocityError,velocityError);maxMomentumError=std::max(maxMomentumError,norm(momentum));maxAngularError=std::max(maxAngularError,norm(angular));
        maxClosing=std::max(maxClosing,std::max(0.,-postClosing));double energy=double(energyDelta(b,actual.velocities));maxEnergyGain=std::max(maxEnergyGain,energy);maxEnergyBoundExcess=std::max(maxEnergyBoundExcess,energy-double(actual.energyUpper));
        bool fail=!exact||velocityError>1e-5||norm(momentum)>1e-7||norm(angular)>1e-8||std::abs(J-actual.impulse)>1e-7||std::max(0.,-postClosing)>1e-5||energy>0||energy>actual.energyUpper+1e-12||(expectContact&&J>0&&actual.status!=NumiElasticYarnSolved);
        if(fail){++impulseFailures;if(impulseFailures<9)std::cerr<<"impulse_failure="<<impulses<<" v_error="<<velocityError<<" momentum="<<norm(momentum)<<" angular="<<norm(angular)<<" energy="<<energy<<" bound="<<actual.energyUpper<<'\n';}
    }
};
int main(int argc,char** argv){
    std::string fixturePath,fixtureOutput,fixtureJson="null";
    for(int i=1;i<argc;++i){std::string arg=argv[i];
        if(arg=="--help"){std::cout<<"Usage: numi-solver-elastic-yarn-contact [--fixture PATH] [--fixture-output PATH]\n";return 0;}
        if((arg=="--fixture"||arg=="--fixture-output")&&i+1<argc){(arg=="--fixture"?fixturePath:fixtureOutput)=argv[++i];continue;}
        std::cerr<<"Unknown or incomplete argument: "<<arg<<'\n';return 2;
    }
    if(!fixtureOutput.empty()&&fixturePath.empty()){std::cerr<<"--fixture-output requires --fixture\n";return 2;}
    Test test;NumiStaticVec3 t[3]={{-.02f,-.02f,0},{.02f,-.02f,0},{0,.02f,0}};
    for(auto segment:std::array<std::array<NumiStaticVec3,2>,8>{{{{{0,0,-.04f},{0,0,.04f}}},{{{-.1f,0,.003f},{.1f,0,.003f}}},{{{.025f,-.025f,.003f},{.026f,-.024f,.003f}}},{{{0,0,.003f},{0,0,.003f}}},{{{-.02f,-.02f,0},{.02f,-.02f,0}}},{{{-.1f,0,0},{.1f,0,0}}},{{{.04f,0,.003f},{.03f,.01f,.003f}}},{{{-.1f,.03f,.003f},{.1f,.03f,.003f}}}}})test.geometry(segment[0],segment[1],t,.004f);
    // Interior crossing: endpoint-only tests miss a thick-yarn boundary contact.
    auto crossing=numiElasticYarnClosestSegmentTriangle({0,0,-.04f},{0,0,.04f},t,.004f);
    auto first=oracle({0,0,-.04f},{0,0,-.04f},t),second=oracle({0,0,.04f},{0,0,.04f},t);
    if(crossing.valid&&crossing.axisCrossing&&!crossing.normalDefined&&crossing.gap<0&&first.distance>.004&&second.distance>.004)++test.negativeControls;else ++test.geometryFailures;
    std::mt19937 rng(938742);std::uniform_real_distribution<float> position(-.2f,.2f),mass(.001f,.1f),velocity(-5.f,5.f),positive(.0001f,.007f);
    for(int i=0;i<12000;++i){NumiStaticVec3 triangle[3];for(auto& p:triangle)p={position(rng),position(rng),position(rng)};NumiStaticVec3 a{position(rng),position(rng),position(rng)},b{position(rng),position(rng),position(rng)};test.geometry(a,b,triangle,positive(rng));}
    // Construct genuine five-node contacts: triangle vertices respond according
    // to their own masses. Unequal velocities/masses are never replaced by a
    // fruit pose, aggregate rigid inverse mass or a prescribed triangle motion.
    for(int i=0;i<2000;++i){NumiElasticYarnBody b[5]{};b[0].position={-.005f,0,.003999f};b[1].position={.005f,0,.003999f};for(int k=0;k<3;++k)b[k+2].position=t[k];
        for(int k=0;k<5;++k){b[k].mass=mass(rng);b[k].velocity={velocity(rng),velocity(rng),velocity(rng)};}
        auto w=numiElasticYarnClosestSegmentTriangle(b[0].position,b[1].position,t,.004f);V tri{};for(int k=0;k<3;++k)tri=add(tri,mul(wide(b[k+2].velocity),w.triangleWeights[k]));
        b[0].velocity.z=b[1].velocity.z=float(tri[2]-2-positive(rng)*100);test.impulse(b,.004f);
        auto interior=numiElasticYarnWitnessAtCallerValidatedCoordinates(b[0].position,b[1].position,t,.004f,.5f,.25f,.5f);
        V faceVelocity{};for(int k=0;k<3;++k)faceVelocity=add(faceVelocity,mul(wide(b[k+2].velocity),interior.triangleWeights[k]));
        b[0].velocity.z=b[1].velocity.z=float(faceVelocity[2]-2-positive(rng)*100);test.impulse(b,.004f,true,&interior);
    }
    NumiElasticYarnBody invalid[5]{};for(int i=0;i<5;++i){invalid[i].mass=.01f;invalid[i].position=i<2?NumiStaticVec3{float(i)*.01f,0,.004f}:t[i-2];}
    for(float bad:{0.f,-1.f,INFINITY,NAN}){auto original=invalid[3].mass;invalid[3].mass=bad;if(!numiElasticYarnResolveNormal(invalid,.004f).valid)++test.invalidControls;else ++test.impulseFailures;invalid[3].mass=original;}
    for(auto coordinates:std::array<std::array<float,3>,4>{{{{-.1f,.25f,.5f}},{{.5f,-.1f,.5f}},{{.5f,.75f,.75f}},{{NAN,.25f,.5f}}}}){
        auto result=numiElasticYarnResolveCallerValidatedNormal(invalid,.004f,coordinates[0],coordinates[1],coordinates[2]);
        if(!result.valid&&result.status==NumiElasticYarnInvalid)++test.invalidControls;else ++test.impulseFailures;
    }
    auto old=invalid[0].position;invalid[0].position={0,0,-.04f};invalid[1].position={0,0,.04f};auto ambiguous=numiElasticYarnResolveNormal(invalid,.004f);if(!ambiguous.valid&&ambiguous.status==NumiElasticYarnAmbiguousNormal)++test.invalidControls;else ++test.impulseFailures;invalid[0].position=old;
    NumiStaticVec3 degenerate[3]={{0,0,0},{.01f,0,0},{.02f,0,0}};if(!numiElasticYarnClosestSegmentTriangle({0,0,.1f},{0,0,.2f},degenerate,.004f).valid)++test.invalidControls;else ++test.geometryFailures;
    // Omitting triangle inverse masses makes a reciprocal five-node impulse
    // over-large. Freezing their response instead violates momentum closure.
    NumiElasticYarnBody control[5]{};control[0]={{-.005f,0,.003999f},{0,0,-1},1};control[1]={{.005f,0,.003999f},{0,0,-1},1};for(int i=2;i<5;++i)control[i]={t[i-2],{0,0,0},.001f};
    auto witness=numiElasticYarnClosestSegmentTriangle(control[0].position,control[1].position,t,.004f);double weights[5]={1-witness.yarnParameter,witness.yarnParameter,-witness.triangleWeights[0],-witness.triangleWeights[1],-witness.triangleWeights[2]};double yarnK=weights[0]*weights[0]+weights[1]*weights[1];double wrongJ=1/yarnK;NumiStaticVec3 wrong[5];V nonreciprocalMomentum{};
    for(int i=0;i<5;++i){wrong[i]=narrow(add(wide(control[i].velocity),mul(wide(witness.normal),wrongJ*weights[i]/control[i].mass)));if(i<2)nonreciprocalMomentum=add(nonreciprocalMomentum,mul(sub(wide(wrong[i]),wide(control[i].velocity)),control[i].mass));}
    double wrongGain=double(energyDelta(control,wrong));if(wrongGain>1)++test.negativeControls;else ++test.impulseFailures;if(norm(nonreciprocalMomentum)>.1)++test.negativeControls;else ++test.impulseFailures;test.impulse(control,.004f);
    // Status controls include open motion and geometric separation. The large
    // common velocity below is a representation stress control, not a scene.
    NumiElasticYarnBody statusBodies[5];for(int i=0;i<5;++i)statusBodies[i]=control[i];
    statusBodies[0].velocity.z=statusBodies[1].velocity.z=1;
    auto inactive=numiElasticYarnResolveCallerValidatedNormal(statusBodies,.004f,.5f,.25f,.5f);
    if(inactive.valid&&inactive.status==NumiElasticYarnInactive&&inactive.impulse==0)++test.statusControls;else ++test.impulseFailures;
    statusBodies[0].position.z=statusBodies[1].position.z=.04f;
    auto separated=numiElasticYarnResolveNormal(statusBodies,.004f);
    if(separated.valid&&separated.status==NumiElasticYarnSeparated&&separated.impulse==0)++test.statusControls;else ++test.impulseFailures;
    const float rejectionMass[5]={.027200127020478249f,.014301369898021221f,.014095899648964405f,.042445488274097443f,.096785306930541992f};
    const float rejectionVelocity[5]={999999.1875f,999999.1875f,1000015.75f,999998.625f,999991.4375f};
    for(int i=0;i<5;++i){statusBodies[i]=control[i];statusBodies[i].mass=rejectionMass[i];statusBodies[i].velocity={0,0,rejectionVelocity[i]};}
    auto rejected=numiElasticYarnResolveCallerValidatedNormal(statusBodies,.004f,.5f,.25f,.5f);
    // Independent reconstruction of the would-be accepted FP32 candidate. The
    // module deliberately withholds these velocities on rejection.
    auto rejectionWitness=numiElasticYarnWitnessAtCallerValidatedCoordinates(statusBodies[0].position,statusBodies[1].position,t,.004f,.5f,.25f,.5f);
    float rejectionWeights[5]={.5f,.5f,-.25f,-.25f,-.5f};float rejectionClosing=0,rejectionK=0;
    NumiStaticVec3 rejectionGradients[5],unsafeVelocities[5];
    for(int i=0;i<5;++i){rejectionGradients[i]=numiStaticScale(rejectionWitness.normal,rejectionWeights[i]);rejectionClosing+=numiStaticDot(rejectionGradients[i],statusBodies[i].velocity);rejectionK+=numiStaticDot(rejectionGradients[i],rejectionGradients[i])/statusBodies[i].mass;}
    float unsafeImpulse=-rejectionClosing/rejectionK;
    bool rejectionStateUnchanged=rejected.impulse==0&&!numiElasticYarnCanPublish(rejected);
    for(int i=0;i<5;++i){auto g=rejectionGradients[i],v=statusBodies[i].velocity;float response=unsafeImpulse/statusBodies[i].mass;unsafeVelocities[i]={std::fma(g.x,response,v.x),std::fma(g.y,response,v.y),std::fma(g.z,response,v.z)};
        rejectionStateUnchanged&=std::memcmp(&v,&rejected.velocities[i],sizeof(v))==0;}
    double rejectedGain=double(energyDelta(statusBodies,unsafeVelocities));
    if(!rejected.valid&&rejected.status==NumiElasticYarnEnergyRejected&&rejectedGain>0&&rejected.energyUpper>=rejectedGain&&rejectionStateUnchanged){++test.statusControls;++test.negativeControls;}else ++test.impulseFailures;
    // Optional source-bound owning-node fixture exported from a real FEM state.
    if(!fixturePath.empty()){
        std::ifstream f(fixturePath);NumiElasticYarnBody b[5];float radius;f>>radius;
        for(auto& q:b)f>>q.position.x>>q.position.y>>q.position.z>>q.velocity.x>>q.velocity.y>>q.velocity.z>>q.mass;
        if(!f)++test.impulseFailures;
        else{
            NumiStaticVec3 triangle[3]={b[2].position,b[3].position,b[4].position};
            test.geometry(b[0].position,b[1].position,triangle,radius);test.impulse(b,radius);
            // Flat parallel closest set: use its interior row, preserving all
            // three actual FEM nodal states and both authored yarn endpoints.
            auto interior=numiElasticYarnWitnessAtCallerValidatedCoordinates(b[0].position,b[1].position,triangle,radius,.5f,1.f/3,1.f/3);
            auto ref=oracle(b[0].position,b[1].position,triangle);
            if(!interior.valid||!interior.normalDefined||!ref.valid||std::abs(interior.distance-ref.distance)>2e-6)++test.geometryFailures;
            test.impulse(b,radius,true,&interior);
            auto result=numiElasticYarnResolveCallerValidatedNormal(b,radius,.5f,1.f/3,1.f/3);
            int motionCount=0;for(int i=0;i<5;++i)motionCount+=norm(sub(wide(result.velocities[i]),wide(b[i].velocity)))>1e-7;
            if(motionCount!=5)++test.impulseFailures;
            std::ostringstream detail;
            detail<<std::setprecision(17)<<"{\n  \"valid\":"<<(result.valid?"true":"false")<<",\n  \"status\":"<<result.status
                <<",\n  \"explicit_parallel_interior_row\":true,\n  \"five_nodes_moving\":"<<motionCount
                <<",\n  \"yarn_parameter\":"<<interior.yarnParameter<<",\n  \"triangle_weights\":["<<interior.triangleWeights[0]<<","<<interior.triangleWeights[1]<<","<<interior.triangleWeights[2]<<"]"
                <<",\n  \"distance_m\":"<<interior.distance<<",\n  \"FP64_global_minimum_distance_m\":"<<ref.distance
                <<",\n  \"impulse_Ns\":"<<result.impulse<<",\n  \"kinetic_delta_J\":"<<double(energyDelta(b,result.velocities))
                <<",\n  \"energy_interval_upper_J\":"<<result.energyUpper<<",\n  \"five_final_velocities\":[";
            for(int i=0;i<5;++i){if(i)detail<<",";detail<<"["<<result.velocities[i].x<<","<<result.velocities[i].y<<","<<result.velocities[i].z<<"]";}detail<<"]\n}\n";fixtureJson=detail.str();
            if(!fixtureOutput.empty()){std::ofstream output(fixtureOutput);output<<fixtureJson;if(!output)++test.impulseFailures;}
        }
    }
    bool pass=test.geometryFailures==0&&test.impulseFailures==0&&test.negativeControls==4&&test.invalidControls==10&&test.statusControls==3&&test.vertexMotions>0;
    std::cout<<std::setprecision(17)<<"{\n  \"backend\":\"CPU_FP32_five_dynamic_nodes\",\n  \"result\":\""<<(pass?"PASS":"FAIL")<<"\",\n  \"geometry_cases\":"<<test.geometries<<",\n  \"geometry_failures\":"<<test.geometryFailures<<",\n  \"geometry_ambiguous_normal_cases\":"<<test.geometryAmbiguous<<",\n  \"maximum_FP64_distance_error_m\":"<<test.maxDistanceError<<",\n  \"maximum_witness_distance_error_m\":"<<test.maxWitnessError<<",\n  \"impulse_cases\":"<<test.impulses<<",\n  \"impulse_failures\":"<<test.impulseFailures<<",\n  \"normal_row_status_counts\":["<<test.statusCounts[0]<<","<<test.statusCounts[1]<<","<<test.statusCounts[2]<<","<<test.statusCounts[3]<<","<<test.statusCounts[4]<<","<<test.statusCounts[5]<<"],\n  \"energy_rejected_candidates\":"<<test.energyRejected<<",\n  \"triangle_vertex_motion_events\":"<<test.vertexMotions<<",\n  \"all_five_node_motion_events\":"<<test.allFiveMotions<<",\n  \"explicit_parallel_rows\":"<<test.explicitRows<<",\n  \"maximum_FP64_velocity_error_m_s\":"<<test.maxVelocityError<<",\n  \"maximum_FP64_impulse_error_Ns\":"<<test.maxImpulseError<<",\n  \"maximum_actual_mass_linear_momentum_error_Ns\":"<<test.maxMomentumError<<",\n  \"maximum_actual_mass_angular_momentum_error_Nms\":"<<test.maxAngularError<<",\n  \"maximum_remaining_closing_m_s\":"<<test.maxClosing<<",\n  \"maximum_actual_kinetic_gain_J\":"<<test.maxEnergyGain<<",\n  \"maximum_energy_interval_bound_excess_J\":"<<test.maxEnergyBoundExcess<<",\n  \"negative_controls\":"<<test.negativeControls<<",\n  \"invalid_controls\":"<<test.invalidControls<<",\n  \"status_controls\":"<<test.statusControls<<",\n  \"rejected_FP32_candidate_actual_kinetic_gain_J\":"<<rejectedGain<<",\n  \"rejected_FP32_candidate_energy_upper_J\":"<<rejected.energyUpper<<",\n  \"wrong_omit_triangle_inverse_mass_gain_J\":"<<wrongGain<<",\n  \"wrong_frozen_triangle_momentum_error_Ns\":"<<norm(nonreciprocalMomentum)<<",\n  \"energy_rejection_returns_original_velocities\":"<<(rejectionStateUnchanged?"true":"false")<<",\n  \"owning_node_fixture\":"<<fixtureJson<<",\n  \"native_GPU_execution\":false,\n  \"full_scene_qualified\":false\n}\n";return pass?0:1;
}

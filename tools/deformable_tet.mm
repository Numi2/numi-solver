#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/deformable_tet.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V = std::array<double, 3>;
using M = std::array<V, 3>; // Rows, independent of Metal's column storage.
V xyz(mr_float4 v) { return {v.x, v.y, v.z}; }
mr_float4 f4(V v, double w = 0) { return {float(v[0]), float(v[1]), float(v[2]), float(w)}; }
V sub(V a, V b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
double dot(V a, V b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
double norm(V a) { return std::sqrt(dot(a,a)); }
V cross(V a, V b) { return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
V mul(M a, V b) { return {dot(a[0],b),dot(a[1],b),dot(a[2],b)}; }
M product(M a, M b) {
    M c{};
    for (int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) c[i][j]+=a[i][k]*b[k][j];
    return c;
}
double determinant(M a) { return dot(a[0],cross(a[1],a[2])); }
M inverse(M a) {
    M c{{cross(a[1],a[2]),cross(a[2],a[0]),cross(a[0],a[1])}}, b{};
    double d=determinant(a);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i][j]=c[j][i]/d;
    return b;
}
M identity() { return {{{1,0,0},{0,1,0},{0,0,1}}}; }
NumiDeformableTetInput makeInput(M deformation, V shift={}, double scale=1,
                                double mu=4000, double lambda=6000) {
    std::array<V,4> rest{{{0,0,0},{.1*scale,0,0},{0,.13*scale,0},{.01*scale,.02*scale,.09*scale}}};
    M restEdges{};
    for(int j=0;j<3;++j)for(int i=0;i<3;++i)restEdges[i][j]=rest[j+1][i];
    M inv=inverse(restEdges);
    NumiDeformableTetInput input{};
    for(int n=0;n<4;++n) {
        V p=mul(deformation,rest[n]);
        for(int k=0;k<3;++k)p[k]+=shift[k];
        input.positions[n]=f4(p);
    }
    for(int i=0;i<3;++i)input.inverseRestRows[i]=f4(inv[i],i==0?determinant(restEdges)/6:0);
    input.material={float(mu),float(lambda),0,0};
    input.control.x=NUMI_DEFORMABLE_TET_ABI_VERSION;
    return input;
}
double energy(const NumiDeformableTetInput& input, std::array<V,4> positions) {
    M edges{}, rest{};
    for(int j=0;j<3;++j)for(int i=0;i<3;++i)edges[i][j]=positions[j+1][i]-positions[0][i];
    for(int i=0;i<3;++i)rest[i]=xyz(input.inverseRestRows[i]);
    M F=product(edges,rest);
    double J=determinant(F), invariant=0;
    if(!(J>0))throw std::runtime_error("oracle element inverted");
    for(auto row:F)invariant+=dot(row,row);
    double l=std::log(J), mu=input.material.x, lambda=input.material.y;
    return input.inverseRestRows[0].w*(.5*mu*(invariant-3)-mu*l+.5*lambda*l*l);
}
std::vector<NumiDeformableTetOutput> runGPU(id<MTLDevice> device,
                                          id<MTLComputePipelineState> pipeline,
                                          const std::vector<NumiDeformableTetInput>& inputs) {
    auto in=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(inputs[0]) options:MTLResourceStorageModeShared];
    auto out=[device newBufferWithLength:inputs.size()*sizeof(NumiDeformableTetOutput) options:MTLResourceStorageModeShared];
    auto queue=[device newCommandQueue];
    if(!in||!out||!queue)throw std::runtime_error("Metal allocation failed");
    auto command=[queue commandBuffer]; auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline]; [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    uint32_t count=uint32_t(inputs.size()); [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
    std::vector<NumiDeformableTetOutput> result(inputs.size());
    std::memcpy(result.data(),out.contents,result.size()*sizeof(result[0])); return result;
}
struct Frame {
    std::vector<NumiDeformableTetInput> elements;
    std::vector<mr_float4> velocityAndMass;
    std::vector<NumiDeformableTetOutput> outputs;
};
std::vector<Frame> simulate(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
                          const std::vector<NumiDeformableTetInput>& inputs, float dt) {
    std::vector<mr_float4> velocities(4*inputs.size(),{0,0,0,.05f});
    auto states=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(inputs[0]) options:MTLResourceStorageModeShared];
    auto velocity=[device newBufferWithBytes:velocities.data() length:velocities.size()*sizeof(velocities[0]) options:MTLResourceStorageModeShared];
    auto out=[device newBufferWithLength:inputs.size()*sizeof(NumiDeformableTetOutput) options:MTLResourceStorageModeShared];
    auto queue=[device newCommandQueue];
    if(!states||!velocity||!out||!queue)throw std::runtime_error("simulation allocation failed");
    std::vector<Frame> frames;
    auto capture=[&] {
        Frame frame{std::vector<NumiDeformableTetInput>(inputs.size()),velocities,
                    std::vector<NumiDeformableTetOutput>(inputs.size())};
        std::memcpy(frame.elements.data(),states.contents,inputs.size()*sizeof(inputs[0]));
        std::memcpy(frame.velocityAndMass.data(),velocity.contents,velocities.size()*sizeof(velocities[0]));
        std::memcpy(frame.outputs.data(),out.contents,frame.outputs.size()*sizeof(frame.outputs[0]));
        frames.push_back(frame);
    };
    // The first captured state is authored; its force/energy is evaluated below
    // through the independent CPU energy, not uninitialized device output.
    std::memset(out.contents,0,inputs.size()*sizeof(NumiDeformableTetOutput));capture();
    uint32_t count=uint32_t(inputs.size());mr_float4 gravity{0,0,-9.81f,dt};
    uint32_t steps=uint32_t(std::lround(.5/dt)),batch=steps/50;
    for(uint32_t completed=0;completed<steps;completed+=batch) {
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:states offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];[encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBuffer:velocity offset:0 atIndex:3];[encoder setBytes:&gravity length:sizeof(gravity) atIndex:4];
        for(uint32_t step=0;step<batch;++step) {
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        capture();
    }
    return frames;
}
V center(const Frame& frame,size_t element) {
    V result{};double total=0;
    for(int n=0;n<4;++n) {
        double mass=frame.velocityAndMass[4*element+n].w;total+=mass;
        V p=xyz(frame.elements[element].positions[n]);for(int k=0;k<3;++k)result[k]+=mass*p[k];
    }
    for(auto& x:result)x/=total;return result;
}
bool rejectedAdvancePreservesState(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
                                  NumiDeformableTetInput input,bool invalidMass) {
    std::array<mr_float4,4> velocities{{{0,0,0,.05f},{0,0,0,.05f},{0,0,0,.05f},{0,0,0,.05f}}};
    if(invalidMass)velocities[2].w=0;
    else velocities[3].z=-2000; // Would invert the element in one tick.
    auto states=[device newBufferWithBytes:&input length:sizeof(input) options:MTLResourceStorageModeShared];
    auto velocity=[device newBufferWithBytes:velocities.data() length:sizeof(velocities) options:MTLResourceStorageModeShared];
    auto output=[device newBufferWithLength:sizeof(NumiDeformableTetOutput) options:MTLResourceStorageModeShared];
    auto queue=[device newCommandQueue];
    if(!states||!velocity||!output||!queue)throw std::runtime_error("rollback allocation failed");
    std::memset(output.contents,0,sizeof(NumiDeformableTetOutput));
    auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
    uint32_t count=1;mr_float4 gravity{0,0,-9.81f,1e-4f};
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:states offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];[encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder setBuffer:velocity offset:0 atIndex:3];[encoder setBytes:&gravity length:sizeof(gravity) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
    auto result=static_cast<const NumiDeformableTetOutput*>(output.contents);
    uint32_t expected=invalidMass?NUMI_DEFORMABLE_TET_FAILURE_MATERIAL:NUMI_DEFORMABLE_TET_FAILURE_INVERSION;
    return (result->control.x&expected)!=0&&(result->control.y&expected)!=0&&std::memcmp(&input,states.contents,sizeof(input))==0&&
        std::memcmp(velocities.data(),velocity.contents,sizeof(velocities))==0;
}
double mechanicalEnergy(const Frame& frame,size_t element) {
    std::array<V,4> positions{};double result=0;
    for(int n=0;n<4;++n) {
        positions[n]=xyz(frame.elements[element].positions[n]);auto v=frame.velocityAndMass[4*element+n];
        result+=.5*v.w*dot(xyz(v),xyz(v))+v.w*9.81f*positions[n][2];
    }
    return result+energy(frame.elements[element],positions);
}
void exportFrames(const std::string& prefix,const std::vector<Frame>& frames) {
    if(prefix.empty())return;
    std::ofstream trace(prefix+".csv");if(!trace)throw std::runtime_error("cannot write element trace");
    trace<<"frame,time_s,element,node,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,mass_kg\n"<<std::setprecision(12);
    for(size_t frame=0;frame<frames.size();++frame) {
        std::ofstream obj(prefix+"-"+std::to_string(frame)+".obj");if(!obj)throw std::runtime_error("cannot write element OBJ");
        obj<<"# Native Metal elastic tetrahedra, no contact surface\n"<<std::setprecision(12);
        for(size_t element=0;element<frames[frame].elements.size();++element)for(int n=0;n<4;++n) {
            auto p=frames[frame].elements[element].positions[n];auto v=frames[frame].velocityAndMass[4*element+n];
            obj<<"v "<<p.x<<' '<<p.y<<' '<<p.z<<'\n';
            trace<<frame<<','<<frame*.01<<','<<element<<','<<n<<','<<p.x<<','<<p.y<<','<<p.z<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\n';
        }
        for(size_t element=0;element<frames[frame].elements.size();++element)for(auto face:std::array<std::array<int,3>,4>{{{1,3,2},{1,2,4},{1,4,3},{2,3,4}}})
            obj<<"f "<<4*element+face[0]<<' '<<4*element+face[1]<<' '<<4*element+face[2]<<'\n';
    }
}
int run(const std::string& trajectoryPrefix) {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device)throw std::runtime_error("Metal device unavailable");
    NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_DEFORMABLE_TET_METALLIB] error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto function=[library newFunctionWithName:@"numi_deformable_tet_evaluate"];
    auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
    double angle=1.1,c=std::cos(angle),s=std::sin(angle);
    M rotation{{{c,-s,0},{s,c,0},{0,0,1}}};
    std::vector<NumiDeformableTetInput> inputs{makeInput(identity()),makeInput(identity(),{2,-3,8}),makeInput(rotation)};
    M shear=identity();shear[0][1]=.35;
    M compression{{{.7,0,0},{0,1.1,0},{0,0,1.2}}};
    inputs.push_back(makeInput(shear));inputs.push_back(makeInput(compression));
    inputs.push_back(makeInput(compression,{},2));
    inputs.push_back(makeInput(compression,{},1,8000,12000));
    inputs.push_back(makeInput(product(rotation,compression)));
    inputs.push_back(makeInput(compression,{},1,1000,200000));
    for(int n=0;n<16;++n) {
        M F=identity();
        F[0][0]=.65+.035*n;F[1][1]=.85+.013*n;F[2][2]=1.2-.017*n;
        F[0][1]=.1*std::sin(n);F[1][2]=.2*std::cos(n);F[2][0]=.07*std::sin(2*n);
        inputs.push_back(makeInput(F,{.01*n,-.02*n,.03*n},1,4000,6000));
    }
    size_t validCount=inputs.size();std::vector<uint32_t> expected;
    auto negative=[&](NumiDeformableTetInput input,uint32_t bit){inputs.push_back(input);expected.push_back(bit);};
    auto bad=inputs[0];bad.control.x=0;negative(bad,NUMI_DEFORMABLE_TET_FAILURE_ABI);
    bad=inputs[0];bad.positions[1].x=std::numeric_limits<float>::quiet_NaN();negative(bad,NUMI_DEFORMABLE_TET_FAILURE_NONFINITE);
    bad=inputs[0];bad.inverseRestRows[0].w*=2;negative(bad,NUMI_DEFORMABLE_TET_FAILURE_REST_GEOMETRY);
    bad=inputs[0];bad.material.x=-1;negative(bad,NUMI_DEFORMABLE_TET_FAILURE_MATERIAL);
    bad=inputs[0];bad.positions[3].z*=-1;negative(bad,NUMI_DEFORMABLE_TET_FAILURE_INVERSION);
    bad=inputs[0];bad.positions[3].z=0;negative(bad,NUMI_DEFORMABLE_TET_FAILURE_INVERSION);
    auto first=runGPU(device,pipeline,inputs),second=runGPU(device,pipeline,inputs);
    bool passed=std::memcmp(first.data(),second.data(),first.size()*sizeof(first[0]))==0;
    double maximumForceError=0,maximumEnergyError=0,maximumNetForce=0,maximumTorque=0;
    for(size_t i=0;i<validCount;++i) {
        auto& input=inputs[i];auto& output=first[i];
        std::array<V,4> x{};for(int n=0;n<4;++n)x[n]=xyz(input.positions[n]);
        double expectedEnergy=energy(input,x),forceScale=1,maximumElementForce=0;
        V net{},torque{};
        for(int n=0;n<4;++n) {
            V force=xyz(output.forces[n]);forceScale=std::max(forceScale,norm(force));
            maximumElementForce=std::max(maximumElementForce,norm(force));
            for(int k=0;k<3;++k) {
                auto plus=x,minus=x;double h=1e-6;
                plus[n][k]+=h;minus[n][k]-=h;
                double oracle=-(energy(input,plus)-energy(input,minus))/(2*h);
                maximumForceError=std::max(maximumForceError,std::abs(force[k]-oracle));
                passed=passed&&std::abs(force[k]-oracle)<2e-4+5e-5*std::abs(oracle);
                net[k]+=force[k];
            }
            V moment=cross(sub(x[n],x[0]),force);
            for(int k=0;k<3;++k)torque[k]+=moment[k];
        }
        double e=std::abs(output.energyAndVolumeRatio.x-expectedEnergy);
        maximumEnergyError=std::max(maximumEnergyError,e);
        maximumNetForce=std::max(maximumNetForce,norm(net));maximumTorque=std::max(maximumTorque,norm(torque));
        passed=passed&&output.control.x==0&&std::isfinite(e)&&e<2e-6+2e-5*std::abs(expectedEnergy)&&
            norm(net)<1e-5*forceScale&&norm(torque)<2e-6*forceScale;
        if(i<3)passed=passed&&maximumElementForce<2e-4&&std::abs(output.energyAndVolumeRatio.x)<1e-5;
    }
    const auto close=[](double a,double b){return std::abs(a-b)<1e-5+2e-5*std::abs(b);};
    // Geometry scaling doubles lengths: force scales as area, energy as volume.
    for(int n=0;n<4;++n)for(int k=0;k<3;++k) {
        passed=passed&&close(xyz(first[5].forces[n])[k],4*xyz(first[4].forces[n])[k]);
        passed=passed&&close(xyz(first[6].forces[n])[k],2*xyz(first[4].forces[n])[k]);
        V transformed=mul(rotation,xyz(first[4].forces[n]));
        passed=passed&&close(xyz(first[7].forces[n])[k],transformed[k]);
    }
    passed=passed&&close(first[5].energyAndVolumeRatio.x,8*first[4].energyAndVolumeRatio.x)&&
        close(first[6].energyAndVolumeRatio.x,2*first[4].energyAndVolumeRatio.x)&&
        close(first[7].energyAndVolumeRatio.x,first[4].energyAndVolumeRatio.x);
    for(size_t n=0;n<expected.size();++n) {
        auto& output=first[validCount+n];
        passed=passed&&(output.control.x&expected[n])!=0&&output.energyAndVolumeRatio.x==0;
        for(auto force:output.forces)passed=passed&&norm(xyz(force))==0;
    }
    auto advance=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"numi_deformable_tet_advance"] error:&error];
    if(!advance)throw std::runtime_error(error.localizedDescription.UTF8String);
    bool rollbackExact=rejectedAdvancePreservesState(device,advance,inputs[0],true)&&
        rejectedAdvancePreservesState(device,advance,inputs[0],false);
    passed=passed&&rollbackExact;
    std::vector<NumiDeformableTetInput> bodies{makeInput(identity(),{0,0,2}),makeInput(compression,{.4,0,2}),makeInput(shear,{.8,0,2})};
    auto motion=simulate(device,advance,bodies,1e-4f),replayed=simulate(device,advance,bodies,1e-4f),refined=simulate(device,advance,bodies,5e-5f);
    bool constitutiveQualified=passed;
    bool motionExact=true,allStepsValid=true;double maximumEnergyDrift=0,maximumComError=0,maximumDeformationChange=0,minimumJ=1e30,maximumRefinementDifference=0;
    for(size_t frame=0;frame<motion.size();++frame) {
        motionExact=motionExact&&std::memcmp(motion[frame].elements.data(),replayed[frame].elements.data(),bodies.size()*sizeof(bodies[0]))==0&&
            std::memcmp(motion[frame].velocityAndMass.data(),replayed[frame].velocityAndMass.data(),4*bodies.size()*sizeof(mr_float4))==0&&
            std::memcmp(motion[frame].outputs.data(),replayed[frame].outputs.data(),bodies.size()*sizeof(NumiDeformableTetOutput))==0;
        for(size_t element=0;element<bodies.size();++element) {
            V initialCenter=center(motion[0],element),currentCenter=center(motion[frame],element);
            double t=frame*.01,expectedHeight=initialCenter[2]-.5*9.81f*t*(t+1e-4f);
            maximumComError=std::max(maximumComError,std::abs(currentCenter[2]-expectedHeight));
            V refinedCenter=center(refined[frame],element);
            maximumComError=std::max(maximumComError,std::abs(refinedCenter[2]-(initialCenter[2]-.5*9.81f*t*(t+5e-5f))));
            double drift=std::abs(mechanicalEnergy(motion[frame],element)-mechanicalEnergy(motion[0],element));
            maximumEnergyDrift=std::max(maximumEnergyDrift,drift);
            maximumEnergyDrift=std::max(maximumEnergyDrift,std::abs(mechanicalEnergy(refined[frame],element)-mechanicalEnergy(refined[0],element)));
            if(frame>0) {
                allStepsValid=allStepsValid&&motion[frame].outputs[element].control.x==0&&refined[frame].outputs[element].control.x==0&&
                    motion[frame].outputs[element].control.y==0&&refined[frame].outputs[element].control.y==0;
                minimumJ=std::min({minimumJ,double(motion[frame].outputs[element].energyAndVolumeRatio.y),double(refined[frame].outputs[element].energyAndVolumeRatio.y)});
            }
            if(element>0)for(int n=0;n<4;++n) {
                V original=sub(xyz(motion[0].elements[element].positions[n]),initialCenter);
                V current=sub(xyz(motion[frame].elements[element].positions[n]),currentCenter);
                maximumDeformationChange=std::max(maximumDeformationChange,norm(sub(original,current)));
                V fine=sub(xyz(refined[frame].elements[element].positions[n]),refinedCenter);
                maximumRefinementDifference=std::max(maximumRefinementDifference,norm(sub(current,fine)));
            }
        }
    }
    bool dynamicsQualified=motionExact&&allStepsValid&&rollbackExact&&minimumJ>0&&maximumComError<5e-4&&maximumEnergyDrift<.003&&maximumDeformationChange>.001&&maximumRefinementDifference<5e-4;
    passed=passed&&dynamicsQualified;
    exportFrames(trajectoryPrefix,motion);
    std::cout<<std::setprecision(12)<<"device="<<device.name.UTF8String<<" backend=Apple_Metal\n"
             <<"valid_elements="<<validCount<<" rejected_elements="<<expected.size()<<" replay_exact="
             <<(std::memcmp(first.data(),second.data(),first.size()*sizeof(first[0]))==0)<<'\n'
             <<"maximum_force_gradient_error_N="<<maximumForceError<<" maximum_energy_error_J="<<maximumEnergyError
             <<" maximum_net_force_N="<<maximumNetForce<<" maximum_torque_Nm="<<maximumTorque<<'\n'
             <<"dynamic_elements=3 simulated_seconds=.5 steps=5000 refined_steps=10000 captured_frames="<<motion.size()<<" dynamic_replay_exact="<<motionExact<<'\n'
             <<"maximum_mechanical_energy_drift_J="<<maximumEnergyDrift<<" maximum_COM_free_fall_error_m="<<maximumComError
             <<" minimum_sampled_volume_ratio="<<minimumJ<<" maximum_relative_shape_change_m="<<maximumDeformationChange<<'\n'
             <<"maximum_relative_shape_refinement_difference_m="<<maximumRefinementDifference<<'\n'
             <<"all_advanced_candidates_valid="<<allStepsValid<<'\n'
             <<"rejected_advance_state_exact="<<rollbackExact<<" deformable_dynamics_qualified="<<dynamicsQualified<<'\n'
             <<"deformable_constitutive_qualified="<<constitutiveQualified<<" deformable_qualified="<<passed<<'\n';
    return passed?0:1;
}
}
int main(int argc,const char* const* argv) {
    @autoreleasepool {
        try {
            std::string prefix;
            if(argc==3&&std::string(argv[1])=="--trajectory")prefix=argv[2];
            else if(argc!=1)throw std::runtime_error("usage: numi-solver-deformable-tet [--trajectory PREFIX]");
            return run(prefix);
        }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
    }
}

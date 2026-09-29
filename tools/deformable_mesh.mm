#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/deformable_mesh.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V = std::array<double,3>;
using Face = std::array<uint32_t,3>;
V xyz(mr_float4 p){return {p.x,p.y,p.z};}
mr_float4 f4(V p,double w=0){return {float(p[0]),float(p[1]),float(p[2]),float(w)};}
uint32_t corner(mr_uint4 ids,int n){return std::array<uint32_t,4>{ids.x,ids.y,ids.z,ids.w}[n];}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double norm(V a){return std::sqrt(dot(a,a));}
struct Mesh {
    std::vector<mr_float4> positions,velocityAndMass;
    std::vector<NumiDeformableMeshElement> elements;
    std::vector<Face> faces;
    std::vector<uint32_t> offsets;
    std::vector<mr_uint4> incidence;
};
Mesh makeMesh(const uint32_t refinement=0,const bool shortestDiagonal=false){
    Mesh mesh;double phi=(1+std::sqrt(5.0))/2,radius=.05;
    std::vector<V> surface{{-1,phi,0},{1,phi,0},{-1,-phi,0},{1,-phi,0},
        {0,-1,phi},{0,1,phi},{0,-1,-phi},{0,1,-phi},
        {phi,0,-1},{phi,0,1},{-phi,0,-1},{-phi,0,1}};
    for(auto& p:surface){double scale=radius/norm(p);for(auto& x:p)x*=scale;}
    mesh.positions.push_back({0,0,.17f,0});
    for(auto p:surface){p[2]+=.17;mesh.positions.push_back(f4(p));}
    // Derive the convex boundary from the authored vertices, then orient it
    // outward. Each face and the center own one positive-volume tetrahedron.
    for(uint32_t a=0;a<12;++a)for(uint32_t b=a+1;b<12;++b)for(uint32_t c=b+1;c<12;++c){
        V normal=cross(sub(surface[b],surface[a]),sub(surface[c],surface[a]));
        if(norm(normal)<1e-12)continue;
        Face face{a+1,b+1,c+1};
        if(dot(normal,surface[a])<0){for(auto& x:normal)x=-x;std::swap(face[1],face[2]);}
        bool boundary=true;
        for(auto p:surface)if(dot(normal,sub(p,surface[a]))>1e-10)boundary=false;
        if(boundary)mesh.faces.push_back(face);
    }
    if(mesh.faces.size()!=20)throw std::runtime_error("icosahedron boundary is not twenty triangles");
    std::vector<std::array<uint32_t,4>> tets;
    for(auto face:mesh.faces)tets.push_back({0,face[0],face[1],face[2]});
    for(uint32_t level=0;level<refinement;++level){
        // Shared midpoints preserve the exact original piecewise-flat body.
        // No radius projection or changed material is hidden in refinement.
        std::map<std::pair<uint32_t,uint32_t>,uint32_t> midpointNodes;
        auto midpoint=[&](uint32_t first,uint32_t second){
            auto edge=std::minmax(first,second);
            auto key=std::pair{edge.first,edge.second};
            auto found=midpointNodes.find(key);if(found!=midpointNodes.end())return found->second;
            V point{};for(int k=0;k<3;++k)point[k]=.5*(xyz(mesh.positions[first])[k]+xyz(mesh.positions[second])[k]);
            uint32_t node=uint32_t(mesh.positions.size());mesh.positions.push_back(f4(point));
            midpointNodes.emplace(key,node);return node;
        };
        std::vector<std::array<uint32_t,4>> children;
        for(auto tet:tets){
            uint32_t a=tet[0],b=tet[1],c=tet[2],d=tet[3];
            uint32_t ab=midpoint(a,b),ac=midpoint(a,c),ad=midpoint(a,d),
                     bc=midpoint(b,c),bd=midpoint(b,d),cd=midpoint(c,d);
            // The four central children tile the midpoint octahedron. Always
            // choosing ab-cd can progressively lengthen their edges under
            // refinement. The explicit shortest-diagonal candidate preserves
            // the exterior faces and volume while bounding this deterioration.
            const std::array<std::array<std::array<uint32_t,4>,4>,3> central{{
                {{{ab,ac,ad,cd},{ab,ac,bc,cd},{ab,ad,bd,cd},{ab,bc,bd,cd}}},
                {{{ac,ab,ad,bd},{ac,ad,cd,bd},{ac,cd,bc,bd},{ac,bc,ab,bd}}},
                {{{ad,ab,ac,bc},{ad,ac,cd,bc},{ad,cd,bd,bc},{ad,bd,ab,bc}}}
            }};
            uint32_t diagonal=0;
            if(shortestDiagonal){
                const std::array<std::pair<uint32_t,uint32_t>,3> choices{{{ab,cd},{ac,bd},{ad,bc}}};
                double shortest=dot(sub(xyz(mesh.positions[ab]),xyz(mesh.positions[cd])),
                                    sub(xyz(mesh.positions[ab]),xyz(mesh.positions[cd])));
                for(uint32_t choice=1;choice<choices.size();++choice){
                    const auto [first,second]=choices[choice];
                    const V edge=sub(xyz(mesh.positions[first]),xyz(mesh.positions[second]));
                    const double squared=dot(edge,edge);
                    if(squared<shortest){shortest=squared;diagonal=choice;}
                }
            }
            for(auto child:std::array<std::array<uint32_t,4>,8>{{
                {a,ab,ac,ad},{b,ab,bc,bd},{c,ac,bc,cd},{d,ad,bd,cd},
                central[diagonal][0],central[diagonal][1],central[diagonal][2],central[diagonal][3]}}){
                V x=sub(xyz(mesh.positions[child[1]]),xyz(mesh.positions[child[0]]));
                V y=sub(xyz(mesh.positions[child[2]]),xyz(mesh.positions[child[0]]));
                V z=sub(xyz(mesh.positions[child[3]]),xyz(mesh.positions[child[0]]));
                if(dot(x,cross(y,z))<0)std::swap(child[1],child[2]);
                children.push_back(child);
            }
        }
        std::vector<Face> boundary;
        for(auto face:mesh.faces){
            uint32_t a=face[0],b=face[1],c=face[2],ab=midpoint(a,b),bc=midpoint(b,c),ac=midpoint(a,c);
            for(auto child:std::array<Face,4>{{{a,ab,ac},{ab,b,bc},{ac,bc,c},{ab,bc,ac}}})boundary.push_back(child);
        }
        tets=std::move(children);mesh.faces=std::move(boundary);
    }
    std::vector<double> masses(mesh.positions.size());
    for(auto tet:tets){
        NumiDeformableMeshElement element{};
        element.nodes={tet[0],tet[1],tet[2],tet[3]};
        V a=sub(xyz(mesh.positions[tet[1]]),xyz(mesh.positions[tet[0]]));
        V b=sub(xyz(mesh.positions[tet[2]]),xyz(mesh.positions[tet[0]]));
        V c=sub(xyz(mesh.positions[tet[3]]),xyz(mesh.positions[tet[0]]));
        double det=dot(a,cross(b,c)),volume=det/6;
        if(!(volume>0))throw std::runtime_error("invalid authored tetrahedron");
        std::array<V,3> rows{cross(b,c),cross(c,a),cross(a,b)};
        for(int row=0;row<3;++row){for(auto& x:rows[row])x/=det;
            element.inverseRestRows[row]=f4(rows[row],row==0?volume:0);}
        element.material={3000,6000,0,0};element.control.x=NUMI_DEFORMABLE_TET_ABI_VERSION;
        mesh.elements.push_back(element);
        for(int n=0;n<4;++n)masses[corner(element.nodes,n)]+=1000*element.inverseRestRows[0].w/4;
    }
    for(double mass:masses)mesh.velocityAndMass.push_back({0,0,0,float(mass)});
    std::vector<std::vector<mr_uint4>> nodeIncidence(mesh.positions.size());
    for(uint32_t element=0;element<mesh.elements.size();++element)for(uint32_t n=0;n<4;++n){
        const uint32_t node=corner(mesh.elements[element].nodes,n);
        nodeIncidence[node].push_back({element,n,node,0});
    }
    for(uint32_t node=0;node<mesh.positions.size();++node){
        mesh.offsets.push_back(uint32_t(mesh.incidence.size()));
        // Appending in element/corner order preserves the previous sorted
        // gather exactly, without rescanning every element for every node.
        mesh.incidence.insert(mesh.incidence.end(),nodeIncidence[node].begin(),nodeIncidence[node].end());
    }
    mesh.offsets.push_back(uint32_t(mesh.incidence.size()));
    if(mesh.incidence.size()!=4*mesh.elements.size())throw std::runtime_error("incorrect incidence count");
    std::map<std::pair<uint32_t,uint32_t>,uint32_t> directedEdges;
    for(auto face:mesh.faces)for(int edge=0;edge<3;++edge)
        ++directedEdges[{face[edge],face[(edge+1)%3]}];
    for(auto [edge,count]:directedEdges){
        auto reverse=directedEdges.find({edge.second,edge.first});
        if(count!=1||reverse==directedEdges.end()||reverse->second!=1)
            throw std::runtime_error("boundary is not an oriented manifold");
    }
    return mesh;
}
struct Frame {
    std::vector<mr_float4> positions,velocities;
    NumiDeformableMeshStatus status;
};
struct Pipelines {
    id<MTLComputePipelineState> evaluate,predict,finishVerlet,validate,commit;
};
std::vector<Frame> simulate(id<MTLDevice> device,Pipelines pipelines,const Mesh& mesh,float dt,bool plane=true,
                          uint32_t stepLimit=0,uint32_t abi=NUMI_DEFORMABLE_MESH_ABI_VERSION,
                          Frame* lastFreePrediction=nullptr,
                          uint32_t integrator=NUMI_DEFORMABLE_MESH_INTEGRATOR_EULER,
                          const char* phase=nullptr){
    auto buffer=[&](const void* data,size_t bytes){
        id<MTLBuffer> result=data?[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]:
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if(!result)throw std::runtime_error("mesh allocation failed");
        if(!data)std::memset(result.contents,0,bytes);return result;
    };
    auto elements=buffer(mesh.elements.data(),mesh.elements.size()*sizeof(mesh.elements[0]));
    auto positions=buffer(mesh.positions.data(),mesh.positions.size()*sizeof(mr_float4));
    auto velocities=buffer(mesh.velocityAndMass.data(),mesh.velocityAndMass.size()*sizeof(mr_float4));
    auto previous=buffer(nullptr,mesh.elements.size()*sizeof(NumiDeformableTetOutput));
    auto candidate=buffer(nullptr,mesh.elements.size()*sizeof(NumiDeformableTetOutput));
    auto freeOutputs=buffer(nullptr,mesh.elements.size()*sizeof(NumiDeformableTetOutput));
    auto offsets=buffer(mesh.offsets.data(),mesh.offsets.size()*sizeof(uint32_t));
    auto incidence=buffer(mesh.incidence.data(),mesh.incidence.size()*sizeof(mr_uint4));
    auto candidatePositions=buffer(nullptr,mesh.positions.size()*sizeof(mr_float4));
    auto candidateVelocities=buffer(nullptr,mesh.positions.size()*sizeof(mr_float4));
    auto freePositions=buffer(nullptr,mesh.positions.size()*sizeof(mr_float4));
    auto freeVelocities=buffer(nullptr,mesh.positions.size()*sizeof(mr_float4));
    auto contact=buffer(nullptr,mesh.positions.size()*sizeof(mr_float4));
    NumiDeformableMeshStatus initial{};initial.ledger.y=1;
    auto status=buffer(&initial,sizeof(initial));
    auto queue=[device newCommandQueue];if(!queue)throw std::runtime_error("mesh command queue failed");
    NumiDeformableMeshConfig config{{uint32_t(mesh.positions.size()),uint32_t(mesh.elements.size()),uint32_t(mesh.incidence.size()),abi},
        {0,0,-9.81f,dt},{0,plane?1.0f:0.0f,0,0},{integrator,0,0,0}};
    std::vector<Frame> frames;
    auto capture=[&]{Frame frame{std::vector<mr_float4>(mesh.positions.size()),std::vector<mr_float4>(mesh.positions.size()),{}};
        std::memcpy(frame.positions.data(),positions.contents,positions.length);
        std::memcpy(frame.velocities.data(),velocities.contents,velocities.length);
        std::memcpy(&frame.status,status.contents,sizeof(initial));frames.push_back(frame);};
    capture();uint32_t steps=stepLimit?stepLimit:uint32_t(std::lround(.5/dt)),captureInterval=std::max(1u,steps/100);
    // Keep large-mesh command buffers bounded while preserving the same 101
    // capture times and the ordered accepted-step sequence.
    uint32_t batch=mesh.positions.size()>2057?std::min(50u,captureInterval):captureInterval;
    for(uint32_t completed=0;completed<steps;completed+=batch){
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setBytes:&config length:sizeof(config) atIndex:0];
        [encoder setBuffer:elements offset:0 atIndex:1];[encoder setBuffer:positions offset:0 atIndex:2];
        [encoder setBuffer:velocities offset:0 atIndex:3];[encoder setBuffer:previous offset:0 atIndex:4];
        [encoder setBuffer:offsets offset:0 atIndex:5];[encoder setBuffer:incidence offset:0 atIndex:6];
        [encoder setBuffer:candidatePositions offset:0 atIndex:7];[encoder setBuffer:candidateVelocities offset:0 atIndex:8];
        [encoder setBuffer:contact offset:0 atIndex:9];[encoder setBuffer:status offset:0 atIndex:10];
        [encoder setBuffer:candidate offset:0 atIndex:11];
        [encoder setBuffer:freePositions offset:0 atIndex:12];[encoder setBuffer:freeVelocities offset:0 atIndex:13];
        [encoder setBuffer:freeOutputs offset:0 atIndex:14];
        auto dispatch=[&](id<MTLComputePipelineState> pipeline,uint32_t count){
            [encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];};
        for(uint32_t step=0;step<batch;++step){
            dispatch(pipelines.evaluate,config.counts.y);dispatch(pipelines.predict,config.counts.x);
            [encoder setBuffer:freePositions offset:0 atIndex:2];[encoder setBuffer:freeOutputs offset:0 atIndex:4];
            dispatch(pipelines.evaluate,config.counts.y);
            [encoder setBuffer:candidatePositions offset:0 atIndex:2];[encoder setBuffer:candidate offset:0 atIndex:4];
            dispatch(pipelines.evaluate,config.counts.y);
            [encoder setBuffer:positions offset:0 atIndex:2];[encoder setBuffer:previous offset:0 atIndex:4];
            if(integrator==NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET)
                dispatch(pipelines.finishVerlet,config.counts.x);
            dispatch(pipelines.validate,1);dispatch(pipelines.commit,config.counts.x);
        }
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        if((completed+batch)%captureInterval==0){
            capture();
            if(phase&&(frames.size()-1)%20==0)
                std::cerr<<"mesh_progress phase="<<phase<<" captured="<<frames.size()-1<<"/100 accepted_steps="
                    <<frames.back().status.control.w<<" rejected_steps="<<frames.back().status.control.z
                    <<" minimum_volume_ratio="<<frames.back().status.ledger.y<<'\n';
        }
    }
    if(lastFreePrediction){
        lastFreePrediction->positions.resize(mesh.positions.size());
        lastFreePrediction->velocities.resize(mesh.positions.size());
        std::memcpy(lastFreePrediction->positions.data(),freePositions.contents,freePositions.length);
        std::memcpy(lastFreePrediction->velocities.data(),freeVelocities.contents,freeVelocities.length);
    }
    return frames;
}
double elasticEnergy(const Mesh& mesh,const Frame& frame){
    double total=0;
    for(auto element:mesh.elements){
        std::array<V,3> edges{};
        for(int n=0;n<3;++n)edges[n]=sub(xyz(frame.positions[corner(element.nodes,n+1)]),xyz(frame.positions[element.nodes.x]));
        std::array<V,3> F{};
        for(int column=0;column<3;++column)for(int row=0;row<3;++row)for(int k=0;k<3;++k)
            F[column][row]+=edges[k][row]*xyz(element.inverseRestRows[k])[column];
        double J=dot(F[0],cross(F[1],F[2])),invariant=dot(F[0],F[0])+dot(F[1],F[1])+dot(F[2],F[2]);
        if(!(J>0))throw std::runtime_error("exported mesh inverted");
        double l=std::log(J);
        total+=element.inverseRestRows[0].w*(.5*element.material.x*(invariant-3)-element.material.x*l+.5*element.material.y*l*l);
    }
    return total;
}
double mechanicalEnergy(const Mesh& mesh,const Frame& frame){
    double total=elasticEnergy(mesh,frame);
    for(size_t n=0;n<frame.positions.size();++n){auto v=frame.velocities[n];
        total+=.5*v.w*dot(xyz(v),xyz(v))+v.w*9.81f*frame.positions[n].z;}
    return total;
}
V center(const Frame& frame){
    V c{};double mass=0;
    for(size_t n=0;n<frame.positions.size();++n){double m=frame.velocities[n].w;mass+=m;
        for(int k=0;k<3;++k)c[k]+=m*xyz(frame.positions[n])[k];}
    for(auto& x:c)x/=mass;return c;
}
double height(const Frame& frame){
    double low=1e30,high=-1e30;for(auto p:frame.positions){low=std::min(low,double(p.z));high=std::max(high,double(p.z));}return high-low;
}
void exportFrames(const std::string& prefix,const Mesh& mesh,const std::vector<Frame>& frames,uint32_t integrator){
    if(prefix.empty())return;
    std::ofstream csv(prefix+".csv");if(!csv)throw std::runtime_error("cannot write mesh trajectory");
    csv<<"frame,time_s,node,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,mass_kg\n"<<std::setprecision(12);
    std::ofstream topology(prefix+"-topology.csv");if(!topology)throw std::runtime_error("cannot write mesh topology");
    std::ofstream energy(prefix+"-energy.csv");if(!energy)throw std::runtime_error("cannot write mesh energy ledger");
    energy<<"frame,time_s,accepted_steps,rejected_steps,normal_kinetic_loss_J,contact_elastic_change_J,contact_gravity_change_J,contact_mechanical_change_J,maximum_positive_contact_step_J,free_integration_change_J,absolute_free_integration_change_J,maximum_absolute_free_step_J,elastic_force_work_J,absolute_position_projection_work_J,maximum_position_projection_step_J,maximum_absolute_contact_step_J,positive_contact_work_J,contact_kinetic_change_J,contact_force_response_change_J,absolute_contact_force_response_change_J,maximum_contact_force_response_step_J,integrator,independent_mechanical_energy_J\n"<<std::setprecision(12);
    topology<<"element,node0,node1,node2,node3,rest_volume_m3,mu_Pa,lambda_Pa\n"<<std::setprecision(12);
    for(size_t e=0;e<mesh.elements.size();++e){auto tet=mesh.elements[e];
        topology<<e<<','<<tet.nodes.x<<','<<tet.nodes.y<<','<<tet.nodes.z<<','<<tet.nodes.w<<','<<tet.inverseRestRows[0].w<<','<<tet.material.x<<','<<tet.material.y<<'\n';}
    for(size_t f=0;f<frames.size();++f){
        const auto& s=frames[f].status;
        energy<<f<<','<<f*.005<<','<<s.control.w<<','<<s.control.z<<','<<s.ledger.w<<','
            <<s.contactWork.x<<','<<s.contactWork.y<<','<<s.contactWork.z<<','<<s.contactWork.w<<','
            <<s.integration.x<<','<<s.integration.y<<','<<s.integration.z<<','<<s.integration.w<<','
            <<s.projectionBounds.x<<','<<s.projectionBounds.y<<','<<s.projectionBounds.z<<','<<s.projectionBounds.w<<','
            <<s.contactKinetic.x<<','<<s.contactKinetic.y<<','<<s.contactKinetic.z<<','<<s.contactKinetic.w<<','<<integrator<<','
            <<mechanicalEnergy(mesh,frames[f])<<'\n';
        std::ofstream obj(prefix+"-"+std::to_string(f)+".obj");if(!obj)throw std::runtime_error("cannot write mesh OBJ");
        obj<<"# Native shared-node nonlinear elastic mesh and frictionless inelastic plane\n"<<std::setprecision(12);
        for(size_t n=0;n<frames[f].positions.size();++n){auto p=frames[f].positions[n],v=frames[f].velocities[n];
            obj<<"v "<<p.x<<' '<<p.y<<' '<<p.z<<'\n';
            csv<<f<<','<<f*.005<<','<<n<<','<<p.x<<','<<p.y<<','<<p.z<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\n';}
        for(auto face:mesh.faces)obj<<"f "<<face[0]+1<<' '<<face[1]+1<<' '<<face[2]+1<<'\n';
    }
}
int topologyProbe(const bool shortestDiagonal=false){
    bool pass=true;
    const std::array<uint32_t,5> expectedNodes{13,55,309,2057,14993};
    for(uint32_t level=0;level<expectedNodes.size();++level){
        const auto mesh=makeMesh(level,shortestDiagonal);
        std::vector<mr_uint4> legacy;
        std::vector<uint32_t> offsets;
        if(level<4)for(uint32_t node=0;node<mesh.positions.size();++node){
            offsets.push_back(uint32_t(legacy.size()));
            for(uint32_t element=0;element<mesh.elements.size();++element)for(uint32_t n=0;n<4;++n)
                if(corner(mesh.elements[element].nodes,n)==node)legacy.push_back({element,n,node,0});
        }
        if(level==4){
            // Independent flat corner sort has the node/element/corner order
            // of the legacy scan without its node-times-element cost.
            for(uint32_t e=0;e<mesh.elements.size();++e)for(uint32_t n=0;n<4;++n)
                legacy.push_back({e,n,corner(mesh.elements[e].nodes,n),0});
            std::sort(legacy.begin(),legacy.end(),[](auto a,auto b){
                return a.z!=b.z?a.z<b.z:(a.x!=b.x?a.x<b.x:a.y<b.y);});
            size_t entry=0;
            for(uint32_t node=0;node<mesh.positions.size();++node){
                offsets.push_back(uint32_t(entry));
                while(entry<legacy.size()&&legacy[entry].z==node)++entry;
            }
        }
        offsets.push_back(uint32_t(legacy.size()));
        const bool exact=offsets==mesh.offsets&&legacy.size()==mesh.incidence.size()&&
            std::memcmp(legacy.data(),mesh.incidence.data(),legacy.size()*sizeof(mr_uint4))==0;
        const bool counts=mesh.positions.size()==expectedNodes[level]&&
            mesh.elements.size()==(20u<<(3*level))&&mesh.faces.size()==(20u<<(2*level));
        pass&=exact&&counts;
        std::cout<<"mesh_topology_refinement="<<level<<" shared_nodes="<<mesh.positions.size()
            <<" tetrahedra="<<mesh.elements.size()<<" boundary_triangles="<<mesh.faces.size()
            <<" sorted_incidence_reference_exact="<<exact
            <<" reference="<<(level<4?"legacy_scan":"independent_flat_corner_sort")<<" expected_counts="<<counts<<'\n';
        if(shortestDiagonal){
            const auto legacyMesh=makeMesh(level);
            using TriangleGeometry=std::array<V,3>;
            auto boundaryGeometry=[](const Mesh& body){
                std::vector<TriangleGeometry> geometry;
                for(auto face:body.faces){TriangleGeometry points{xyz(body.positions[face[0]]),
                    xyz(body.positions[face[1]]),xyz(body.positions[face[2]])};
                    std::sort(points.begin(),points.end());geometry.push_back(points);}
                std::sort(geometry.begin(),geometry.end());return geometry;
            };
            const bool boundaryExact=boundaryGeometry(mesh)==boundaryGeometry(legacyMesh);
            // Independently reconstruct the element boundary. Counts and
            // sorted incidence alone cannot detect duplicate or missing
            // octahedron children.
            auto canonical=[](Face face){return std::min({face,Face{face[1],face[2],face[0]},
                Face{face[2],face[0],face[1]}});};
            std::map<Face,std::vector<Face>> owners;
            for(auto element:mesh.elements){
                const uint32_t a=element.nodes.x,b=element.nodes.y,c=element.nodes.z,d=element.nodes.w;
                for(const Face face:std::array<Face,4>{{{b,c,d},{a,d,c},{a,b,d},{a,c,b}}}){
                    Face key=face;std::sort(key.begin(),key.end());owners[key].push_back(face);
                }
            }
            bool internalOwnersValid=true;
            std::vector<Face> extractedBoundary,authoredBoundary;
            for(const auto& [key,faces]:owners){
                if(faces.size()==1)extractedBoundary.push_back(canonical(faces[0]));
                else if(faces.size()!=2||canonical(faces[0])!=canonical(Face{faces[1][2],faces[1][1],faces[1][0]}))
                    internalOwnersValid=false;
            }
            for(auto face:mesh.faces)authoredBoundary.push_back(canonical(face));
            std::sort(extractedBoundary.begin(),extractedBoundary.end());
            std::sort(authoredBoundary.begin(),authoredBoundary.end());
            internalOwnersValid&=extractedBoundary==authoredBoundary;
            bool originalNodesExact=true;
            for(uint32_t node=0;node<13;++node)
                originalNodesExact&=xyz(mesh.positions[node])==xyz(legacyMesh.positions[node]);
            double mass=0,legacyMass=0,minimumQuality=1,maximumEdgeAspect=1;
            for(auto v:mesh.velocityAndMass)mass+=v.w;
            for(auto v:legacyMesh.velocityAndMass)legacyMass+=v.w;
            for(auto element:mesh.elements){
                double squaredEdges=0,shortest=1,longest=0;
                for(uint32_t first=0;first<4;++first)for(uint32_t second=first+1;second<4;++second){
                    const V edge=sub(xyz(mesh.positions[corner(element.nodes,first)]),
                                     xyz(mesh.positions[corner(element.nodes,second)]));
                    const double squared=dot(edge,edge);
                    squaredEdges+=squared;shortest=std::min(shortest,squared);longest=std::max(longest,squared);
                }
                minimumQuality=std::min(minimumQuality,12*std::pow(3*double(element.inverseRestRows[0].w),2./3)/squaredEdges);
                maximumEdgeAspect=std::max(maximumEdgeAspect,std::sqrt(longest/shortest));
            }
            const bool geometryPreserved=boundaryExact&&originalNodesExact&&std::abs(mass-legacyMass)<2e-7;
            const bool shapeQualityPreserved=minimumQuality>.84&&maximumEdgeAspect<1.46;
            pass&=geometryPreserved&&internalOwnersValid&&shapeQualityPreserved;
            std::cout<<std::setprecision(12)<<"refinement_diagonal=shortest original_nodes_exact="<<originalNodesExact
                <<" boundary_geometry_exact="<<boundaryExact<<" mass_difference_kg="<<std::abs(mass-legacyMass)
                <<" minimum_tet_mean_ratio="<<minimumQuality<<" maximum_tet_edge_aspect="<<maximumEdgeAspect
                <<" physical_rest_geometry_preserved="<<geometryPreserved
                <<" internal_face_owners_valid="<<internalOwnersValid
                <<" shape_quality_preserved="<<shapeQualityPreserved<<'\n';
        }
    }
    std::cout<<"mesh_topology_probe_qualified="<<pass<<'\n';
    return pass?0:1;
}

int run(const std::string& prefix,const uint32_t refinement,const float timestep,const uint32_t integrator,
        const bool shortestDiagonal=false){
    auto device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("Metal device unavailable");
    NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_DEFORMABLE_MESH_METALLIB] error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto pipeline=[&](NSString* name){auto result=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
        if(!result)throw std::runtime_error(error.localizedDescription.UTF8String);return result;};
    Pipelines pipelines{pipeline(@"numi_deformable_mesh_evaluate"),pipeline(@"numi_deformable_mesh_predict"),
        pipeline(@"numi_deformable_mesh_finish_verlet"),
        pipeline(@"numi_deformable_mesh_validate"),pipeline(@"numi_deformable_mesh_commit")};
    auto mesh=makeMesh(refinement,shortestDiagonal);
    auto motion=simulate(device,pipelines,mesh,timestep,true,0,NUMI_DEFORMABLE_MESH_ABI_VERSION,nullptr,integrator,"primary"),
         replay=simulate(device,pipelines,mesh,timestep,true,0,NUMI_DEFORMABLE_MESH_ABI_VERSION,nullptr,integrator,"replay"),
         fine=simulate(device,pipelines,mesh,timestep/2,true,0,NUMI_DEFORMABLE_MESH_ABI_VERSION,nullptr,integrator,"half_timestep");
    bool rollbackExact=true;uint32_t rejectedCases=0;
    auto reject=[&](Mesh malformed,uint32_t bit,uint32_t abi=NUMI_DEFORMABLE_MESH_ABI_VERSION,bool invalidIntegrator=false){
        auto states=simulate(device,pipelines,malformed,1e-4f,true,1,abi,nullptr,invalidIntegrator?2:integrator);
        auto last=states.back();
        bool valid=(last.status.control.y&bit)!=0&&last.status.control.w==0&&last.status.control.z==1&&
            std::memcmp(states[0].positions.data(),last.positions.data(),mesh.positions.size()*sizeof(mr_float4))==0&&
            std::memcmp(states[0].velocities.data(),last.velocities.data(),mesh.positions.size()*sizeof(mr_float4))==0&&
            std::memcmp(&states[0].status.ledger,&last.status.ledger,
                sizeof(NumiDeformableMeshStatus)-offsetof(NumiDeformableMeshStatus,ledger))==0;
        rollbackExact=rollbackExact&&valid;rejectedCases+=valid;
    };
    auto malformed=mesh;malformed.velocityAndMass[3].w=0;reject(malformed,NUMI_DEFORMABLE_MESH_FAILURE_STATE);
    malformed=mesh;malformed.velocityAndMass[0].x=2000;reject(malformed,NUMI_DEFORMABLE_TET_FAILURE_INVERSION);
    malformed=mesh;malformed.elements[0].nodes.x=uint32_t(mesh.positions.size());reject(malformed,NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY);
    malformed=mesh;malformed.incidence[1]=malformed.incidence[0];reject(malformed,NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY);
    reject(mesh,NUMI_DEFORMABLE_TET_FAILURE_ABI,0);
    reject(mesh,NUMI_DEFORMABLE_MESH_FAILURE_STATE,NUMI_DEFORMABLE_MESH_ABI_VERSION,true);
    // Independent FP64 reconstruction from the actual pre-projection buffers
    // distinguishes elastic/gravity position work from velocity-only loss.
    auto supported=mesh;float low=1;for(auto p:supported.positions)low=std::min(low,p.z);
    for(auto& p:supported.positions)p.z-=low;
    for(auto& v:supported.velocityAndMass)v.z=-1;
    Frame freePrediction;
    auto contactProbe=simulate(device,pipelines,supported,1e-4f,true,1,NUMI_DEFORMABLE_MESH_ABI_VERSION,&freePrediction,integrator);
    const auto& probe=contactProbe.back();
    double probeElastic=elasticEnergy(supported,probe)-elasticEnergy(supported,freePrediction);
    double probeGravity=0,probeKinetic=0;
    for(size_t n=0;n<supported.positions.size();++n){
        auto a=probe.velocities[n],b=freePrediction.velocities[n];
        probeGravity+=a.w*9.81f*(probe.positions[n].z-freePrediction.positions[n].z);
        probeKinetic+=.5*a.w*(dot(xyz(a),xyz(a))-dot(xyz(b),xyz(b)));
    }
    double probeError=std::max({std::abs(probeElastic-probe.status.contactWork.x),
        std::abs(probeGravity-probe.status.contactWork.y),std::abs(probeKinetic-probe.status.contactKinetic.x),
        std::abs(mechanicalEnergy(supported,probe)-mechanicalEnergy(supported,freePrediction)-probe.status.contactWork.z)});
    bool probeValid=probe.status.control.w==1&&probe.status.control.y==0&&probeGravity>0&&
        std::abs(probeElastic)>1e-7&&probeKinetic<0&&probeError<1e-5;
    bool restingSupportValid=true;double restingImpulse=0,restingKineticLoss=0,restingPositionWork=0;
    if(integrator==NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET){
        for(auto& v:supported.velocityAndMass)v.z=0;
        auto rest=simulate(device,pipelines,supported,1e-4f,true,1,NUMI_DEFORMABLE_MESH_ABI_VERSION,nullptr,integrator).back();
        restingImpulse=rest.status.ledger.x;restingKineticLoss=rest.status.ledger.w;
        restingPositionWork=rest.status.projectionBounds.x;
        restingSupportValid=rest.status.control.w==1&&rest.status.control.y==0&&restingImpulse>0&&
            restingKineticLoss==0&&restingPositionWork==0&&rest.status.contactKinetic.x==0;
    }
    bool exact=true,allValid=true;double difference=0,energyIncrease=0,maximumElastic=0,maximumShape=0,minimumHeight=height(motion[0]),minimumClearance=1e30;
    double initialEnergy=mechanicalEnergy(mesh,motion[0]),freeFallError=0,maximumMomentumError=0;
    double maximumEnergyAccountingError=0,maximumContactComponentsError=0;
    double mass=0;for(auto v:mesh.velocityAndMass)mass+=v.w;
    V initialCenter=center(motion[0]);double finalHeight=height(motion.back());
    for(size_t f=0;f<motion.size();++f){
        exact=exact&&std::memcmp(motion[f].positions.data(),replay[f].positions.data(),mesh.positions.size()*sizeof(mr_float4))==0&&
            std::memcmp(motion[f].velocities.data(),replay[f].velocities.data(),mesh.positions.size()*sizeof(mr_float4))==0&&
            std::memcmp(&motion[f].status,&replay[f].status,sizeof(NumiDeformableMeshStatus))==0;
        for(const auto* state:{&motion[f],&fine[f]}){
            maximumEnergyAccountingError=std::max(maximumEnergyAccountingError,std::abs(
                mechanicalEnergy(mesh,*state)-initialEnergy-state->status.integration.x-state->status.contactWork.z));
            maximumContactComponentsError=std::max(maximumContactComponentsError,std::abs(double(
                state->status.contactWork.x)+state->status.contactWork.y+state->status.contactKinetic.x-state->status.contactWork.z));
            allValid=allValid&&state->status.control.y==0&&state->status.control.z==0;
            energyIncrease=std::max(energyIncrease,mechanicalEnergy(mesh,*state)-initialEnergy);
            maximumElastic=std::max(maximumElastic,elasticEnergy(mesh,*state));
            double momentum=0;for(auto v:state->velocities)momentum+=v.w*v.z;
            maximumMomentumError=std::max(maximumMomentumError,std::abs(momentum-(-mass*9.81f*f*.005+state->status.ledger.x)));
            if(state->status.ledger.x==0){double dt=state==&motion[f]?timestep:timestep/2,t=f*.005;
                double offset=integrator==NUMI_DEFORMABLE_MESH_INTEGRATOR_EULER?dt:0;
                freeFallError=std::max(freeFallError,std::abs(center(*state)[2]-(initialCenter[2]-.5*9.81f*t*(t+offset))));}
        }
        minimumHeight=std::min(minimumHeight,height(motion[f]));
        for(size_t n=0;n<mesh.positions.size();++n){
            difference=std::max(difference,norm(sub(xyz(motion[f].positions[n]),xyz(fine[f].positions[n]))));
            minimumClearance=std::min(minimumClearance,double(motion[f].positions[n].z));
            maximumShape=std::max(maximumShape,norm(sub(sub(xyz(motion[f].positions[n]),center(motion[f])),sub(xyz(motion[0].positions[n]),initialCenter))));
        }
    }
    auto final=motion.back().status;auto finalFine=fine.back().status;
    bool accountingValid=probeValid&&restingSupportValid&&maximumEnergyAccountingError<1e-4&&maximumContactComponentsError<1e-5;
    const uint32_t steps=uint32_t(std::lround(.5/timestep));
    bool qualified=exact&&allValid&&rollbackExact&&accountingValid&&final.control.w==steps&&finalFine.control.w==2*steps&&final.ledger.x>0&&
        final.ledger.y>0&&minimumClearance>=0&&maximumShape>.001&&height(motion[0])-minimumHeight>.001&&
        finalHeight-minimumHeight>.001&&difference<.001&&energyIncrease<.001&&freeFallError<2e-5&&maximumMomentumError<2e-4;
    exportFrames(prefix,mesh,motion,integrator);
    if(!prefix.empty())exportFrames(prefix+"-half",mesh,fine,integrator);
    std::cout<<std::setprecision(12)<<"device="<<device.name.UTF8String<<" backend=Apple_Metal\n"
        <<"shared_nodes="<<mesh.positions.size()<<" tetrahedra="<<mesh.elements.size()<<" boundary_triangles="<<mesh.faces.size()<<" mesh_refinement="<<refinement<<" density_kg_m3=1000 mass_kg="<<mass<<'\n'
        <<"refinement_diagonal="<<(shortestDiagonal?"shortest":"legacy")<<'\n'
        <<"simulated_seconds=.5 steps="<<steps<<" refined_steps="<<2*steps<<" timestep_s="<<timestep
        <<" captured_frames="<<motion.size()<<" replay_exact="<<exact<<" all_candidates_valid="<<allValid<<'\n'
        <<"maximum_nodal_refinement_difference_m="<<difference<<" maximum_relative_shape_change_m="<<maximumShape<<'\n'
        <<"initial_height_m="<<height(motion[0])<<" minimum_height_m="<<minimumHeight<<" final_height_m="<<finalHeight<<'\n'
        <<"minimum_clearance_m="<<minimumClearance<<" minimum_all_step_volume_ratio="<<final.ledger.y
        <<" maximum_internal_net_force_N="<<final.ledger.z<<'\n'
        <<"bench_normal_impulse_Ns="<<final.ledger.x<<" removed_normal_kinetic_energy_J="<<final.ledger.w
        <<" maximum_momentum_balance_error_Ns="<<maximumMomentumError<<'\n'
        <<"maximum_mechanical_energy_increase_J="<<energyIncrease<<" maximum_elastic_energy_J="<<maximumElastic
        <<" maximum_free_fall_COM_error_m="<<freeFallError<<'\n'
        <<"rejected_cases="<<rejectedCases<<" rejected_mesh_state_and_ledger_exact="<<rollbackExact<<'\n'
        <<"mesh_ABI="<<NUMI_DEFORMABLE_MESH_ABI_VERSION<<" integrator="<<(integrator==0?"euler":"support_verlet")
        <<" projection_work_probe_pass="<<probeValid
        <<" projection_work_probe_maximum_error_J="<<probeError<<'\n'
        <<"resting_support_probe_pass="<<restingSupportValid<<" resting_support_impulse_Ns="<<restingImpulse
        <<" resting_support_kinetic_loss_J="<<restingKineticLoss<<" resting_support_position_work_J="<<restingPositionWork<<'\n'
        <<"contact_elastic_change_J="<<final.contactWork.x<<" contact_gravity_change_J="<<final.contactWork.y
        <<" contact_mechanical_change_J="<<final.contactWork.z<<" maximum_positive_contact_step_J="<<final.contactWork.w<<'\n'
        <<"free_integration_change_J="<<final.integration.x<<" absolute_free_integration_change_J="<<final.integration.y
        <<" maximum_absolute_free_step_J="<<final.integration.z<<" elastic_force_work_J="<<final.integration.w<<'\n'
        <<"half_step_free_integration_change_J="<<finalFine.integration.x
        <<" half_step_absolute_free_integration_change_J="<<finalFine.integration.y
        <<" half_step_contact_mechanical_change_J="<<finalFine.contactWork.z<<'\n'
        <<"absolute_position_projection_work_J="<<final.projectionBounds.x
        <<" half_step_absolute_position_projection_work_J="<<finalFine.projectionBounds.x
        <<" positive_contact_work_J="<<final.projectionBounds.w<<'\n'
        <<"contact_kinetic_change_J="<<final.contactKinetic.x<<" contact_force_response_change_J="<<final.contactKinetic.y
        <<" absolute_contact_force_response_change_J="<<final.contactKinetic.z<<'\n'
        <<"maximum_independent_energy_accounting_error_J="<<maximumEnergyAccountingError
        <<" maximum_contact_component_sum_error_J="<<maximumContactComponentsError
        <<" energy_accounting_valid="<<accountingValid<<'\n'
        <<"energy_defect_qualification=OPEN spatial_convergence=OPEN material_calibration=OPEN\n"
        <<"deformable_mesh_drop_qualified="<<qualified<<'\n';
    return qualified?0:1;
}
}
int main(int argc,const char* const* argv){@autoreleasepool{try{
    std::string prefix;uint32_t refinement=0,integrator=NUMI_DEFORMABLE_MESH_INTEGRATOR_EULER;
    bool topologyOnly=false,shortestDiagonal=false;float timestep=1e-4f;
    for(int argument=1;argument<argc;++argument){
        std::string option=argv[argument];
        if(option=="--trajectory"&&argument+1<argc)prefix=argv[++argument];
        else if(option=="--topology-probe")topologyOnly=true;
        else if(option=="--refinement-diagonal"&&argument+1<argc){
            const std::string value=argv[++argument];
            if(value!="legacy"&&value!="shortest")throw std::runtime_error("refinement diagonal must be legacy or shortest");
            shortestDiagonal=value=="shortest";
        }
        else if(option=="--integrator"&&argument+1<argc){
            std::string value=argv[++argument];
            if(value=="euler")integrator=NUMI_DEFORMABLE_MESH_INTEGRATOR_EULER;
            else if(value=="support-verlet")integrator=NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET;
            else throw std::runtime_error("integrator must be euler or support-verlet");
        }
        else if(option=="--timestep"&&argument+1<argc){
            std::string value=argv[++argument];size_t parsed=0;double requested=std::stod(value,&parsed);
            bool supported=false;
            for(uint32_t half=0;half<=5;++half){double dt=1e-4/std::pow(2.,half);
                if(parsed==value.size()&&std::isfinite(requested)&&std::abs(requested-dt)<1e-15){
                    timestep=float(dt);supported=true;break;}}
            if(!supported)throw std::runtime_error("timestep must be 100 microseconds divided by 2^k, k=0..5");
        }
        else if(option=="--mesh-refinement"&&argument+1<argc){
            std::string value=argv[++argument];
            if(value!="0"&&value!="1"&&value!="2"&&value!="3"&&value!="4")throw std::runtime_error("mesh refinement must be 0, 1, 2, 3, or 4");
            refinement=uint32_t(std::stoul(value));
        } else throw std::runtime_error("usage: numi-solver-deformable-mesh [--trajectory PREFIX] [--mesh-refinement 0|1|2|3|4] [--refinement-diagonal legacy|shortest] [--timestep DT] [--integrator euler|support-verlet] [--topology-probe]");
    }
    if(topologyOnly)return topologyProbe(shortestDiagonal);
    return run(prefix,refinement,timestep,integrator,shortestDiagonal);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}}

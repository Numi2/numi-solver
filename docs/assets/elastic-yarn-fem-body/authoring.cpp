#include "numi/deformable_mesh.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
Mesh makeMesh(const uint32_t refinement=0){
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
            for(auto child:std::array<std::array<uint32_t,4>,8>{{
                {a,ab,ac,ad},{b,ab,bc,bd},{c,ac,bc,cd},{d,ad,bd,cd},
                {ab,ac,ad,cd},{ab,ac,bc,cd},{ab,ad,bd,cd},{ab,bc,bd,cd}}}){
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
    for(auto face:mesh.faces)for(int edge=0;edge<3;++edge){
        uint32_t a=face[edge],b=face[(edge+1)%3];int forward=0,reverse=0;
        for(auto other:mesh.faces)for(int e=0;e<3;++e){
            forward+=other[e]==a&&other[(e+1)%3]==b;
            reverse+=other[e]==b&&other[(e+1)%3]==a;
        }
        if(forward!=1||reverse!=1)throw std::runtime_error("boundary is not an oriented manifold");
    }
    return mesh;
}

} // anonymous namespace
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("output directory required");
    auto mesh=makeMesh(3);std::string path=argv[1];
    const auto dump=[&](const std::string& name,const auto& values){
        std::ofstream output(path+"/"+name,std::ios::binary);
        if(!output)throw std::runtime_error("cannot write pack");
        output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));
        if(!output)throw std::runtime_error("pack write failed");
    };
    dump("rest-positions.f32x4.bin",mesh.positions);
    dump("rest-velocity-mass.f32x4.bin",mesh.velocityAndMass);
    dump("elements.abi1.bin",mesh.elements);
    dump("boundary-faces.u32x3.bin",mesh.faces);
    dump("incidence-offsets.u32.bin",mesh.offsets);
    dump("incidence.u32x4.bin",mesh.incidence);
    double mass=0,volume=0;for(auto v:mesh.velocityAndMass)mass+=v.w;
    for(auto e:mesh.elements)volume+=e.inverseRestRows[0].w;
    std::cout<<std::setprecision(17)<<"source_mesh_nodes="<<mesh.positions.size()
        <<" elements="<<mesh.elements.size()<<" boundary_faces="<<mesh.faces.size()
        <<" incidence_entries="<<mesh.incidence.size()<<" total_mass_kg="<<mass
        <<" represented_total_volume_m3="<<volume<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

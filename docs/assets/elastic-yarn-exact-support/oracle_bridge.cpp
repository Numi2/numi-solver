#define main retainedPrototypeMain
#include "frozen/prototype.cpp"
#undef main
namespace {
ccd::Motion motion(const float* input){ccd::Motion m;for(unsigned i=0;i<5;++i){m.start[i]={input[3*i],input[3*i+1],input[3*i+2]};m.end[i]={input[15+3*i],input[16+3*i],input[17+3*i]};}m.radius=input[30];return m;}
}
extern "C" double independent_gap(const float* input,double time){return ccd::oracle(motion(input),time);}
extern "C" double interval_motion_upper(const float* input){auto m=motion(input);double travel[2]={};for(unsigned i=0;i<5;++i){ccd::IV d;for(unsigned k=0;k<3;++k)d[k]=ccd::sub(ccd::exact(ccd::component(m.end[i],k)),ccd::exact(ccd::component(m.start[i],k)));travel[i>=2]=std::max(travel[i>=2],ccd::norm(d).hi);}return ccd::up(travel[0]+travel[1]);}
extern "C" int admit_face(const float* input,unsigned budget,unsigned* used){auto m=motion(input);*used=0;
 const auto admit=[&](auto&& self,double l,double h)->int {
  if(*used>=budget)return 2;++*used;if(ccd::faceRange(m,l,h))return 0;
  double mid=(l+h)*.5;if(!ccd::witness(m,l).valid||!ccd::witness(m,mid).valid||!ccd::witness(m,h).valid)return 1;
  if(!(mid>l&&mid<h))return 2;int left=self(self,l,mid);return left?left:self(self,mid,h);
 };return admit(admit,0,1);
}
extern "C" double legacy_plane_lower(const float* input,double l,double h,const double* n){return ccd::planeLower(motion(input),l,h,{n[0],n[1],n[2]});}
extern "C" double safe_plane_lower(const float* input,double l,double h,const double* n){
 if(!std::isfinite(l)||!std::isfinite(h)||l<0||h>1||l>h||!(input[30]>0)||!std::isfinite(input[30]))return -INFINITY;
 for(unsigned i=0;i<30;++i)if(!std::isfinite(input[i]))return -INFINITY;
 for(unsigned i=0;i<3;++i)if(!std::isfinite(n[i]))return -INFINITY;
 if(n[0]==0&&n[1]==0&&n[2]==0)return -INFINITY;
 return ccd::planeLower(motion(input),l,h,{n[0],n[1],n[2]});
}

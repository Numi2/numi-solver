#define main numiClothProgramMain
#include "../tools/cloth_bag.cpp"
#undef main
#include <sstream>
int main(int argc,char**argv) {
  if(argc!=2) return 2;
  gFiniteBench=true;
  ClothModel cloth=makeCloth(Scenario::pickup);
  auto balls=makeBalls(Scenario::pickup);
  std::ifstream stream(argv[1]);std::string line;size_t node=0,fruitCount=0;
  while(std::getline(stream,line)) {
    std::istringstream row(line);std::string type;row>>type;
    if(type=="v") {auto &p=cloth.particles.at(node++);row>>p.position.x>>p.position.y>>p.position.z;p.previous=p.position;}
    if(line.rfind("# ball ",0)==0) {row.str(line.substr(7));row.clear();size_t i;std::string label;row>>i>>label;auto &b=balls.at(i);row>>b.position.x>>b.position.y>>b.position.z>>label>>b.radius;b.previous=b.position;++fruitCount;}
  }
  if(node!=cloth.particles.size()||fruitCount!=12) return 2;
  Metrics metrics;const double dt=1./(120*96);std::vector<BallYarnContactImpulse> contacts(12*cloth.yarnSegments.size());
  std::array<BallPairContactImpulse,kBallPairCount> pairs{};std::array<double,12> ground{};
  auto report=[&](unsigned pass,const char*phase) {
    double peak=0;size_t ball=0,segment=0;
    for(size_t b=0;b<12;++b)for(size_t s=0;s<cloth.yarnSegments.size();++s){auto edge=cloth.yarnSegments[s];auto closest=closestPointsOnSegments(balls[b].position,balls[b].position,cloth.particles[edge.first].position,cloth.particles[edge.second].position);double gap=balls[b].radius+kClothRadius-length(closest.secondPoint-balls[b].position);if(gap>peak){peak=gap;ball=b;segment=s;}}
    auto edge=cloth.yarnSegments[segment];
    std::cout<<std::setprecision(17)<<pass<<','<<phase<<','<<peak<<','<<ball<<','<<edge.first<<','<<edge.second<<','<<measurePrimitiveSelfPenetration(cloth)<<','<<measureLocalNodePenetration(cloth)<<','<<measureStrainLimitViolation(cloth)<<','<<measureGroundPenetration(cloth.particles,balls)<<'\n';
  };
  std::cout<<"pass,phase,fruit_yarn_m,fruit,node_a,node_b,primitive_m,local_m,strain_m,static_m\n";report(0,"input");
  for(unsigned pass=1;pass<=160;++pass) {
    solvePrimitiveSelfCollision(cloth,false,metrics,dt,nullptr);report(pass,"primitive");
    for(unsigned s=0;s<3;++s)solveStrainLimits(cloth.particles,cloth.distances);report(pass,"strain");
    for(unsigned final=0;final<2;++final) {
      solveLocalNodeContacts(cloth,true,metrics.localNodeContacts);report(pass,"local");
      size_t pair=0;for(size_t a=0;a<12;++a)for(size_t b=a+1;b<12;++b)solveBallPair(balls[a],balls[b],dt,pairs[pair++]);report(pass,"fruit_pair");
      for(size_t b=0;b<12;++b) {
        for(size_t s=0;s<cloth.yarnSegments.size();++s)solveBallYarn(cloth.particles,cloth.yarnSegments[s],balls[b],dt,true,contacts[b*cloth.yarnSegments.size()+s],metrics.ballYarnContacts);
        std::string phase="fruit_yarn_"+std::to_string(b);report(pass,phase.c_str());
      }
      double c=0,b=0;solveGround(cloth.particles,balls,dt,ground,c,b);report(pass,"ground");
    }
  }
}

// Independent convex signed-distance minimization in FP64. The signed SDF of
// a convex box is convex along an axial line; ternary interval search therefore
// finds its global minimum, including flat face intervals and deep crossings.
// The floor minimum is attained at an endpoint. This deliberately does not
// reuse the shared FP32 feature/interval minimizer being qualified.
double independentStaticYarnGap(DVec3 a, DVec3 b, double radius) {
    const auto boxGap = [&](double u) {
        const auto p=a+(b-a)*u;
        return numi::bench::sample({}, {p.x,p.y,p.z}, radius).gap;
    };
    double lo=0,hi=1;
    for(unsigned i=0;i<100;++i) {
        const double left=lo+(hi-lo)/3,right=hi-(hi-lo)/3;
        if(boxGap(left)<boxGap(right)) hi=right; else lo=left;
    }
    return std::min({boxGap(0),boxGap(1),boxGap((lo+hi)*.5),std::min(a.z,b.z)+.75-radius});
}
// Dispatch the exact production velocity kernel against the retained energy
// counterexamples. No advance or constraint solve can hide a kinetic increase.
bool runStaticYarnCoupledVelocityRegression(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                           const Pipelines& pipelines) {
    NumiClothBagGPUConfig config{};
    config.control=u4(NUMI_CLOTH_BAG_GPU_ABI_VERSION,3,2,0);
    config.constraintCounts=u4(0,0,2,0);
    config.gravityAndTimestep=f4(0,0,0,.001f);
    config.clothMaterial=f4(.004f,0,0,0);
    std::array<NumiClothBagGPUParticle,3> initial{};
    initial[0].positionAndInverseMass=f4(-1,0,.312493205038f,0);
    initial[1].positionAndInverseMass=f4(1,0,-.04f,1);
    initial[2].positionAndInverseMass=f4(-1,0,-.392493205038f,0);
    initial[1].velocity=f4(-1,0,0,0);
    std::array<NumiClothBagGPUDistance,2> edges{};
    edges[0].particlesAndColor=u4(0,1,0,0);
    edges[1].particlesAndColor=u4(1,2,1,0);
    struct Result {
        std::array<NumiClothBagGPUParticle,3> particles{};
        std::array<NumiClothBagGPUStaticYarnContact,2> contacts{};
        NumiClothBagGPUFrictionStatus friction{};
        std::uint32_t failure{};
    };
    const auto execute=[&](unsigned sweeps) {
        const auto bytes=[device](const void* data,NSUInteger size) {
            id<MTLBuffer> b=[device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
            if(b==nil) throw std::runtime_error("coupled static yarn regression buffer failed");
            return b;
        };
        Result result;
        auto configBuffer=bytes(&config,sizeof(config));
        auto particleBuffer=bytes(initial.data(),sizeof(initial));
        auto edgeBuffer=bytes(edges.data(),sizeof(edges));
        auto contactBuffer=bytes(result.contacts.data(),sizeof(result.contacts));
        auto failureBuffer=bytes(&result.failure,sizeof(result.failure));
        auto frictionBuffer=bytes(&result.friction,sizeof(result.friction));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(command==nil || encoder==nil) throw std::runtime_error("coupled static yarn regression encoder failed");
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:edgeBuffer offset:0 atIndex:2];
        [encoder setBuffer:contactBuffer offset:0 atIndex:3];
        [encoder setBuffer:failureBuffer offset:0 atIndex:5];
        [encoder setBuffer:frictionBuffer offset:0 atIndex:6];
        const auto colorSweep=[&](unsigned mode) {
            for(unsigned edge=0;edge<2;++edge) {
                NumiClothBagGPUBatch batch{u4(edge,1,edge,mode)};
                [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
                dispatch(encoder,pipelines.staticYarn,1);
            }
        };
        for(unsigned i=0;i<sweeps;++i) {
            colorSweep(2);
            if(config.clothMaterial.z>0) colorSweep(3);
        }
        if(config.clothMaterial.z>0) colorSweep(2);
        colorSweep(5);
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)
            throw std::runtime_error("coupled static yarn regression command failed: "+errorText(command.error));
        std::memcpy(result.particles.data(),particleBuffer.contents,sizeof(result.particles));
        std::memcpy(result.contacts.data(),contactBuffer.contents,sizeof(result.contacts));
        std::memcpy(&result.friction,frictionBuffer.contents,sizeof(result.friction));
        std::memcpy(&result.failure,failureBuffer.contents,sizeof(result.failure));
        return result;
    };
    const auto first=execute(kStaticYarnVelocitySweeps),replay=execute(kStaticYarnVelocitySweeps);
    const auto insufficient=execute(3);
    const bool exact=first.failure==replay.failure &&
        std::memcmp(first.particles.data(),replay.particles.data(),sizeof(first.particles))==0 &&
        std::memcmp(first.contacts.data(),replay.contacts.data(),sizeof(first.contacts))==0 &&
        std::memcmp(&first.friction,&replay.friction,sizeof(first.friction))==0;
    const auto residual=[&](const Result& result) {
        double worst=0;
        for(const auto& edge:edges) {
            const auto a=d3(result.particles[edge.particlesAndColor.x].positionAndInverseMass);
            const auto b=d3(result.particles[edge.particlesAndColor.y].positionAndInverseMass);
            const auto gap=[&](double u){const auto p=a+(b-a)*u;return numi::bench::sample({}, {p.x,p.y,p.z},.004f).gap;};
            double lo=0,hi=1;
            for(unsigned i=0;i<100;++i) {const double l=lo+(hi-lo)/3,r=hi-(hi-lo)/3;if(gap(l)<gap(r))hi=r;else lo=l;}
            const double u=(lo+hi)*.5;
            const auto point=a+(b-a)*u;
            const auto sample=numi::bench::sample({}, {point.x,point.y,point.z},.004f);
            const auto v=d3(result.particles[edge.particlesAndColor.x].velocity)*(1-u)+
                         d3(result.particles[edge.particlesAndColor.y].velocity)*u;
            worst=std::min(worst,v.x*sample.normal.x+v.y*sample.normal.y+v.z*sample.normal.z);
        }
        return worst;
    };
    const double worst=residual(first),negative=residual(insufficient);
    const auto velocity=d3(first.particles[1].velocity);
    bool positionsUnchanged=true;
    for(unsigned i=0;i<3;++i) positionsUnchanged &= std::memcmp(&first.particles[i].positionAndInverseMass,
        &initial[i].positionAndInverseMass,sizeof(mr_float4))==0;
    const bool valid=first.failure==0 && replay.failure==0 && exact && positionsUnchanged &&
        worst>=-5e-5 && dot(velocity,velocity)<=1 &&
        (insufficient.failure & NUMI_CLOTH_BAG_GPU_FAILURE_STATIC_YARN)!=0 && negative<-.1;
    std::cout << "static_yarn_coupled_velocity_case=two_pinned_yarns color_sweeps=" << kStaticYarnVelocitySweeps
              << " minimum_independent_normal_velocity_m_s=" << worst
              << " insufficient_three_sweeps_residual_m_s=" << negative
              << " insufficient_failure_flags=" << insufficient.failure
              << " failure_flags=" << first.failure << " replay_exact=" << exact
              << " passed=" << valid << '\n';
    // A frictionless matched control must keep all transverse speed. The
    // two rounded-edge normals may still dissipate normal kinetic energy.
    initial[1].velocity.y=1;
    const auto frictionless=execute(kStaticYarnVelocitySweeps),frictionlessReplay=execute(kStaticYarnVelocitySweeps);
    const bool frictionlessExact=frictionless.failure==frictionlessReplay.failure &&
        std::memcmp(frictionless.particles.data(),frictionlessReplay.particles.data(),sizeof(frictionless.particles))==0 &&
        std::memcmp(frictionless.contacts.data(),frictionlessReplay.contacts.data(),sizeof(frictionless.contacts))==0 &&
        std::memcmp(&frictionless.friction,&frictionlessReplay.friction,sizeof(frictionless.friction))==0;
    const auto frictionlessVelocity=d3(frictionless.particles[1].velocity);
    const bool frictionlessValid=frictionless.failure==0 && frictionlessExact &&
        residual(frictionless)>=-5e-5 && std::abs(frictionlessVelocity.y-1)<1e-6 &&
        dot(frictionlessVelocity,frictionlessVelocity)<=2 &&
        frictionless.contacts[0].witness.w==0 && frictionless.contacts[1].witness.w==0;
    std::cout << "static_yarn_coupled_friction_control=mu_zero residual_slip_y_m_s=" << frictionlessVelocity.y
              << " failure_flags=" << frictionless.failure << " replay_exact=" << frictionlessExact
              << " passed=" << frictionlessValid << '\n';

    struct Cone { double capacity,used,slack,gap; };
    const auto independentCone=[&](const Result& result,unsigned index,double mu) {
        const auto& edge=edges[index];
        const DVec3 a=d3(result.particles[edge.particlesAndColor.x].positionAndInverseMass);
        const DVec3 b=d3(result.particles[edge.particlesAndColor.y].positionAndInverseMass);
        const auto sampleAt=[&](double u) {
            const DVec3 p=a+(b-a)*u;
            return numi::bench::sample({}, {p.x,p.y,p.z},.004f);
        };
        double lo=0,hi=1;
        for(unsigned iteration=0;iteration<100;++iteration) {
            const double l=lo+(hi-lo)/3,r=hi-(hi-lo)/3;
            if(sampleAt(l).gap<sampleAt(r).gap)hi=r;else lo=l;
        }
        const auto witness=sampleAt((lo+hi)*.5);
        const DVec3 normal{witness.normal.x,witness.normal.y,witness.normal.z};
        const auto& contact=result.contacts[index];
        const DVec3 reaction=d3(contact.normalImpulse)+d3(contact.positionReaction)/.001;
        const double capacity=mu*std::max(0.,dot(reaction,normal));
        const double used=contact.witness.w;
        return Cone{capacity,used,capacity-used,witness.gap};
    };

    config.clothMaterial.z=.5f;
    const auto sliding=execute(kStaticYarnVelocitySweeps),slidingReplay=execute(kStaticYarnVelocitySweeps);
    const bool slidingExact=sliding.failure==slidingReplay.failure &&
        std::memcmp(sliding.particles.data(),slidingReplay.particles.data(),sizeof(sliding.particles))==0 &&
        std::memcmp(sliding.contacts.data(),slidingReplay.contacts.data(),sizeof(sliding.contacts))==0 &&
        std::memcmp(&sliding.friction,&slidingReplay.friction,sizeof(sliding.friction))==0;
    const auto slidingVelocity=d3(sliding.particles[1].velocity);
    const auto slidingConeA=independentCone(sliding,0,.5),slidingConeB=independentCone(sliding,1,.5);
    const bool slidingValid=sliding.failure==0 && slidingReplay.failure==0 && slidingExact &&
        residual(sliding)>=-5e-5 && slidingVelocity.y>1e-3 && slidingVelocity.y<1 &&
        dot(slidingVelocity,slidingVelocity)<=2 &&
        slidingConeA.capacity>0 && slidingConeB.capacity>0 &&
        std::abs(slidingConeA.slack)<1e-5 && std::abs(slidingConeB.slack)<1e-5 &&
        slidingConeA.gap>=-2e-6 && slidingConeB.gap>=-2e-6;
    std::cout << "static_yarn_coupled_friction_case=two_pinned_yarns mu=0.5"
              << " residual_slip_y_m_s=" << slidingVelocity.y
              << " minimum_independent_normal_velocity_m_s=" << residual(sliding)
              << " independent_cone_a_capacity_Ns=" << slidingConeA.capacity
              << " independent_cone_a_used_Ns=" << slidingConeA.used
              << " independent_cone_b_capacity_Ns=" << slidingConeB.capacity
              << " independent_cone_b_used_Ns=" << slidingConeB.used
              << " kinetic_change_J=" << .5*(dot(slidingVelocity,slidingVelocity)-2)
              << " failure_flags=" << sliding.failure << " replay_exact=" << slidingExact
              << " passed=" << slidingValid << '\n';

    config.clothMaterial.z=1.2f;
    const auto sticking=execute(kStaticYarnVelocitySweeps),stickingReplay=execute(kStaticYarnVelocitySweeps);
    const bool stickingExact=sticking.failure==stickingReplay.failure &&
        std::memcmp(sticking.particles.data(),stickingReplay.particles.data(),sizeof(sticking.particles))==0 &&
        std::memcmp(sticking.contacts.data(),stickingReplay.contacts.data(),sizeof(sticking.contacts))==0 &&
        std::memcmp(&sticking.friction,&stickingReplay.friction,sizeof(sticking.friction))==0;
    const auto stickingVelocity=d3(sticking.particles[1].velocity);
    const auto stickingConeA=independentCone(sticking,0,1.2),stickingConeB=independentCone(sticking,1,1.2);
    const bool stickingValid=sticking.failure==0 && stickingReplay.failure==0 && stickingExact &&
        residual(sticking)>=-5e-5 && std::abs(stickingVelocity.y)<1e-4 &&
        dot(stickingVelocity,stickingVelocity)<=2 &&
        stickingConeA.used>=0 && stickingConeB.used>=0 &&
        stickingConeA.slack>1e-3 && stickingConeB.slack>1e-3 &&
        stickingConeA.gap>=-2e-6 && stickingConeB.gap>=-2e-6;
    std::cout << "static_yarn_coupled_friction_case=two_pinned_yarns mu=1.2"
              << " residual_slip_y_m_s=" << stickingVelocity.y
              << " minimum_independent_normal_velocity_m_s=" << residual(sticking)
              << " independent_cone_a_capacity_Ns=" << stickingConeA.capacity
              << " independent_cone_a_used_Ns=" << stickingConeA.used
              << " independent_cone_b_capacity_Ns=" << stickingConeB.capacity
              << " independent_cone_b_used_Ns=" << stickingConeB.used
              << " kinetic_change_J=" << .5*(dot(stickingVelocity,stickingVelocity)-2)
              << " failure_flags=" << sticking.failure << " replay_exact=" << stickingExact
              << " passed=" << stickingValid << '\n';
    return valid && frictionlessValid && slidingValid && stickingValid;
}

bool runStaticYarnEnergyRegressions(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                   const Pipelines& pipelines) {
    struct Result {
        std::array<NumiClothBagGPUParticle,2> particles{};
        NumiClothBagGPUStaticYarnContact contact{};
        NumiClothBagGPUFrictionStatus friction{};
        std::uint32_t failure{};
    };
    bool passed = true;
    for (const unsigned mode : {2u,3u}) {
        NumiClothBagGPUConfig config{};
        config.control = u4(NUMI_CLOTH_BAG_GPU_ABI_VERSION,2,1,0);
        config.constraintCounts = u4(0,0,2,0);
        config.gravityAndTimestep = f4(0,0,0,.001f);
        config.clothMaterial = f4(.004f,0,.45f,0);
        Result initial;
        const float radius=.004f, root=std::sqrt(.5f);
        const float impactX=.75f+radius*root, impactZ=-.08f-radius*root;
        const auto body = [](float x,float z,float vz) {
            NumiClothBagGPUParticle p{};
            p.positionAndInverseMass=f4(x,0,z,20000);
            p.previousAndMass=f4(x,0,z,1.f/20000);
            p.velocity=f4(0,0,vz,0);
            return p;
        };
        const float floorZ=-.75f+radius+2e-7f;
        initial.particles[0]=body(mode==2 ? impactX+(floorZ-impactZ) : .75f+radius,floorZ,100);
        initial.particles[1]=body(mode==2 ? 1.f : .75f+radius,
                                  mode==2 ? impactZ+(1-impactX) : .1f,mode==2 ? 0.f : -1.f);
        if (mode==3) initial.contact.normalImpulse=f4(.01f,0,0,.01f);
        NumiClothBagGPUDistance edge{};
        edge.particlesAndColor=u4(0,1,0,0);
        NumiClothBagGPUBatch batch{u4(0,1,0,mode)};
        const auto execute = [&]() {
            const auto bytes = [device](const void* data,NSUInteger size) {
                id<MTLBuffer> b=[device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
                if(b==nil) throw std::runtime_error("static yarn regression buffer allocation failed");
                return b;
            };
            auto configBuffer=bytes(&config,sizeof(config));
            auto particleBuffer=bytes(initial.particles.data(),sizeof(initial.particles));
            auto edgeBuffer=bytes(&edge,sizeof(edge));
            auto contactBuffer=bytes(&initial.contact,sizeof(initial.contact));
            auto failureBuffer=bytes(&initial.failure,sizeof(initial.failure));
            auto frictionBuffer=bytes(&initial.friction,sizeof(initial.friction));
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(command==nil || encoder==nil) throw std::runtime_error("static yarn regression encoder failed");
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:edgeBuffer offset:0 atIndex:2];
            [encoder setBuffer:contactBuffer offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBuffer:failureBuffer offset:0 atIndex:5];
            [encoder setBuffer:frictionBuffer offset:0 atIndex:6];
            dispatch(encoder,pipelines.staticYarn,1);
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted)
                throw std::runtime_error("static yarn regression command failed: "+errorText(command.error));
            Result out;
            std::memcpy(out.particles.data(),particleBuffer.contents,sizeof(out.particles));
            std::memcpy(&out.contact,contactBuffer.contents,sizeof(out.contact));
            std::memcpy(&out.friction,frictionBuffer.contents,sizeof(out.friction));
            std::memcpy(&out.failure,failureBuffer.contents,sizeof(out.failure));
            return out;
        };
        const auto first=execute(), replay=execute();
        const bool exact=first.failure==replay.failure &&
            std::memcmp(first.particles.data(),replay.particles.data(),sizeof(first.particles))==0 &&
            std::memcmp(&first.contact,&replay.contact,sizeof(first.contact))==0 &&
            std::memcmp(&first.friction,&replay.friction,sizeof(first.friction))==0;
        long double delta=0;
        bool unchangedPositions=true,finite=true;
        for(unsigned i=0;i<2;++i) {
            unchangedPositions &= std::memcmp(&first.particles[i].positionAndInverseMass,
                &initial.particles[i].positionAndInverseMass,sizeof(mr_float4))==0;
            const auto before=d3(initial.particles[i].velocity),after=d3(first.particles[i].velocity);
            for(unsigned axis=0;axis<3;++axis) {
                const double a=axis==0?before.x:axis==1?before.y:before.z;
                const double b=axis==0?after.x:axis==1?after.y:after.z;
                delta+=.5L*initial.particles[i].previousAndMass.w*(static_cast<long double>(b)-a)*(static_cast<long double>(b)+a);
                finite &= std::isfinite(b);
            }
        }
        const bool valid=first.failure==0 && replay.failure==0 && exact && finite &&
            unchangedPositions && delta<=0 && first.particles[0].velocity.z>0 &&
            first.particles[0].velocity.w==0 &&
            (mode==2 ? first.contact.normalImpulse.w>0 :
             first.contact.witness.w>0 && first.contact.witness.w<=.0045f);
        passed &= valid;
        std::cout << "static_yarn_energy_case=" << (mode==2?"normal_slack":"friction_slack")
                  << " kinetic_change_J=" << static_cast<double>(delta)
                  << " outward_endpoint_velocity_m_s=" << first.particles[0].velocity.z
                  << " outward_endpoint_support_Ns=" << first.particles[0].velocity.w
                  << " normal_impulse_Ns=" << first.contact.normalImpulse.w
                  << " friction_impulse_Ns=" << first.contact.witness.w
                  << " failure_flags=" << first.failure << " replay_exact=" << exact
                  << " passed=" << valid << '\n';
    }
    return passed;
}

// Focused production-kernel qualification; no rendered geometry is substituted.
int runStaticYarnProbe(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       const Pipelines& pipelines) {
    struct Case {
        const char* name;
        DVec3 a, b, va, vb, gravity;
        float massA, massB, timestep, friction;
        unsigned steps;
        bool requireImpact, requireEndpointOnlyMiss, requireFreeDeparture;
        bool requireUnresolvedReject = false;
    };
    const std::array<Case, 9> cases{{
        {"bridge_top_equal", {-.9,0,.024},{.9,0,.024},{0,0,-4},{0,0,-4},{},1,1,.01f,0,1,true,true,false},
        {"bridge_top_unequal", {-.9,0,.024},{.9,0,.024},{0,0,-4},{0,0,-4},{},1,3,.01f,0,1,true,true,false},
        {"bridge_side", {.774,0,-.2},{.774,0,.12},{-4,0,0},{-4,0,0},{},1,2,.01f,0,1,true,true,false},
        {"through_entire_bench", {1,0,-.2},{1,0,.12},{-200,0,0},{-200,0,0},{},1,1,.01f,0,1,true,true,false},
        {"rotating_bridge", {-.9,0,.024},{.9,0,.024},{0,0,-3},{0,0,-5},{},2,1,.01f,0,1,true,true,false},
        {"friction_free_bridge", {-.9,0,.024},{.9,0,.024},{0,2,-4},{0,2,-4},{},1,2,.01f,0,1,true,true,false},
        {"unsupported_departure", {-.9,0,.004},{.9,0,.004},{0,2,2},{0,2,2},{},1,2,.01f,.5f,1,false,false,true},
        {"sustained_bridge", {-.9,0,.004},{.9,0,.004},{0,1,0},{0,1,0},{0,0,-9.81},1,2,.001f,.5f,64,true,false,false},
        {"endpoint_clip_obscures_earlier_interior", {0,0,.1},{2,0,-.05},{0,0,-20},{0,0,6},{},1,1,.01f,0,1,false,false,false,true},
    }};
    bool passed = true;
    double worstGap = 0, maximumEnergyIncrease = 0;
    for (const auto& c : cases) {
        InitialState initial;
        initial.config.control = u4(NUMI_CLOTH_BAG_GPU_ABI_VERSION, 2, 1, 0);
        initial.config.constraintCounts = u4(0, 0, 2, 0);
        initial.config.gravityAndTimestep = f4(c.gravity.x,c.gravity.y,c.gravity.z,c.timestep);
        initial.config.clothMaterial = f4(.004f, 0, c.friction, 0);
        const auto body = [](DVec3 p, DVec3 v, float mass) {
            NumiClothBagGPUParticle b{};
            b.positionAndInverseMass = f4(p.x,p.y,p.z,1/mass);
            b.previousAndMass = f4(p.x,p.y,p.z,mass);
            b.velocity = f4(v.x,v.y,v.z,0);
            return b;
        };
        initial.particles = {body(c.a,c.va,c.massA),body(c.b,c.vb,c.massB)};
        NumiClothBagGPUDistance yarn{};
        yarn.particlesAndColor = u4(0,1,0,0);
        yarn.material = f4(length(c.b-c.a),0,0,1);
        initial.distances = {yarn}; initial.distanceBatches = {{u4(0,1,0,0)}};
        std::vector<NumiClothBagGPUConfig> configs(c.steps, initial.config);
        const auto first = runGPU(device,queue,pipelines,initial,0,0,configs);
        const auto replay = runGPU(device,queue,pipelines,initial,0,0,configs);
        const auto& a = first.particles[0]; const auto& b = first.particles[1];
        // Independent FP64 whole-segment signed-distance audit, using the
        // same face/edge minimizer as retained scene geometry.
        const auto pointA = d3(a.positionAndInverseMass), pointB = d3(b.positionAndInverseMass);
        const double gap = independentStaticYarnGap(pointA,pointB,.004f);
        const bool exact = bitwiseEqualPhysicalState(first,replay);
        const auto& contact = first.staticYarnContacts[0];
        if(c.requireUnresolvedReject) {
            // Endpoint A impacts at t=.48. At that time the endpoint-safe
            // piecewise path contains >9mm of whole-yarn penetration, while
            // its recorded final clipped chord is clear. This case must
            // reject until common-event stepping is implemented.
            const double pathGap=independentStaticYarnGap({0,0,.004},{2,0,-.05+6*.0048},.004);
            const double clippedFinalGap=independentStaticYarnGap({0,0,.004},{2,0,.01},.004);
            const auto expected=NUMI_CLOTH_BAG_GPU_FAILURE_STATIC_YARN | NUMI_CLOTH_BAG_GPU_FAILURE_STATIC_ADVANCE;
            const bool valid=exact && (first.failure&expected)==expected &&
                (replay.failure&expected)==expected && pathGap<-.009 && clippedFinalGap>=-1e-7;
            passed &= valid;
            std::cout << "static_yarn_unresolved_case=" << c.name
                      << " endpoint_clipped_piecewise_gap_m=" << pathGap
                      << " clipped_final_gap_m=" << clippedFinalGap
                      << " failure_flags=" << first.failure << " replay_exact=" << exact
                      << " explicit_rejection_passed=" << valid << '\n';
            if(gAdaptiveStaticContact) {
                NativeTransactionReport firstReport,replayReport;
                const auto refined=runGPUTransactional(device,queue,pipelines,initial,0,0,configs,firstReport);
                const auto refinedReplay=runGPUTransactional(device,queue,pipelines,initial,0,0,configs,replayReport);
                const double refinedGap=independentStaticYarnGap(d3(refined.particles[0].positionAndInverseMass),
                    d3(refined.particles[1].positionAndInverseMass),.004f);
                const auto va=d3(refined.particles[0].velocity),vb=d3(refined.particles[1].velocity);
                const double deltaK=.5*c.massA*(dot(va,va)-dot(c.va,c.va))+
                                    .5*c.massB*(dot(vb,vb)-dot(c.vb,c.vb));
                const bool refinedExact=bitwiseEqualPhysicalState(refined,refinedReplay)&&
                    hashNativeTransaction(firstReport)==hashNativeTransaction(replayReport);
                const bool refinedValid=firstReport.complete&&replayReport.complete&&
                    refined.failure==0&&refinedReplay.failure==0&&refinedExact&&
                    firstReport.committedSeconds==static_cast<double>(c.timestep)&&
                    firstReport.rejectedAttempts>0&&firstReport.maximumDepth>0&&
                    refinedGap>=-2e-6&&std::isfinite(deltaK)&&deltaK<=1e-4;
                passed &= refinedValid;
                std::cout<<"static_yarn_adaptive_case="<<c.name<<" complete="<<firstReport.complete
                         <<" committed_time_s="<<firstReport.committedSeconds<<" rejected_attempts="<<firstReport.rejectedAttempts
                         <<" accepted_leaves="<<firstReport.acceptedLeaves<<" maximum_depth="<<firstReport.maximumDepth
                         <<" independent_final_gap_m="<<refinedGap<<" kinetic_change_J="<<deltaK
                         <<" failure_flags="<<refined.failure<<" grid_and_state_replay_exact="<<refinedExact
                         <<" passed="<<refinedValid<<'\n';
            }
            continue;
        }
        bool valid = first.failure == 0 && replay.failure == 0 && exact &&
                     std::isfinite(gap) && gap >= -2e-6;
        if (c.requireImpact) valid &= contact.normalImpulse.w > 0;
        const double energyBefore = .5*c.massA*dot(c.va,c.va)+.5*c.massB*dot(c.vb,c.vb);
        const auto velocityA=d3(a.velocity),velocityB=d3(b.velocity);
        const double energyAfter = .5*c.massA*dot(velocityA,velocityA)+.5*c.massB*dot(velocityB,velocityB);
        if (length(c.gravity)==0) valid &= energyAfter <= energyBefore + 1e-4*std::max(1.,energyBefore);
        if (c.requireEndpointOnlyMiss) {
            const auto freeA=c.a+c.va*c.timestep,freeB=c.b+c.vb*c.timestep;
            const double gapA=std::min(numi::bench::sample({}, {freeA.x,freeA.y,freeA.z},.004f).gap,freeA.z+.75-.004f);
            const double gapB=std::min(numi::bench::sample({}, {freeB.x,freeB.y,freeB.z},.004f).gap,freeB.z+.75-.004f);
            valid &= gapA > .01 && gapB > .01;
        }
        if (c.friction==0 && std::string_view(c.name)=="friction_free_bridge")
            valid &= std::abs(a.velocity.y-2)<1e-5 && std::abs(b.velocity.y-2)<1e-5;
        if (c.requireFreeDeparture) valid &=
            length(velocityA-c.va)<1e-5 && length(velocityB-c.vb)<1e-5 &&
            contact.normalImpulse.w==0 && contact.witness.w==0 && contact.witness.z==0 &&
            a.velocity.w==0 && b.velocity.w==0;
        passed &= valid; worstGap=std::min(worstGap,double(gap));
        if (length(c.gravity)==0) maximumEnergyIncrease=std::max(maximumEnergyIncrease,energyAfter-energyBefore);
        std::cout << "static_yarn_case=" << c.name << " gap_m=" << gap
                  << " normal_impulse_Ns=" << contact.normalImpulse.w
                  << " friction_impulse_Ns=" << contact.witness.w
                  << " kinetic_change_J=" << energyAfter-energyBefore
                  << " failure_flags=" << first.failure << " replay_exact=" << exact
                  << " passed=" << valid << '\n';
    }
    passed &= runStaticYarnEnergyRegressions(device,queue,pipelines);
    passed &= runStaticYarnCoupledVelocityRegression(device,queue,pipelines);
    std::cout << "static_yarn_cases=" << cases.size() << " static_yarn_energy_cases=2 minimum_independent_gap_m=" << worstGap
              << " maximum_unforced_kinetic_increase_J=" << maximumEnergyIncrease
              << " static_yarn_probe_passed=" << passed << '\n';
    return passed ? 0 : 1;
}

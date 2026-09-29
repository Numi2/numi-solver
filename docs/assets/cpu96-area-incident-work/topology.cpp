#define main frozen_cloth_entry_for_topology_only
#include "frozen/cloth_bag.cpp"
#undef main

#include <queue>

int main() {
    const ClothModel cloth = makeCloth(Scenario::recorded);
    constexpr std::array<std::uint32_t, 3> failed{{1433u, 1444u, 1445u}};
    const auto owns = [&](const std::uint32_t x) {
        return std::find(failed.begin(), failed.end(), x) != failed.end();
    };
    const auto touches = [&](const Triangle& t) {
        return owns(t.first) || owns(t.second) || owns(t.third);
    };
    std::vector<std::uint32_t> incidentFaces;
    for (std::uint32_t i = 0; i < cloth.renderTriangles.size(); ++i)
        if (touches(cloth.renderTriangles[i])) incidentFaces.push_back(i);
    if (cloth.renderTriangles.size() <= 2813u) return 2;
    const Triangle selected = cloth.renderTriangles[2813u];
    if (!(selected.first == failed[0] && selected.second == failed[1] && selected.third == failed[2])) return 3;
    std::vector<std::uint32_t> incidentDistance, incidentBend, incidentKnot, incidentGrip, incidentSegment;
    for (std::uint32_t i=0;i<cloth.distances.size();++i) {
        const auto& d=cloth.distances[i];
        if (owns(d.first) || owns(d.second)) incidentDistance.push_back(i);
    }
    for (std::uint32_t i=0;i<cloth.bends.size();++i) {
        const auto& b=cloth.bends[i];
        if (owns(b.first) || owns(b.middle) || owns(b.third)) incidentBend.push_back(i);
    }
    for (std::uint32_t i=0;i<cloth.knots.size();++i) {
        const auto& k=cloth.knots[i];
        if (owns(k.warpFirst) || owns(k.warpSecond) || owns(k.weftFirst) || owns(k.weftSecond)) incidentKnot.push_back(i);
    }
    for (std::uint32_t i=0;i<cloth.grips.size();++i)
        if (owns(cloth.grips[i].particle)) incidentGrip.push_back(i);
    for (std::uint32_t i=0;i<cloth.yarnSegments.size();++i) {
        const auto& e=cloth.yarnSegments[i];
        if (owns(e.first) || owns(e.second)) incidentSegment.push_back(i);
    }
    std::size_t incidentLocalNodePairs=0;
    for (const auto& e:cloth.localNodePairs)
        incidentLocalNodePairs+=owns(e.first)||owns(e.second);
    std::vector<std::vector<std::uint32_t>> graph(cloth.particles.size());
    for (const auto& d : cloth.distances) {
        graph[d.first].push_back(d.second);
        graph[d.second].push_back(d.first);
    }
    std::vector<bool> seen(graph.size());
    std::queue<std::uint32_t> q;
    q.push(failed[0]);seen[failed[0]]=true;
    std::size_t reachable = 0;
    while (!q.empty()) {
        const auto u=q.front();q.pop();++reachable;
        for (auto v:graph[u]) if (!seen[v]) {seen[v]=true;q.push(v);}
    }
    std::cout << std::setprecision(17)
              << "{\"cloth_nodes\":" << cloth.particles.size()
              << ",\"render_faces\":" << cloth.renderTriangles.size()
              << ",\"failed_face_index\":2813,\"failed_face_owners\":[1433,1444,1445]"
              << ",\"failed_face_masses_kg\":["
              << cloth.particles[failed[0]].mass << ',' << cloth.particles[failed[1]].mass << ','
              << cloth.particles[failed[2]].mass << ']'
              << ",\"incident_faces\":[";
    for (std::size_t j=0;j<incidentFaces.size();++j) {
        if (j) std::cout << ',';
        std::cout << incidentFaces[j];
    }
    std::cout << "],\"incident_distance_constraints\":" << incidentDistance.size()
              << ",\"incident_bend_constraints\":" << incidentBend.size()
              << ",\"incident_knot_constraints\":" << incidentKnot.size()
              << ",\"incident_grip_constraints\":" << incidentGrip.size()
              << ",\"incident_yarn_segments\":" << incidentSegment.size()
              << ",\"incident_local_node_pairs\":" << incidentLocalNodePairs
              << ",\"distance_graph_reachable_from_failed_owner\":" << reachable
              << ",\"all_distances\":" << cloth.distances.size()
              << ",\"all_bends\":" << cloth.bends.size()
              << ",\"all_knots\":" << cloth.knots.size()
              << ",\"all_yarn_segments\":" << cloth.yarnSegments.size()
              << ",\"incident_distance_rows\":[";
    for (std::size_t j=0;j<incidentDistance.size();++j) {
        if (j) std::cout << ',';
        const auto i=incidentDistance[j];const auto& d=cloth.distances[i];
        std::cout << '[' << i << ',' << d.first << ',' << d.second << ',' << d.restLength << ',' << d.compliance << ']';
    }
    std::cout << "],\"incident_bend_rows\":[";
    for (std::size_t j=0;j<incidentBend.size();++j) {
        if (j) std::cout << ',';
        const auto i=incidentBend[j];const auto& b=cloth.bends[i];
        std::cout << '[' << i << ',' << b.first << ',' << b.middle << ',' << b.third << ','
                  << b.restChord << ',' << b.restArc << ',' << b.compliance << ']';
    }
    std::cout << "],\"incident_knot_rows\":[";
    for (std::size_t j=0;j<incidentKnot.size();++j) {
        if (j) std::cout << ',';
        const auto i=incidentKnot[j];const auto& k=cloth.knots[i];
        std::cout << '[' << i << ',' << k.warpFirst << ',' << k.warpSecond << ','
                  << k.weftFirst << ',' << k.weftSecond << ',' << k.restCosine << ','
                  << k.compliance << ']';
    }
    std::cout << "]}\n";
    return 0;
}

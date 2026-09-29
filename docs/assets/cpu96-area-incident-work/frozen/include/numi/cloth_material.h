#ifndef NUMI_CLOTH_MATERIAL_H
#define NUMI_CLOTH_MATERIAL_H

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace numi {

inline constexpr std::string_view kClothMaterialSchema =
    "numi.cloth.material.v1";

struct ClothMaterialParameters {
  double ordinaryNodeMassKg{0.000050};
  double hemNodeMassKg{0.000100};
  double yarnRadiusM{0.004};
  double axialBodyComplianceMPerN{1.0e-8};
  double axialCuffComplianceMPerN{1.0e-9};
  double knotCompliance{2.0e-6};
  double bendBodyComplianceMPerN{8.0e-2};
  double bendCuffComplianceMPerN{1.0e-8};
  double gripComplianceMPerN{2.0e-4};
  double clothGroundFriction{0.45};
  double clothSelfFriction{0.34};
  double fruitYarnFriction{0.36};
  double fruitGroundFriction{0.42};
  double fruitPairFriction{0.30};
  double fruitRollingResistance{0.015};
  double yarnCrossflowDrag{1.10};
  double yarnSkinFriction{0.010};
  double fruitDrag{0.47};
  double fruitRotationalDrag{0.010};
};

struct ClothMaterialArtifact {
  ClothMaterialParameters values{};
  bool loaded{};
  std::string parametersHash{"authored-defaults"};
  std::string observationsHash{"none"};
};

inline std::string trimClothMaterial(std::string value) {
  const auto isSpace = [](const unsigned char character) {
    return character == ' ' || character == '\t' || character == '\r' ||
           character == '\n';
  };
  while (!value.empty() && isSpace(value.front())) {
    value.erase(value.begin());
  }
  while (!value.empty() && isSpace(value.back())) {
    value.pop_back();
  }
  return value;
}

inline double parseClothMaterialNumber(const std::string &text,
                                       const std::string &name) {
  std::size_t consumed = 0u;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid cloth material value for " + name);
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("non-finite cloth material value for " + name);
  }
  return value;
}

inline void validateClothMaterial(const ClothMaterialParameters &value) {
  const auto bounded = [](const double candidate, const double minimum,
                          const double maximum, const std::string_view name) {
    if (!(candidate >= minimum && candidate <= maximum)) {
      throw std::runtime_error(
          "cloth material parameter outside contract bounds: " +
          std::string(name));
    }
  };
  bounded(value.ordinaryNodeMassKg, 0.000010, 0.000200,
          "ordinary_node_mass_kg");
  bounded(value.hemNodeMassKg, 0.000020, 0.000400, "hem_node_mass_kg");
  bounded(value.yarnRadiusM, 0.0005, 0.010, "yarn_radius_m");
  bounded(value.axialBodyComplianceMPerN, 1.0e-11, 1.0e-4,
          "axial_body_compliance_m_per_n");
  bounded(value.axialCuffComplianceMPerN, 1.0e-12, 1.0e-5,
          "axial_cuff_compliance_m_per_n");
  bounded(value.knotCompliance, 1.0e-10, 1.0e-2, "knot_compliance");
  bounded(value.bendBodyComplianceMPerN, 1.0e-5, 100.0,
          "bend_body_compliance_m_per_n");
  bounded(value.bendCuffComplianceMPerN, 1.0e-12, 1.0e-3,
          "bend_cuff_compliance_m_per_n");
  bounded(value.gripComplianceMPerN, 1.0e-8, 1.0, "grip_compliance_m_per_n");
  bounded(value.clothGroundFriction, 0.01, 2.0, "cloth_ground_friction");
  bounded(value.clothSelfFriction, 0.01, 2.0, "cloth_self_friction");
  bounded(value.fruitYarnFriction, 0.01, 2.0, "fruit_yarn_friction");
  bounded(value.fruitGroundFriction, 0.01, 2.0, "fruit_ground_friction");
  bounded(value.fruitPairFriction, 0.01, 2.0, "fruit_pair_friction");
  bounded(value.fruitRollingResistance, 0.0001, 0.20,
          "fruit_rolling_resistance");
  bounded(value.yarnCrossflowDrag, 0.10, 3.0, "yarn_crossflow_drag");
  bounded(value.yarnSkinFriction, 0.0001, 0.20, "yarn_skin_friction");
  bounded(value.fruitDrag, 0.05, 2.0, "fruit_drag");
  bounded(value.fruitRotationalDrag, 0.0001, 0.20, "fruit_rotational_drag");
}

inline ClothMaterialArtifact
loadClothMaterialArtifact(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open cloth material artifact: " + path);
  }
  ClothMaterialArtifact result;
  result.loaded = true;
  std::unordered_map<std::string, double ClothMaterialParameters::*> fields{
      {"ordinary_node_mass_kg", &ClothMaterialParameters::ordinaryNodeMassKg},
      {"hem_node_mass_kg", &ClothMaterialParameters::hemNodeMassKg},
      {"yarn_radius_m", &ClothMaterialParameters::yarnRadiusM},
      {"axial_body_compliance_m_per_n",
       &ClothMaterialParameters::axialBodyComplianceMPerN},
      {"axial_cuff_compliance_m_per_n",
       &ClothMaterialParameters::axialCuffComplianceMPerN},
      {"knot_compliance", &ClothMaterialParameters::knotCompliance},
      {"bend_body_compliance_m_per_n",
       &ClothMaterialParameters::bendBodyComplianceMPerN},
      {"bend_cuff_compliance_m_per_n",
       &ClothMaterialParameters::bendCuffComplianceMPerN},
      {"grip_compliance_m_per_n",
       &ClothMaterialParameters::gripComplianceMPerN},
      {"cloth_ground_friction", &ClothMaterialParameters::clothGroundFriction},
      {"cloth_self_friction", &ClothMaterialParameters::clothSelfFriction},
      {"fruit_yarn_friction", &ClothMaterialParameters::fruitYarnFriction},
      {"fruit_ground_friction", &ClothMaterialParameters::fruitGroundFriction},
      {"fruit_pair_friction", &ClothMaterialParameters::fruitPairFriction},
      {"fruit_rolling_resistance",
       &ClothMaterialParameters::fruitRollingResistance},
      {"yarn_crossflow_drag", &ClothMaterialParameters::yarnCrossflowDrag},
      {"yarn_skin_friction", &ClothMaterialParameters::yarnSkinFriction},
      {"fruit_drag", &ClothMaterialParameters::fruitDrag},
      {"fruit_rotational_drag", &ClothMaterialParameters::fruitRotationalDrag},
  };
  std::unordered_set<std::string> seen;
  std::string schema;
  std::string line;
  while (std::getline(stream, line)) {
    line = trimClothMaterial(std::move(line));
    if (line.empty() || line.starts_with('#')) {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("cloth material line must contain '='");
    }
    const std::string name = trimClothMaterial(line.substr(0u, separator));
    const std::string text = trimClothMaterial(line.substr(separator + 1u));
    if (name.empty() || text.empty() || !seen.insert(name).second) {
      throw std::runtime_error(
          "cloth material fields must be nonempty and unique");
    }
    if (name == "schema") {
      schema = text;
    } else if (name == "parameters_hash") {
      result.parametersHash = text;
    } else if (name == "observations_hash") {
      result.observationsHash = text;
    } else if (const auto field = fields.find(name); field != fields.end()) {
      result.values.*(field->second) = parseClothMaterialNumber(text, name);
    } else {
      throw std::runtime_error("unknown cloth material field: " + name);
    }
  }
  if (schema != kClothMaterialSchema) {
    throw std::runtime_error("cloth material schema mismatch");
  }
  if (!seen.contains("parameters_hash") ||
      !seen.contains("observations_hash")) {
    throw std::runtime_error("cloth material provenance hashes are required");
  }
  for (const auto &[name, unused] : fields) {
    static_cast<void>(unused);
    if (!seen.contains(name)) {
      throw std::runtime_error("cloth material artifact is missing: " + name);
    }
  }
  validateClothMaterial(result.values);
  return result;
}

} // namespace numi

#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kResultSchema = "numi.cloth.calibration.result.v1";
constexpr std::string_view kMaterialSchema = "numi.cloth.material.v1";

struct Parameter {
  std::string name;
  double base{};
  double minimum{};
  double maximum{};
};

enum class Split : std::uint8_t {
  calibration,
  heldout,
};

struct Observation {
  std::string trial;
  Split split{};
  std::string observable;
  double measured{};
  double sigma{};
  double baseline{};
  std::vector<double> sensitivities;
};

struct Metrics {
  double rmseZ{};
  double maximumAbsoluteZ{};
};

struct Options {
  std::string parametersPath;
  std::string observationsPath;
  std::string reportPath;
  std::string materialPath;
  double ridge{1.0e-10};
  double maximumCondition{1.0e4};
  double maximumCalibrationRMSEZ{2.0};
  double maximumHeldoutRMSEZ{2.0};
  double maximumHeldoutAbsoluteZ{4.0};
  bool requireHeldoutImprovement{true};
};

std::string trim(std::string value) {
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

std::vector<std::string> parseCSVRow(const std::string &line,
                                     const std::size_t lineNumber) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t index = 0u; index < line.size(); ++index) {
    const char character = line[index];
    if (quoted) {
      if (character == '"') {
        if (index + 1u < line.size() && line[index + 1u] == '"') {
          field.push_back('"');
          ++index;
        } else {
          quoted = false;
        }
      } else {
        field.push_back(character);
      }
    } else if (character == '"') {
      if (!field.empty()) {
        throw std::runtime_error("CSV quote must begin a field at line " +
                                 std::to_string(lineNumber));
      }
      quoted = true;
    } else if (character == ',') {
      fields.push_back(trim(field));
      field.clear();
    } else {
      field.push_back(character);
    }
  }
  if (quoted) {
    throw std::runtime_error("unterminated CSV quote at line " +
                             std::to_string(lineNumber));
  }
  fields.push_back(trim(field));
  return fields;
}

std::string readText(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  std::ostringstream result;
  result << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw std::runtime_error("cannot read " + path);
  }
  return result.str();
}

std::vector<std::vector<std::string>> readCSV(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<std::vector<std::string>> rows;
  std::string line;
  std::size_t lineNumber = 0u;
  while (std::getline(stream, line)) {
    ++lineNumber;
    const std::string stripped = trim(line);
    if (stripped.empty() || stripped.starts_with('#')) {
      continue;
    }
    rows.push_back(parseCSVRow(line, lineNumber));
  }
  if (rows.empty()) {
    throw std::runtime_error("CSV has no rows: " + path);
  }
  return rows;
}

double parseFinite(const std::string &text, const std::string_view field) {
  std::size_t consumed = 0u;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid number for " + std::string(field) + ": " +
                             text);
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("non-finite or trailing data for " +
                             std::string(field) + ": " + text);
  }
  return value;
}

std::uint64_t fnv1a64(const std::string &text) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::vector<Parameter> loadParameters(const std::string &path) {
  const auto rows = readCSV(path);
  const std::vector<std::string> expected{"name", "base", "minimum", "maximum"};
  if (rows.front() != expected) {
    throw std::runtime_error(
        "parameter CSV header must be name,base,minimum,maximum");
  }
  std::vector<Parameter> parameters;
  std::unordered_set<std::string> names;
  for (std::size_t row = 1u; row < rows.size(); ++row) {
    if (rows[row].size() != expected.size()) {
      throw std::runtime_error("parameter CSV row has wrong field count");
    }
    Parameter parameter{
        .name = rows[row][0],
        .base = parseFinite(rows[row][1], "base"),
        .minimum = parseFinite(rows[row][2], "minimum"),
        .maximum = parseFinite(rows[row][3], "maximum"),
    };
    if (parameter.name.empty() || !names.insert(parameter.name).second) {
      throw std::runtime_error("parameter names must be nonempty and unique");
    }
    if (!(parameter.minimum > 0.0 && parameter.minimum < parameter.base &&
          parameter.base < parameter.maximum)) {
      throw std::runtime_error(
          "parameter bounds must satisfy 0 < minimum < base < "
          "maximum for " +
          parameter.name);
    }
    parameters.push_back(std::move(parameter));
  }
  if (parameters.empty()) {
    throw std::runtime_error("parameter CSV has no parameters");
  }
  return parameters;
}

std::vector<Observation>
loadObservations(const std::string &path,
                 const std::vector<Parameter> &parameters) {
  const auto rows = readCSV(path);
  if (rows.front().size() != 6u + parameters.size()) {
    throw std::runtime_error(
        "observation CSV must have six fixed fields and one dlog_ "
        "sensitivity per parameter");
  }
  const std::vector<std::string> fixed{"trial_id", "split", "observable",
                                       "measured", "sigma", "baseline"};
  for (std::size_t index = 0u; index < fixed.size(); ++index) {
    if (rows.front()[index] != fixed[index]) {
      throw std::runtime_error("observation CSV fixed header mismatch at " +
                               fixed[index]);
    }
  }
  std::unordered_map<std::string, std::size_t> sensitivityColumn;
  for (std::size_t column = fixed.size(); column < rows.front().size();
       ++column) {
    const std::string &name = rows.front()[column];
    if (!name.starts_with("dlog_")) {
      throw std::runtime_error("sensitivity columns must start with dlog_");
    }
    if (!sensitivityColumn.emplace(name.substr(5u), column).second) {
      throw std::runtime_error("duplicate sensitivity column: " + name);
    }
  }
  for (const Parameter &parameter : parameters) {
    if (!sensitivityColumn.contains(parameter.name)) {
      throw std::runtime_error("missing dlog_ sensitivity for " +
                               parameter.name);
    }
  }

  std::vector<Observation> observations;
  std::unordered_set<std::string> trialObservablePairs;
  std::unordered_map<std::string, Split> trialSplits;
  for (std::size_t row = 1u; row < rows.size(); ++row) {
    if (rows[row].size() != rows.front().size()) {
      throw std::runtime_error("observation CSV row has wrong field count");
    }
    Split split{};
    if (rows[row][1] == "calibration") {
      split = Split::calibration;
    } else if (rows[row][1] == "heldout") {
      split = Split::heldout;
    } else {
      throw std::runtime_error("split must be calibration or heldout");
    }
    Observation observation{
        .trial = rows[row][0],
        .split = split,
        .observable = rows[row][2],
        .measured = parseFinite(rows[row][3], "measured"),
        .sigma = parseFinite(rows[row][4], "sigma"),
        .baseline = parseFinite(rows[row][5], "baseline"),
    };
    if (observation.trial.empty() || observation.observable.empty() ||
        !(observation.sigma > 0.0)) {
      throw std::runtime_error(
          "trial, observable, and positive sigma are required");
    }
    const auto [trialSplit, inserted] =
        trialSplits.emplace(observation.trial, observation.split);
    if (!inserted && trialSplit->second != observation.split) {
      throw std::runtime_error(
          "trial IDs may not cross calibration and heldout splits: " +
          observation.trial);
    }
    const std::string identity =
        observation.trial + "\n" + observation.observable;
    if (!trialObservablePairs.insert(identity).second) {
      throw std::runtime_error("duplicate trial and observable pair: " +
                               observation.trial);
    }
    observation.sensitivities.reserve(parameters.size());
    for (const Parameter &parameter : parameters) {
      observation.sensitivities.push_back(parseFinite(
          rows[row][sensitivityColumn.at(parameter.name)], "sensitivity"));
    }
    observations.push_back(std::move(observation));
  }
  return observations;
}

std::vector<double>
symmetricEigenvalues(std::vector<std::vector<double>> matrix) {
  const std::size_t size = matrix.size();
  if (size == 0u) {
    return {};
  }
  constexpr std::size_t sweeps = 100u;
  for (std::size_t sweep = 0u; sweep < sweeps * size * size; ++sweep) {
    std::size_t first = 0u;
    std::size_t second = 0u;
    double largest = 0.0;
    for (std::size_t row = 0u; row < size; ++row) {
      for (std::size_t column = row + 1u; column < size; ++column) {
        const double magnitude = std::abs(matrix[row][column]);
        if (magnitude > largest) {
          largest = magnitude;
          first = row;
          second = column;
        }
      }
    }
    if (largest <= 1.0e-14) {
      break;
    }
    const double angle =
        0.5 * std::atan2(2.0 * matrix[first][second],
                         matrix[second][second] - matrix[first][first]);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (std::size_t index = 0u; index < size; ++index) {
      if (index == first || index == second) {
        continue;
      }
      const double firstValue = matrix[index][first];
      const double secondValue = matrix[index][second];
      matrix[index][first] = matrix[first][index] =
          cosine * firstValue - sine * secondValue;
      matrix[index][second] = matrix[second][index] =
          sine * firstValue + cosine * secondValue;
    }
    const double firstDiagonal = matrix[first][first];
    const double secondDiagonal = matrix[second][second];
    const double offDiagonal = matrix[first][second];
    matrix[first][first] = cosine * cosine * firstDiagonal -
                           2.0 * sine * cosine * offDiagonal +
                           sine * sine * secondDiagonal;
    matrix[second][second] = sine * sine * firstDiagonal +
                             2.0 * sine * cosine * offDiagonal +
                             cosine * cosine * secondDiagonal;
    matrix[first][second] = matrix[second][first] = 0.0;
  }
  std::vector<double> eigenvalues(size);
  for (std::size_t index = 0u; index < size; ++index) {
    eigenvalues[index] = matrix[index][index];
  }
  std::sort(eigenvalues.begin(), eigenvalues.end());
  return eigenvalues;
}

std::vector<double> solveLinearSystem(std::vector<std::vector<double>> matrix,
                                      std::vector<double> rightHandSide) {
  const std::size_t size = matrix.size();
  for (std::size_t column = 0u; column < size; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1u; row < size; ++row) {
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    if (std::abs(matrix[pivot][column]) <= 1.0e-14) {
      throw std::runtime_error("calibration normal matrix is singular");
    }
    std::swap(matrix[pivot], matrix[column]);
    std::swap(rightHandSide[pivot], rightHandSide[column]);
    const double divisor = matrix[column][column];
    for (std::size_t entry = column; entry < size; ++entry) {
      matrix[column][entry] /= divisor;
    }
    rightHandSide[column] /= divisor;
    for (std::size_t row = 0u; row < size; ++row) {
      if (row == column) {
        continue;
      }
      const double factor = matrix[row][column];
      for (std::size_t entry = column; entry < size; ++entry) {
        matrix[row][entry] -= factor * matrix[column][entry];
      }
      rightHandSide[row] -= factor * rightHandSide[column];
    }
  }
  return rightHandSide;
}

Metrics evaluate(const std::vector<Observation> &observations,
                 const Split split, const std::vector<double> &logDeltas) {
  double squared = 0.0;
  double maximum = 0.0;
  std::size_t count = 0u;
  for (const Observation &observation : observations) {
    if (observation.split != split) {
      continue;
    }
    double predicted = observation.baseline;
    for (std::size_t index = 0u; index < logDeltas.size(); ++index) {
      predicted += observation.sensitivities[index] * logDeltas[index];
    }
    const double z = (predicted - observation.measured) / observation.sigma;
    squared += z * z;
    maximum = std::max(maximum, std::abs(z));
    ++count;
  }
  if (count == 0u) {
    return {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
  }
  return {std::sqrt(squared / static_cast<double>(count)), maximum};
}

Options parseOptions(const int argc, const char *const *argv) {
  Options options;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string_view value(argv[argument]);
    const auto requireValue = [&](const std::string_view name) {
      if (argument + 1 >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
    };
    if (value == "--parameters") {
      requireValue(value);
      options.parametersPath = argv[++argument];
    } else if (value == "--observations") {
      requireValue(value);
      options.observationsPath = argv[++argument];
    } else if (value == "--report") {
      requireValue(value);
      options.reportPath = argv[++argument];
    } else if (value == "--material-output") {
      requireValue(value);
      options.materialPath = argv[++argument];
    } else if (value == "--ridge") {
      requireValue(value);
      options.ridge = parseFinite(argv[++argument], "ridge");
    } else if (value == "--max-condition") {
      requireValue(value);
      options.maximumCondition = parseFinite(argv[++argument], "max-condition");
    } else if (value == "--max-calibration-rmse-z") {
      requireValue(value);
      options.maximumCalibrationRMSEZ =
          parseFinite(argv[++argument], "max-calibration-rmse-z");
    } else if (value == "--max-heldout-rmse-z") {
      requireValue(value);
      options.maximumHeldoutRMSEZ =
          parseFinite(argv[++argument], "max-heldout-rmse-z");
    } else if (value == "--max-heldout-max-z") {
      requireValue(value);
      options.maximumHeldoutAbsoluteZ =
          parseFinite(argv[++argument], "max-heldout-max-z");
    } else if (value == "--allow-no-heldout-improvement") {
      options.requireHeldoutImprovement = false;
    } else if (value == "--help") {
      std::cout << "usage: numi-solver-cloth-calibration "
                   "--parameters FILE --observations FILE "
                   "[--report FILE] [--material-output FILE] "
                   "[--ridge LAMBDA] [--max-condition KAPPA] "
                   "[--max-calibration-rmse-z Z] "
                   "[--max-heldout-rmse-z Z] [--max-heldout-max-z Z] "
                   "[--allow-no-heldout-improvement]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(value));
    }
  }
  if (options.parametersPath.empty() || options.observationsPath.empty()) {
    throw std::runtime_error("--parameters and --observations are required");
  }
  if (options.ridge < 0.0 || options.maximumCondition <= 1.0 ||
      options.maximumCalibrationRMSEZ <= 0.0 ||
      options.maximumHeldoutRMSEZ <= 0.0 ||
      options.maximumHeldoutAbsoluteZ <= 0.0) {
    throw std::runtime_error("qualification thresholds are invalid");
  }
  return options;
}

std::string
formatReport(const Options &options, const std::vector<Parameter> &parameters,
             const std::vector<double> &logDeltas,
             const std::size_t calibrationRows, const std::size_t heldoutRows,
             const std::size_t rank, const double condition,
             const Metrics baselineCalibration, const Metrics fittedCalibration,
             const Metrics baselineHeldout, const Metrics fittedHeldout,
             const bool boundsQualified, const bool heldoutImproved,
             const bool qualified, const std::uint64_t parameterHash,
             const std::uint64_t observationHash) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(12) << "schema=" << kResultSchema
         << '\n'
         << "parameters_hash=0x" << std::hex << parameterHash << '\n'
         << "observations_hash=0x" << observationHash << std::dec << '\n'
         << "parameter_count=" << parameters.size()
         << " calibration_rows=" << calibrationRows
         << " heldout_rows=" << heldoutRows << '\n'
         << "rank=" << rank << " condition_number=" << condition
         << " maximum_condition=" << options.maximumCondition << '\n'
         << "baseline_calibration_rmse_z=" << baselineCalibration.rmseZ
         << " fitted_calibration_rmse_z=" << fittedCalibration.rmseZ
         << " maximum_calibration_rmse_z=" << options.maximumCalibrationRMSEZ
         << '\n'
         << "baseline_heldout_rmse_z=" << baselineHeldout.rmseZ
         << " fitted_heldout_rmse_z=" << fittedHeldout.rmseZ
         << " fitted_heldout_max_z=" << fittedHeldout.maximumAbsoluteZ
         << " maximum_heldout_rmse_z=" << options.maximumHeldoutRMSEZ
         << " maximum_heldout_max_z=" << options.maximumHeldoutAbsoluteZ << '\n'
         << "heldout_improved=" << std::boolalpha << heldoutImproved
         << " improvement_required=" << options.requireHeldoutImprovement
         << " bounds_qualified=" << boundsQualified << '\n';
  for (std::size_t index = 0u; index < parameters.size(); ++index) {
    const double fitted = parameters[index].base * std::exp(logDeltas[index]);
    output << "parameter=" << parameters[index].name
           << " base=" << parameters[index].base << " fitted=" << fitted
           << " minimum=" << parameters[index].minimum
           << " maximum=" << parameters[index].maximum
           << " log_delta=" << logDeltas[index] << '\n';
  }
  output << "qualified=" << qualified << '\n'
         << "result=" << (qualified ? "PASS" : "FAIL") << '\n';
  return output.str();
}

void writeFile(const std::string &path, const std::string &contents) {
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot create " + path);
  }
  stream << contents;
  if (!stream) {
    throw std::runtime_error("cannot write " + path);
  }
}

int run(const int argc, const char *const *argv) {
  const Options options = parseOptions(argc, argv);
  const std::string parameterText = readText(options.parametersPath);
  const std::string observationText = readText(options.observationsPath);
  const std::vector<Parameter> parameters =
      loadParameters(options.parametersPath);
  const std::vector<Observation> observations =
      loadObservations(options.observationsPath, parameters);

  std::vector<const Observation *> calibration;
  std::size_t heldoutRows = 0u;
  for (const Observation &observation : observations) {
    if (observation.split == Split::calibration) {
      calibration.push_back(&observation);
    } else {
      ++heldoutRows;
    }
  }
  if (calibration.size() < parameters.size()) {
    throw std::runtime_error(
        "calibration rows must be at least the parameter count");
  }
  if (heldoutRows < 2u) {
    throw std::runtime_error("at least two heldout rows are required");
  }

  const std::size_t count = parameters.size();
  std::vector<double> columnNorms(count, 0.0);
  std::vector<std::vector<double>> design(calibration.size(),
                                          std::vector<double>(count, 0.0));
  std::vector<double> target(calibration.size(), 0.0);
  for (std::size_t row = 0u; row < calibration.size(); ++row) {
    target[row] = (calibration[row]->measured - calibration[row]->baseline) /
                  calibration[row]->sigma;
    for (std::size_t column = 0u; column < count; ++column) {
      design[row][column] =
          calibration[row]->sensitivities[column] / calibration[row]->sigma;
      columnNorms[column] += design[row][column] * design[row][column];
    }
  }
  for (double &norm : columnNorms) {
    norm = std::sqrt(norm);
    if (!(norm > 0.0)) {
      throw std::runtime_error(
          "each parameter needs nonzero calibration sensitivity");
    }
  }

  std::vector<std::vector<double>> gram(count, std::vector<double>(count, 0.0));
  std::vector<double> normalRightHandSide(count, 0.0);
  for (std::size_t row = 0u; row < design.size(); ++row) {
    for (std::size_t first = 0u; first < count; ++first) {
      const double firstValue = design[row][first] / columnNorms[first];
      normalRightHandSide[first] += firstValue * target[row];
      for (std::size_t second = 0u; second < count; ++second) {
        gram[first][second] +=
            firstValue * design[row][second] / columnNorms[second];
      }
    }
  }

  const std::vector<double> eigenvalues = symmetricEigenvalues(gram);
  const double maximumEigenvalue = eigenvalues.back();
  const double rankTolerance =
      maximumEigenvalue / (options.maximumCondition * options.maximumCondition);
  const std::size_t rank = static_cast<std::size_t>(std::count_if(
      eigenvalues.begin(), eigenvalues.end(),
      [rankTolerance](const double value) { return value > rankTolerance; }));
  const double minimumEigenvalue = eigenvalues.front();
  const double condition =
      minimumEigenvalue > 0.0 ? std::sqrt(maximumEigenvalue / minimumEigenvalue)
                              : std::numeric_limits<double>::infinity();

  std::vector<double> logDeltas(count, 0.0);
  if (rank == count && condition <= options.maximumCondition) {
    std::vector<std::vector<double>> regularized = gram;
    for (std::size_t index = 0u; index < count; ++index) {
      regularized[index][index] += options.ridge;
    }
    const std::vector<double> normalizedSolution =
        solveLinearSystem(regularized, normalRightHandSide);
    for (std::size_t index = 0u; index < count; ++index) {
      logDeltas[index] = normalizedSolution[index] / columnNorms[index];
    }
  }

  const std::vector<double> zero(count, 0.0);
  const Metrics baselineCalibration =
      evaluate(observations, Split::calibration, zero);
  const Metrics fittedCalibration =
      evaluate(observations, Split::calibration, logDeltas);
  const Metrics baselineHeldout = evaluate(observations, Split::heldout, zero);
  const Metrics fittedHeldout =
      evaluate(observations, Split::heldout, logDeltas);
  bool boundsQualified = true;
  for (std::size_t index = 0u; index < count; ++index) {
    const double fitted = parameters[index].base * std::exp(logDeltas[index]);
    boundsQualified = boundsQualified && fitted >= parameters[index].minimum &&
                      fitted <= parameters[index].maximum;
  }
  const bool heldoutImproved =
      fittedHeldout.rmseZ + 1.0e-12 < baselineHeldout.rmseZ;
  const bool qualified =
      rank == count && condition <= options.maximumCondition &&
      boundsQualified &&
      fittedCalibration.rmseZ <= options.maximumCalibrationRMSEZ &&
      fittedHeldout.rmseZ <= options.maximumHeldoutRMSEZ &&
      fittedHeldout.maximumAbsoluteZ <= options.maximumHeldoutAbsoluteZ &&
      (!options.requireHeldoutImprovement || heldoutImproved);

  const std::uint64_t parameterHash = fnv1a64(parameterText);
  const std::uint64_t observationHash = fnv1a64(observationText);
  const std::string report = formatReport(
      options, parameters, logDeltas, calibration.size(), heldoutRows, rank,
      condition, baselineCalibration, fittedCalibration, baselineHeldout,
      fittedHeldout, boundsQualified, heldoutImproved, qualified, parameterHash,
      observationHash);
  std::cout << report;
  if (!options.reportPath.empty()) {
    writeFile(options.reportPath, report);
  }
  if (!options.materialPath.empty()) {
    if (!qualified) {
      throw std::runtime_error(
          "refusing material output from unqualified calibration");
    }
    std::ostringstream material;
    material << std::fixed << std::setprecision(12)
             << "schema=" << kMaterialSchema << '\n'
             << "parameters_hash=0x" << std::hex << parameterHash << '\n'
             << "observations_hash=0x" << observationHash << std::dec << '\n';
    for (std::size_t index = 0u; index < count; ++index) {
      material << parameters[index].name << '='
               << parameters[index].base * std::exp(logDeltas[index]) << '\n';
    }
    writeFile(options.materialPath, material.str());
  }
  return qualified ? 0 : 1;
}

} // namespace

int main(const int argc, const char *const *argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 2;
  }
}

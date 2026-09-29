#ifndef NUMI_GRIP_TRAJECTORY_H
#define NUMI_GRIP_TRAJECTORY_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace numi {

inline constexpr std::string_view kGripTrajectorySchemaV1 =
    "numi.grip.trajectory.v1";
inline constexpr std::string_view kGripTrajectorySchemaV2 =
    "numi.grip.trajectory.v2";
inline constexpr std::string_view kGripTrajectorySchemaV3 =
    "numi.grip.trajectory.v3";
inline constexpr std::string_view kGripTrajectorySchema =
    kGripTrajectorySchemaV3;

struct GripTrajectoryVector3 {
  double x{};
  double y{};
  double z{};
};

struct GripTrajectoryQuaternion {
  double x{};
  double y{};
  double z{};
  double w{1.0};
};

struct GripTrajectoryPose {
  double timeSeconds{};
  GripTrajectoryVector3 translationMeters{};
  GripTrajectoryQuaternion orientation{};
  bool active{true};
  std::uint32_t attachmentGeneration{1u};
};

struct GripTrajectory {
  std::vector<GripTrajectoryPose> poses;
  std::string schema;
  std::string contentFingerprint;
  bool selectNearestCuffPatch{};
};

inline std::string trimGripTrajectory(std::string value) {
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

inline double parseGripTrajectoryNumber(const std::string &text,
                                        const std::string_view field) {
  std::size_t consumed = 0u;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid grip trajectory number for " +
                             std::string(field));
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("non-finite grip trajectory number for " +
                             std::string(field));
  }
  return value;
}

inline std::vector<std::string>
splitGripTrajectoryRow(const std::string &line) {
  std::vector<std::string> fields;
  std::size_t begin = 0u;
  while (begin <= line.size()) {
    const std::size_t separator = line.find(',', begin);
    fields.push_back(trimGripTrajectory(line.substr(
        begin, separator == std::string::npos ? std::string::npos
                                              : separator - begin)));
    if (separator == std::string::npos) {
      break;
    }
    begin = separator + 1u;
  }
  return fields;
}

inline double gripQuaternionDot(const GripTrajectoryQuaternion first,
                                const GripTrajectoryQuaternion second) {
  return first.x * second.x + first.y * second.y + first.z * second.z +
         first.w * second.w;
}

inline double gripQuaternionLength(const GripTrajectoryQuaternion quaternion) {
  return std::sqrt(gripQuaternionDot(quaternion, quaternion));
}

inline GripTrajectoryQuaternion
normalizedGripQuaternion(const GripTrajectoryQuaternion quaternion) {
  const double magnitude = gripQuaternionLength(quaternion);
  if (!(magnitude > 1.0e-14)) {
    throw std::runtime_error("grip trajectory quaternion has zero length");
  }
  return {quaternion.x / magnitude, quaternion.y / magnitude,
          quaternion.z / magnitude, quaternion.w / magnitude};
}

inline GripTrajectoryQuaternion
scaledGripQuaternion(const GripTrajectoryQuaternion quaternion,
                     const double scale) {
  return {quaternion.x * scale, quaternion.y * scale, quaternion.z * scale,
          quaternion.w * scale};
}

inline GripTrajectoryQuaternion
interpolateGripQuaternion(const GripTrajectoryQuaternion first,
                          GripTrajectoryQuaternion second,
                          const double fraction) {
  double cosine = gripQuaternionDot(first, second);
  if (cosine < 0.0) {
    second = scaledGripQuaternion(second, -1.0);
    cosine = -cosine;
  }
  cosine = std::clamp(cosine, -1.0, 1.0);
  if (cosine > 0.9995) {
    return normalizedGripQuaternion({
        first.x + fraction * (second.x - first.x),
        first.y + fraction * (second.y - first.y),
        first.z + fraction * (second.z - first.z),
        first.w + fraction * (second.w - first.w),
    });
  }
  const double angle = std::acos(cosine);
  const double sine = std::sin(angle);
  const double firstWeight = std::sin((1.0 - fraction) * angle) / sine;
  const double secondWeight = std::sin(fraction * angle) / sine;
  return normalizedGripQuaternion({
      firstWeight * first.x + secondWeight * second.x,
      firstWeight * first.y + secondWeight * second.y,
      firstWeight * first.z + secondWeight * second.z,
      firstWeight * first.w + secondWeight * second.w,
  });
}

inline GripTrajectoryVector3
rotateGripVector(const GripTrajectoryQuaternion quaternion,
                 const GripTrajectoryVector3 vector) {
  const GripTrajectoryVector3 twiceCross{
      2.0 * (quaternion.y * vector.z - quaternion.z * vector.y),
      2.0 * (quaternion.z * vector.x - quaternion.x * vector.z),
      2.0 * (quaternion.x * vector.y - quaternion.y * vector.x),
  };
  return {
      vector.x + quaternion.w * twiceCross.x + quaternion.y * twiceCross.z -
          quaternion.z * twiceCross.y,
      vector.y + quaternion.w * twiceCross.y + quaternion.z * twiceCross.x -
          quaternion.x * twiceCross.z,
      vector.z + quaternion.w * twiceCross.z + quaternion.x * twiceCross.y -
          quaternion.y * twiceCross.x,
  };
}

inline std::string gripTrajectoryFingerprint(const std::string &text) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  std::ostringstream output;
  output << "0x" << std::hex << hash;
  return output.str();
}

inline GripTrajectory loadGripTrajectory(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open grip trajectory: " + path);
  }
  std::ostringstream sourceStream;
  sourceStream << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw std::runtime_error("cannot read grip trajectory: " + path);
  }
  const std::string source = sourceStream.str();
  std::istringstream lines(source);
  std::string line;
  std::size_t lineNumber = 0u;
  bool sawSchema = false;
  bool sawHeader = false;
  bool permitsReactivation = false;
  GripTrajectory result;
  result.contentFingerprint = gripTrajectoryFingerprint(source);
  const std::vector<std::string> expectedHeader{
      "time_s",          "translation_x_m", "translation_y_m",
      "translation_z_m", "quaternion_x",    "quaternion_y",
      "quaternion_z",    "quaternion_w",    "active",
  };
  bool previousActive = true;
  std::uint32_t attachmentGeneration = 1u;
  while (std::getline(lines, line)) {
    ++lineNumber;
    line = trimGripTrajectory(std::move(line));
    if (line.empty() || line.starts_with('#')) {
      continue;
    }
    if (!sawSchema) {
      const std::string schemaV1 =
          "schema=" + std::string(kGripTrajectorySchemaV1);
      const std::string schemaV2 =
          "schema=" + std::string(kGripTrajectorySchemaV2);
      const std::string schemaV3 =
          "schema=" + std::string(kGripTrajectorySchemaV3);
      if (line != schemaV1 && line != schemaV2 && line != schemaV3) {
        throw std::runtime_error("grip trajectory schema mismatch");
      }
      result.schema = line.substr(std::string("schema=").size());
      permitsReactivation = line == schemaV2 || line == schemaV3;
      result.selectNearestCuffPatch = line == schemaV3;
      sawSchema = true;
      continue;
    }
    const std::vector<std::string> fields = splitGripTrajectoryRow(line);
    if (!sawHeader) {
      if (fields != expectedHeader) {
        throw std::runtime_error("grip trajectory CSV header mismatch");
      }
      sawHeader = true;
      continue;
    }
    if (fields.size() != expectedHeader.size()) {
      throw std::runtime_error("grip trajectory row has wrong field count at " +
                               std::to_string(lineNumber));
    }
    GripTrajectoryPose pose{
        .timeSeconds = parseGripTrajectoryNumber(fields[0], "time_s"),
        .translationMeters =
            {parseGripTrajectoryNumber(fields[1], "translation_x_m"),
             parseGripTrajectoryNumber(fields[2], "translation_y_m"),
             parseGripTrajectoryNumber(fields[3], "translation_z_m")},
        .orientation = {parseGripTrajectoryNumber(fields[4], "quaternion_x"),
                        parseGripTrajectoryNumber(fields[5], "quaternion_y"),
                        parseGripTrajectoryNumber(fields[6], "quaternion_z"),
                        parseGripTrajectoryNumber(fields[7], "quaternion_w")},
    };
    if (fields[8] == "1") {
      pose.active = true;
    } else if (fields[8] == "0") {
      pose.active = false;
    } else {
      throw std::runtime_error("grip trajectory active must be 0 or 1");
    }
    if (pose.timeSeconds < 0.0 ||
        (!result.poses.empty() &&
         !(pose.timeSeconds > result.poses.back().timeSeconds))) {
      throw std::runtime_error(
          "grip trajectory times must be nonnegative and strictly increasing");
    }
    const double quaternionMagnitude = gripQuaternionLength(pose.orientation);
    if (std::abs(quaternionMagnitude - 1.0) > 1.0e-3) {
      throw std::runtime_error(
          "grip trajectory quaternion must be unit length within 1e-3");
    }
    pose.orientation = normalizedGripQuaternion(pose.orientation);
    if (!result.poses.empty() &&
        gripQuaternionDot(result.poses.back().orientation, pose.orientation) <
            0.0) {
      pose.orientation = scaledGripQuaternion(pose.orientation, -1.0);
    }
    if (!previousActive && pose.active) {
      if (!permitsReactivation) {
        throw std::runtime_error(
            "grip trajectory v1 cannot reactivate after release");
      }
      if (attachmentGeneration ==
          std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "grip trajectory has too many attachment generations");
      }
      ++attachmentGeneration;
    }
    pose.attachmentGeneration = attachmentGeneration;
    previousActive = pose.active;
    result.poses.push_back(pose);
  }
  if (!sawSchema || !sawHeader || result.poses.size() < 2u) {
    throw std::runtime_error(
        "grip trajectory requires schema, header, and at least two poses");
  }
  const GripTrajectoryPose &first = result.poses.front();
  const double initialTranslation =
      std::sqrt(first.translationMeters.x * first.translationMeters.x +
                first.translationMeters.y * first.translationMeters.y +
                first.translationMeters.z * first.translationMeters.z);
  const double initialRotation =
      2.0 * std::acos(std::clamp(std::abs(first.orientation.w), 0.0, 1.0));
  if (std::abs(first.timeSeconds) > 1.0e-12 || initialTranslation > 1.0e-9 ||
      initialRotation > 1.0e-9 || !first.active) {
    throw std::runtime_error(
        "grip trajectory must start at t=0 with zero translation, identity "
        "orientation, and an active seam grip");
  }
  return result;
}

inline GripTrajectoryPose sampleGripTrajectory(const GripTrajectory &trajectory,
                                               const double timeSeconds) {
  constexpr double timeTolerance = 1.0e-9;
  if (!std::isfinite(timeSeconds) || timeSeconds < 0.0 ||
      trajectory.poses.empty() ||
      timeSeconds > trajectory.poses.back().timeSeconds + timeTolerance) {
    throw std::runtime_error("grip trajectory sample is outside its duration");
  }
  if (timeSeconds <= trajectory.poses.front().timeSeconds) {
    return trajectory.poses.front();
  }
  if (timeSeconds >= trajectory.poses.back().timeSeconds) {
    return trajectory.poses.back();
  }
  const auto upper = std::lower_bound(
      trajectory.poses.begin(), trajectory.poses.end(), timeSeconds,
      [](const GripTrajectoryPose &pose, const double time) {
        return pose.timeSeconds < time;
      });
  if (upper == trajectory.poses.end()) {
    return trajectory.poses.back();
  }
  const GripTrajectoryPose &second = *upper;
  const GripTrajectoryPose &first = *(upper - 1);
  const double fraction = (timeSeconds - first.timeSeconds) /
                          (second.timeSeconds - first.timeSeconds);
  return {
      .timeSeconds = timeSeconds,
      .translationMeters =
          {first.translationMeters.x + fraction * (second.translationMeters.x -
                                                   first.translationMeters.x),
           first.translationMeters.y + fraction * (second.translationMeters.y -
                                                   first.translationMeters.y),
           first.translationMeters.z + fraction * (second.translationMeters.z -
                                                   first.translationMeters.z)},
      .orientation = interpolateGripQuaternion(first.orientation,
                                               second.orientation, fraction),
      .active = fraction >= 1.0 ? second.active : first.active,
      .attachmentGeneration = fraction >= 1.0
          ? second.attachmentGeneration
          : first.attachmentGeneration,
  };
}

inline bool gripTrajectoryCovers(const GripTrajectory &trajectory,
                                 const double durationSeconds) {
  return std::isfinite(durationSeconds) && durationSeconds >= 0.0 &&
         !trajectory.poses.empty() &&
         trajectory.poses.back().timeSeconds + 1.0e-9 >= durationSeconds;
}

inline double maximumGripTrajectoryRotation(const GripTrajectory &trajectory) {
  double maximum = 0.0;
  for (const GripTrajectoryPose &pose : trajectory.poses) {
    maximum = std::max(
        maximum,
        2.0 * std::acos(std::clamp(std::abs(pose.orientation.w), 0.0, 1.0)));
  }
  return maximum;
}

inline std::uint32_t
gripTrajectoryAttachmentGenerations(const GripTrajectory &trajectory) {
  std::uint32_t maximum = 0u;
  for (const GripTrajectoryPose &pose : trajectory.poses) {
    maximum = std::max(maximum, pose.attachmentGeneration);
  }
  return maximum;
}

} // namespace numi

#endif

#!/usr/bin/env swift

import AppKit
import Foundation

private struct Vec3 {
    var x: Double
    var y: Double
    var z: Double

    static func +(lhs: Vec3, rhs: Vec3) -> Vec3 {
        Vec3(x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z)
    }

    static func -(lhs: Vec3, rhs: Vec3) -> Vec3 {
        Vec3(x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z)
    }

    static func *(lhs: Vec3, rhs: Double) -> Vec3 {
        Vec3(x: lhs.x * rhs, y: lhs.y * rhs, z: lhs.z * rhs)
    }
}

private struct Projected {
    var point: CGPoint
    var depth: Double
}

private func cross(_ first: Vec3, _ second: Vec3) -> Vec3 {
    Vec3(
        x: first.y * second.z - first.z * second.y,
        y: first.z * second.x - first.x * second.z,
        z: first.x * second.y - first.y * second.x
    )
}

private struct Quaternion {
    var w: Double
    var x: Double
    var y: Double
    var z: Double

    static let identity = Quaternion(w: 1.0, x: 0.0, y: 0.0, z: 0.0)

    func rotate(_ point: Vec3) -> Vec3 {
        let vector = Vec3(x: x, y: y, z: z)
        let firstCross = cross(vector, point)
        return point + firstCross * (2.0 * w) +
            cross(vector, firstCross) * 2.0
    }
}

private struct Fruit {
    var center: Vec3
    var radius: Double
    var appearance: Int
    var orientation: Quaternion
}

private struct Grip {
    var center: Vec3
    var active: Bool
    var orientation: Quaternion
    var patchCenterRing: Int
}

private struct StaticBench {
    let minimum = Vec3(x: -0.75, y: -0.5, z: -0.08)
    let maximum = Vec3(x: 0.75, y: 0.5, z: 0)
    let floorHeight = -0.75

    var corners: [Vec3] {
        [minimum.x, maximum.x].flatMap { x in
            [minimum.y, maximum.y].flatMap { y in
                [minimum.z, maximum.z].map { z in Vec3(x: x, y: y, z: z) }
            }
        }
    }
}

private enum PrimitiveKind {
    case bench([CGPoint], CGColor)
    case shadow([CGPoint], CGColor)
    case yarn(CGPoint, CGPoint, Bool)
    case fruit(CGPoint, Fruit)
}

private struct Primitive {
    var depth: Double
    var kind: PrimitiveKind
}

private let around = 48
private let levels = 28
private let bottomGrid = 13
private let bottomInterior = bottomGrid - 2
private let expectedVertices = around * levels + bottomInterior * bottomInterior
private let clothRadiusMeters = 0.004
private let cameraYaw = -0.62
private let cameraPitch = 0.19

private struct ViewBounds {
    var minimumX = Double.infinity
    var maximumX = -Double.infinity
    var minimumY = Double.infinity
    var maximumY = -Double.infinity

    mutating func include(_ point: Vec3, radius: Double) {
        let transformed = camera(point, yaw: cameraYaw, pitch: cameraPitch)
        minimumX = min(minimumX, transformed.x - radius)
        maximumX = max(maximumX, transformed.x + radius)
        minimumY = min(minimumY, transformed.y - radius)
        maximumY = max(maximumY, transformed.y + radius)
    }
}

private func trajectoryBounds(listPath: String) throws -> ViewBounds {
    let listURL = URL(fileURLWithPath: listPath)
    let lines = try String(contentsOf: listURL, encoding: .utf8)
        .split(separator: "\n").map(String.init)
    guard !lines.isEmpty else {
        throw NSError(domain: "NumiClothRenderer", code: 6,
                      userInfo: [NSLocalizedDescriptionKey: "framing list is empty"])
    }
    var bounds = ViewBounds()
    for line in lines {
        let path = line.hasPrefix("/") ? line :
            listURL.deletingLastPathComponent().appendingPathComponent(line).path
        let (vertices, fruits, grip, bench) = try parseOBJ(at: path)
        if let bench {
            for corner in bench.corners {
                bounds.include(corner, radius: 0.01)
                bounds.include(Vec3(x: corner.x, y: corner.y, z: bench.floorHeight), radius: 0.01)
            }
        }
        for vertex in vertices { bounds.include(vertex, radius: clothRadiusMeters) }
        for fruit in fruits {
            bounds.include(fruit.center, radius: fruit.radius)
            let shadowHeight = bench?.floorHeight ?? 0.0
            bounds.include(Vec3(x: fruit.center.x, y: fruit.center.y, z: shadowHeight),
                           radius: fruit.radius * 1.1 +
                               0.08 * max(0, fruit.center.z - shadowHeight - fruit.radius))
        }
        if let grip { bounds.include(grip.center, radius: 0.03) }
    }
    return bounds
}

private func parseOBJ(at path: String) throws -> ([Vec3], [Fruit], Grip?, StaticBench?) {
    let source = try String(contentsOfFile: path, encoding: .utf8)
    var vertices: [Vec3] = []
    var fruits: [Fruit] = []
    var grip: Grip?
    var bench: StaticBench?
    for line in source.split(separator: "\n") {
        let fields = line.split(separator: " ")
        guard let first = fields.first else { continue }
        if fields.prefix(2) == ["#", "static_bench"] {
            guard bench == nil, fields == ["#", "static_bench", "min", "-0.75", "-0.5", "-0.08",
                                           "max", "0.75", "0.5", "0", "floor", "-0.75"] else {
                throw NSError(domain: "NumiClothRenderer", code: 8,
                              userInfo: [NSLocalizedDescriptionKey: "invalid or duplicate authored bench geometry"])
            }
            bench = StaticBench()
        } else if first == "v", fields.count >= 4,
           let x = Double(fields[1]),
           let y = Double(fields[2]),
           let z = Double(fields[3]) {
            vertices.append(Vec3(x: x, y: y, z: z))
        } else if fields.count >= 11,
                  fields[0] == "#",
                  fields[1] == "ball",
                  fields[3] == "center",
                  fields[7] == "radius",
                  fields[9] == "appearance",
                  let x = Double(fields[4]),
                  let y = Double(fields[5]),
                  let z = Double(fields[6]),
                  let radius = Double(fields[8]),
                  let appearance = Int(fields[10]) {
            let orientation: Quaternion
            if fields.count >= 16,
               fields[11] == "orientation",
               let w = Double(fields[12]),
               let qx = Double(fields[13]),
               let qy = Double(fields[14]),
               let qz = Double(fields[15]) {
                orientation = Quaternion(w: w, x: qx, y: qy, z: qz)
            } else {
                orientation = .identity
            }
            fruits.append(Fruit(
                center: Vec3(x: x, y: y, z: z),
                radius: radius,
                appearance: appearance,
                orientation: orientation
            ))
        } else if fields.count >= 6,
                  fields[0] == "#",
                  fields[1] == "grip",
                  fields[2] == "center",
                  let x = Double(fields[3]),
                  let y = Double(fields[4]),
                  let z = Double(fields[5]) {
            let active = fields.count < 8 ||
                fields[6] != "active" || fields[7] == "1"
            let orientation: Quaternion
            if fields.count >= 13, fields[8] == "orientation",
               let w = Double(fields[9]),
               let qx = Double(fields[10]),
               let qy = Double(fields[11]),
               let qz = Double(fields[12]) {
                orientation = Quaternion(w: w, x: qx, y: qy, z: qz)
            } else {
                orientation = .identity
            }
            let patchCenterRing: Int
            if fields.count >= 15, fields[13] == "patch_center",
               let ring = Int(fields[14]) {
                patchCenterRing = ((ring % around) + around) % around
            } else {
                patchCenterRing = 0
            }
            grip = Grip(
                center: Vec3(x: x, y: y, z: z),
                active: active,
                orientation: orientation,
                patchCenterRing: patchCenterRing
            )
        }
    }
    guard vertices.count == expectedVertices else {
        throw NSError(
            domain: "NumiClothRenderer",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey:
                "expected \(expectedVertices) vertices, found \(vertices.count)"]
        )
    }
    return (vertices, fruits, grip, bench)
}

private func mix(_ first: Vec3, _ second: Vec3, _ t: Double) -> Vec3 {
    first + (second - first) * t
}

private func surface(_ vertices: [Vec3], level: Double, ring: Double) -> Vec3 {
    let level0 = max(0, min(levels - 1, Int(floor(level))))
    let level1 = min(levels - 1, level0 + 1)
    let levelT = level - Double(level0)
    let ringFloor = Int(floor(ring))
    let ring0 = (ringFloor % around + around) % around
    let ring1 = (ring0 + 1) % around
    let ringT = ring - Double(ringFloor)
    let first = mix(
        vertices[level0 * around + ring0],
        vertices[level0 * around + ring1],
        ringT
    )
    let second = mix(
        vertices[level1 * around + ring0],
        vertices[level1 * around + ring1],
        ringT
    )
    return mix(first, second, levelT)
}

private func concentricBottomCoordinate(row: Int, column: Int) -> CGPoint {
    let half = 0.5 * Double(bottomGrid - 1)
    let u = (Double(column) - half) / half
    let v = (Double(row) - half) / half
    if abs(u) < 1.0e-12 && abs(v) < 1.0e-12 {
        return .zero
    }
    let radius: Double
    let angle: Double
    if abs(u) >= abs(v) {
        radius = u
        angle = .pi / 4.0 * (v / u)
    } else {
        radius = v
        angle = .pi / 2.0 - .pi / 4.0 * (u / v)
    }
    return CGPoint(x: radius * cos(angle), y: radius * sin(angle))
}

private func bottomVertexIndex(row: Int, column: Int) -> Int {
    if row == 0 || column == 0 ||
       row == bottomGrid - 1 || column == bottomGrid - 1 {
        let coordinate = concentricBottomCoordinate(row: row, column: column)
        var angle = atan2(coordinate.y, coordinate.x)
        if angle < 0.0 { angle += 2.0 * .pi }
        return Int((angle * Double(around) / (2.0 * .pi)).rounded()) % around
    }
    return around * levels +
        (row - 1) * bottomInterior + column - 1
}

private func bottomSurface(
    _ vertices: [Vec3],
    row: Double,
    column: Double
) -> Vec3 {
    let row0 = max(0, min(bottomGrid - 1, Int(floor(row))))
    let row1 = min(bottomGrid - 1, row0 + 1)
    let column0 = max(0, min(bottomGrid - 1, Int(floor(column))))
    let column1 = min(bottomGrid - 1, column0 + 1)
    let rowT = row - Double(row0)
    let columnT = column - Double(column0)
    let first = mix(
        vertices[bottomVertexIndex(row: row0, column: column0)],
        vertices[bottomVertexIndex(row: row0, column: column1)],
        columnT
    )
    let second = mix(
        vertices[bottomVertexIndex(row: row1, column: column0)],
        vertices[bottomVertexIndex(row: row1, column: column1)],
        columnT
    )
    return mix(first, second, rowT)
}

private func camera(_ point: Vec3, yaw: Double, pitch: Double) -> Vec3 {
    let cosYaw = cos(yaw)
    let sinYaw = sin(yaw)
    let cosPitch = cos(pitch)
    let sinPitch = sin(pitch)
    let side = cosYaw * point.x - sinYaw * point.y
    let forward = sinYaw * point.x + cosYaw * point.y
    return Vec3(
        x: side,
        y: cosPitch * point.z - sinPitch * forward,
        z: sinPitch * point.z + cosPitch * forward
    )
}

private func color(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat,
                   _ alpha: CGFloat = 1.0) -> CGColor {
    CGColor(red: red / 255.0, green: green / 255.0,
            blue: blue / 255.0, alpha: alpha)
}

private func render(
    vertices: [Vec3],
    fruits: [Fruit],
    grip: Grip?,
    bench: StaticBench?,
    cameraProfile: String,
    framing: ViewBounds?,
    output: String
) throws {
    let pickupCamera = cameraProfile == "pickup" ||
        cameraProfile == "pickup-wide" || cameraProfile == "trajectory"
    let width = pickupCamera ? 960 : (grip == nil ? 1200 : 800)
    let height = 800
    let yaw = cameraYaw
    let pitch = cameraPitch
    let transformed = vertices.map { camera($0, yaw: yaw, pitch: pitch) }
    var minimumX = transformed.map(\.x).min()!
    var maximumX = transformed.map(\.x).max()!
    var minimumY = transformed.map(\.y).min()!
    var maximumY = transformed.map(\.y).max()!
    for fruit in fruits {
        let center = camera(fruit.center, yaw: yaw, pitch: pitch)
        minimumX = min(minimumX, center.x - fruit.radius)
        maximumX = max(maximumX, center.x + fruit.radius)
        minimumY = min(minimumY, center.y - fruit.radius)
        maximumY = max(maximumY, center.y + fruit.radius)
    }
    if let bench {
        for corner in bench.corners {
            for point in [corner, Vec3(x: corner.x, y: corner.y, z: bench.floorHeight)] {
                let view = camera(point, yaw: yaw, pitch: pitch)
                minimumX = min(minimumX, view.x); maximumX = max(maximumX, view.x)
                minimumY = min(minimumY, view.y); maximumY = max(maximumY, view.y)
            }
        }
    }
    let scale: Double
    let centerX: Double
    let centerY: Double
    if let framing {
        guard minimumX >= framing.minimumX && maximumX <= framing.maximumX &&
              minimumY >= framing.minimumY && maximumY <= framing.maximumY else {
            throw NSError(domain: "NumiClothRenderer", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                              "state lies outside the fixed trajectory framing list"])
        }
        scale = min(Double(width - 120) / (framing.maximumX - framing.minimumX),
                    Double(height - 120) / (framing.maximumY - framing.minimumY))
        centerX = Double(width) * 0.5 -
            (framing.minimumX + framing.maximumX) * 0.5 * scale
        centerY = Double(height) * 0.5 -
            (framing.minimumY + framing.maximumY) * 0.5 * scale
    } else if bench != nil && cameraProfile == "automatic" {
        scale = min(Double(width - 120) / (maximumX - minimumX),
                    Double(height - 120) / (maximumY - minimumY))
        centerX = Double(width) * 0.5 - (minimumX + maximumX) * 0.5 * scale
        centerY = Double(height) * 0.5 - (minimumY + maximumY) * 0.5 * scale
    } else if cameraProfile == "pickup-wide" {
        scale = 400.0
        centerX = Double(width) * 0.5
        centerY = 100.0
    } else if cameraProfile == "pickup" {
        scale = 450.0
        if let grip {
            let transformedGrip = camera(
                grip.center,
                yaw: yaw,
                pitch: pitch
            )
            centerX = Double(width) * 0.5 - transformedGrip.x * scale
            centerY = 500.0 - transformedGrip.y * scale
        } else {
            centerX = Double(width) * 0.5
            centerY = 160.0
        }
    } else if grip != nil {
        scale = 650.0
        centerX = Double(width) * 0.5
        centerY = 180.0
    } else {
        scale = min(
            Double(width - 150) / (maximumX - minimumX),
            Double(height - 150) / (maximumY - minimumY)
        )
        centerX = Double(width) * 0.5 -
            (minimumX + maximumX) * 0.5 * scale
        centerY = Double(height) * 0.50 -
            (minimumY + maximumY) * 0.5 * scale
    }
    func project(_ point: Vec3) -> Projected {
        let transformed = camera(point, yaw: yaw, pitch: pitch)
        return Projected(
            point: CGPoint(
                x: centerX + transformed.x * scale,
                y: centerY + transformed.y * scale
            ),
            depth: transformed.z
        )
    }

    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw NSError(domain: "NumiClothRenderer", code: 3)
    }
    context.setAllowsAntialiasing(true)
    context.setShouldAntialias(true)
    context.setFillColor(color(250, 249, 246))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    // The legacy plane and finite-bench room floor are both unbounded. The
    // declared bench itself is rendered below as its exact finite box.
    context.setFillColor(color(232, 225, 211))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    func groundPoint(screenX: Double, screenY: Double) -> Vec3 {
        let side = (screenX - centerX) / scale
        let forward = bench == nil ? -(screenY - centerY) / (scale * sin(pitch)) :
            (cos(pitch) * bench!.floorHeight - (screenY - centerY) / scale) / sin(pitch)
        return Vec3(
            x: cos(yaw) * side + sin(yaw) * forward,
            y: -sin(yaw) * side + cos(yaw) * forward,
            z: bench?.floorHeight ?? 0.0
        )
    }
    let visibleGround = [
        groundPoint(screenX: 0, screenY: 0),
        groundPoint(screenX: Double(width), screenY: 0),
        groundPoint(screenX: Double(width), screenY: Double(height)),
        groundPoint(screenX: 0, screenY: Double(height))
    ]
    let groundMinimumX = visibleGround.map(\.x).min()!
    let groundMaximumX = visibleGround.map(\.x).max()!
    let groundMinimumY = visibleGround.map(\.y).min()!
    let groundMaximumY = visibleGround.map(\.y).max()!
    // Quantize grid spacing and cap its density. With fixed trajectory framing
    // these bounds and the grid remain identical throughout the animation.
    let groundSpan = max(groundMaximumX - groundMinimumX,
                         groundMaximumY - groundMinimumY)
    let gridSpacing = 0.25 * pow(2.0, max(0.0, ceil(log2(groundSpan / 32.0))))
    let groundHeight = bench?.floorHeight ?? 0.0
    context.setStrokeColor(color(152, 130, 101, 0.18))
    context.setLineWidth(0.7)
    for line in Int(floor(groundMinimumX / gridSpacing))...Int(ceil(groundMaximumX / gridSpacing)) {
        let coordinate = Double(line) * gridSpacing
        context.move(to: project(Vec3(x: coordinate, y: groundMinimumY, z: groundHeight)).point)
        context.addLine(to: project(Vec3(x: coordinate, y: groundMaximumY, z: groundHeight)).point)
        context.strokePath()
    }
    for line in Int(floor(groundMinimumY / gridSpacing))...Int(ceil(groundMaximumY / gridSpacing)) {
        let coordinate = Double(line) * gridSpacing
        context.move(to: project(Vec3(x: groundMinimumX, y: coordinate, z: groundHeight)).point)
        context.addLine(to: project(Vec3(x: groundMaximumX, y: coordinate, z: groundHeight)).point)
        context.strokePath()
    }
    var primitives: [Primitive] = []
    primitives.reserveCapacity(6_900)
    for fruit in fruits {
        var shadowHeight = groundHeight
        if let bench, fruit.center.x >= bench.minimum.x, fruit.center.x <= bench.maximum.x,
           fruit.center.y >= bench.minimum.y, fruit.center.y <= bench.maximum.y,
           fruit.center.z >= bench.maximum.z + fruit.radius {
            shadowHeight = bench.maximum.z
        }
        let clearance = max(0, fruit.center.z - shadowHeight - fruit.radius)
        let shadowRadius = fruit.radius * 1.1 + 0.08 * clearance
        let shade = color(69, 52, 34, 0.24 / (1 + 4 * clearance))
        context.setFillColor(shade)
        var shadowPoints: [Projected] = []
        for sample in 0...40 {
            let angle = 2 * Double.pi * Double(sample) / 40
            let point = project(Vec3(
                x: fruit.center.x + shadowRadius * cos(angle),
                y: fruit.center.y + shadowRadius * sin(angle), z: shadowHeight
            ))
            if bench != nil { shadowPoints.append(point) }
            else if sample == 0 { context.move(to: point.point) }
            else { context.addLine(to: point.point) }
        }
        if bench != nil {
            primitives.append(Primitive(depth: shadowPoints.map(\.depth).reduce(0,+) /
                Double(shadowPoints.count) + 0.00001,
                kind: .shadow(shadowPoints.map(\.point), shade)))
        } else {
            context.closePath()
            context.fillPath()
        }
    }
    if let bench {
        let low = [bench.minimum.x, bench.minimum.y, bench.minimum.z]
        let high = [bench.maximum.x, bench.maximum.y, bench.maximum.z]
        // Subdivide only the presentation of the exact planar box faces so
        // painter depth ordering can occlude nearby cloth and fruit locally.
        for axis in 0..<3 {
            let first = (axis + 1) % 3, second = (axis + 2) % 3
            let firstCount = max(1, Int(ceil((high[first] - low[first]) / 0.08)))
            let secondCount = max(1, Int(ceil((high[second] - low[second]) / 0.08)))
            for side in 0..<2 {
                let shade = axis == 2 && side == 1 ? color(193, 160, 117) :
                    (axis == 2 ? color(95, 73, 48) : color(149, 112, 75))
                for i in 0..<firstCount { for j in 0..<secondCount {
                    var points: [Projected] = []
                    for (u, v) in [(i,j),(i+1,j),(i+1,j+1),(i,j+1)] {
                        var coordinate = low
                        coordinate[axis] = side == 0 ? low[axis] : high[axis]
                        coordinate[first] += (high[first] - low[first]) * Double(u) / Double(firstCount)
                        coordinate[second] += (high[second] - low[second]) * Double(v) / Double(secondCount)
                        points.append(project(Vec3(x: coordinate[0], y: coordinate[1], z: coordinate[2])))
                    }
                    primitives.append(Primitive(depth: points.map(\.depth).reduce(0,+) / 4,
                        kind: .bench(points.map(\.point), shade)))
                } }
            }
        }
    }
    for halfLevel in 0...(2 * (levels - 1)) {
        let level = Double(halfLevel) * 0.5
        let rim = level >= 0.72 * Double(levels - 1)
        for ring in 0..<around {
            let first = project(surface(
                vertices,
                level: level,
                ring: Double(ring)
            ))
            let second = project(surface(
                vertices,
                level: level,
                ring: Double(ring + 1)
            ))
            primitives.append(Primitive(
                depth: 0.5 * (first.depth + second.depth),
                kind: .yarn(first.point, second.point, rim)
            ))
        }
    }
    for halfRing in 0..<(2 * around) {
        let ring = Double(halfRing) * 0.5
        for level in 0..<(levels - 1) {
            let first = project(surface(
                vertices,
                level: Double(level),
                ring: ring
            ))
            let second = project(surface(
                vertices,
                level: Double(level + 1),
                ring: ring
            ))
            primitives.append(Primitive(
                depth: 0.5 * (first.depth + second.depth),
                kind: .yarn(
                    first.point,
                    second.point,
                    Double(level) >= 0.72 * Double(levels - 1)
                )
            ))
        }
    }
    for halfRow in 0...(2 * (bottomGrid - 1)) {
        let row = 0.5 * Double(halfRow)
        for column in 0..<(bottomGrid - 1) {
            let first = project(bottomSurface(
                vertices,
                row: row,
                column: Double(column)
            ))
            let second = project(bottomSurface(
                vertices,
                row: row,
                column: Double(column + 1)
            ))
            primitives.append(Primitive(
                depth: 0.5 * (first.depth + second.depth),
                kind: .yarn(first.point, second.point, false)
            ))
        }
    }
    for halfColumn in 0...(2 * (bottomGrid - 1)) {
        let column = 0.5 * Double(halfColumn)
        for row in 0..<(bottomGrid - 1) {
            let first = project(bottomSurface(
                vertices,
                row: Double(row),
                column: column
            ))
            let second = project(bottomSurface(
                vertices,
                row: Double(row + 1),
                column: column
            ))
            primitives.append(Primitive(
                depth: 0.5 * (first.depth + second.depth),
                kind: .yarn(first.point, second.point, false)
            ))
        }
    }
    for fruit in fruits {
        let projected = project(fruit.center)
        primitives.append(Primitive(
            depth: projected.depth,
            kind: .fruit(projected.point, fruit)
        ))
    }
    primitives.sort { $0.depth < $1.depth }

    let yarnOutline = color(118, 94, 66, 0.23)
    let yarnBody = color(229, 219, 196, 0.92)
    let yarnHighlight = color(255, 253, 241, 0.78)
    let fruitColors: [(CGColor, CGColor, CGColor)] = [
        (color(226, 136, 137), color(166, 39, 55), color(72, 15, 26)),
        (color(172, 204, 91), color(107, 155, 45), color(43, 77, 25)),
        (color(255, 216, 92), color(231, 170, 39), color(125, 78, 10)),
        (color(240, 132, 84), color(218, 75, 42), color(108, 29, 18)),
    ]
    context.setLineCap(.round)
    for primitive in primitives {
        switch primitive.kind {
        case let .shadow(points, shade):
            context.setFillColor(shade)
            context.move(to: points[0])
            for point in points.dropFirst() { context.addLine(to: point) }
            context.closePath()
            context.fillPath()
        case let .bench(points, shade):
            context.setFillColor(shade)
            context.move(to: points[0])
            for point in points.dropFirst() { context.addLine(to: point) }
            context.closePath()
            context.fillPath()
            context.setStrokeColor(shade)
            context.setLineWidth(0.5)
            context.move(to: points[0])
            for point in points.dropFirst() { context.addLine(to: point) }
            context.closePath()
            context.strokePath()
        case let .yarn(first, second, rim):
            let physicalDiameter = CGFloat(
                2.0 * clothRadiusMeters * scale
            )
            let bodyWidth = max(
                2.4,
                physicalDiameter * (rim ? 1.35 : 1.0)
            )
            let deltaX = second.x - first.x
            let deltaY = second.y - first.y
            let segmentLength = max(0.001, hypot(deltaX, deltaY))
            let normalX = -deltaY / segmentLength
            let normalY = deltaX / segmentLength
            context.setStrokeColor(yarnOutline)
            context.setLineWidth(bodyWidth + max(0.9, bodyWidth * 0.22))
            context.move(to: first)
            context.addLine(to: second)
            context.strokePath()
            context.setStrokeColor(yarnBody)
            context.setLineWidth(bodyWidth)
            context.move(to: first)
            context.addLine(to: second)
            context.strokePath()
            let shadowOffset = bodyWidth * 0.17
            context.setStrokeColor(color(133, 108, 76, 0.32))
            context.setLineWidth(max(0.55, bodyWidth * 0.18))
            context.move(to: CGPoint(
                x: first.x - normalX * shadowOffset,
                y: first.y - normalY * shadowOffset
            ))
            context.addLine(to: CGPoint(
                x: second.x - normalX * shadowOffset,
                y: second.y - normalY * shadowOffset
            ))
            context.strokePath()
            context.setStrokeColor(yarnHighlight)
            context.setLineWidth(max(0.7, bodyWidth * 0.18))
            context.move(to: CGPoint(
                x: first.x + normalX * shadowOffset,
                y: first.y + normalY * shadowOffset
            ))
            context.addLine(to: CGPoint(
                x: second.x + normalX * shadowOffset,
                y: second.y + normalY * shadowOffset
            ))
            context.strokePath()
            context.setStrokeColor(color(255, 252, 234, 0.82))
            context.setLineWidth(max(0.65, bodyWidth * 0.14))
            context.setLineDash(
                phase: rim ? 1.1 : 0.0,
                lengths: [
                    max(1.4, bodyWidth * 0.58),
                    max(1.2, bodyWidth * 0.46),
                ]
            )
            context.move(to: CGPoint(
                x: first.x + normalX * bodyWidth * 0.04,
                y: first.y + normalY * bodyWidth * 0.04
            ))
            context.addLine(to: CGPoint(
                x: second.x + normalX * bodyWidth * 0.04,
                y: second.y + normalY * bodyWidth * 0.04
            ))
            context.strokePath()
            context.setLineDash(phase: 0.0, lengths: [])
        case let .fruit(center, fruit):
            let radius = CGFloat(fruit.radius * scale)
            let palette = fruitColors[fruit.appearance % fruitColors.count]
            let colors = [palette.0, palette.1, palette.2] as CFArray
            let locations: [CGFloat] = [0.0, 0.38, 1.0]
            guard let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: colors,
                locations: locations
            ) else { continue }
            context.saveGState()
            context.addEllipse(in: CGRect(
                x: center.x - radius,
                y: center.y - radius,
                width: 2.0 * radius,
                height: 2.0 * radius
            ))
            context.clip()
            context.drawRadialGradient(
                gradient,
                startCenter: CGPoint(
                    x: center.x - 0.34 * radius,
                    y: center.y + 0.38 * radius
                ),
                startRadius: 0.06 * radius,
                endCenter: center,
                endRadius: radius,
                options: []
            )
            context.restoreGState()
            context.setStrokeColor(color(72, 38, 20, 0.30))
            context.setLineWidth(1.5)
            context.strokeEllipse(in: CGRect(
                x: center.x - radius,
                y: center.y - radius,
                width: 2.0 * radius,
                height: 2.0 * radius
            ))
            var bodyMarker = fruit.orientation.rotate(Vec3(
                x: 0.68,
                y: 0.31,
                z: 0.66
            ))
            if camera(bodyMarker, yaw: yaw, pitch: pitch).z < 0.0 {
                bodyMarker = bodyMarker * -1.0
            }
            let markerWorld = fruit.center + bodyMarker *
                (fruit.radius * 0.82)
            let markerCenter = project(markerWorld).point
            let markerRadius = max(2.2, radius * 0.075)
            context.setFillColor(color(67, 38, 24, 0.82))
            context.fillEllipse(in: CGRect(
                x: markerCenter.x - markerRadius,
                y: markerCenter.y - markerRadius,
                width: 2.0 * markerRadius,
                height: 2.0 * markerRadius
            ))
        }
    }

    if let grip, grip.active {
        let center = project(grip.center).point
        let xAxis = project(
            grip.center + grip.orientation.rotate(
                Vec3(x: 0.045, y: 0.0, z: 0.0)
            )
        ).point
        let yAxis = project(
            grip.center + grip.orientation.rotate(
                Vec3(x: 0.0, y: 0.045, z: 0.0)
            )
        ).point
        context.saveGState()
        context.setLineCap(.round)
        context.setStrokeColor(color(242, 145, 38, 0.72))
        context.setLineWidth(1.8)
        for level in (levels - 2)..<levels {
            for offset in -2...2 {
                let ring = (around + grip.patchCenterRing + offset) % around
                let seamPoint = project(
                    vertices[level * around + ring]
                ).point
                context.move(to: center)
                context.addLine(to: seamPoint)
                context.strokePath()
                let nodeRadius: CGFloat =
                    level == levels - 1 && offset == 0 ? 3.2 : 2.3
                context.setFillColor(color(255, 181, 64, 0.94))
                context.fillEllipse(in: CGRect(
                    x: seamPoint.x - nodeRadius,
                    y: seamPoint.y - nodeRadius,
                    width: 2.0 * nodeRadius,
                    height: 2.0 * nodeRadius
                ))
            }
        }
        context.setLineWidth(3.0)
        context.setStrokeColor(color(242, 91, 64, 0.95))
        context.move(to: center)
        context.addLine(to: xAxis)
        context.strokePath()
        context.setStrokeColor(color(76, 177, 218, 0.95))
        context.move(to: center)
        context.addLine(to: yAxis)
        context.strokePath()
        context.restoreGState()
        let marker = CGRect(x: center.x - 9.0, y: center.y - 9.0,
                            width: 18.0, height: 18.0)
        context.setFillColor(color(47, 45, 43, 0.92))
        context.fillEllipse(in: marker)
        context.setStrokeColor(color(255, 177, 65, 1.0))
        context.setLineWidth(4.0)
        context.strokeEllipse(in: marker.insetBy(dx: 2.0, dy: 2.0))
    }

    guard let image = context.makeImage() else {
        throw NSError(domain: "NumiClothRenderer", code: 4)
    }
    let representation = NSBitmapImageRep(cgImage: image)
    guard let data = representation.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "NumiClothRenderer", code: 5)
    }
    try data.write(to: URL(fileURLWithPath: output))
}

guard [3, 4, 6].contains(CommandLine.arguments.count) else {
    fputs(
        "usage: render_cloth_obj.swift INPUT.obj OUTPUT.png " +
        "[pickup|pickup-wide|trajectory --framing-list FILE]\n",
        stderr
    )
    exit(2)
}

let cameraProfile = CommandLine.arguments.count >= 4
    ? CommandLine.arguments[3]
    : "automatic"
guard cameraProfile == "automatic" || cameraProfile == "pickup" ||
      cameraProfile == "pickup-wide" || cameraProfile == "trajectory" else {
    fputs(
        "render_cloth_obj.swift: camera profile must be pickup or " +
        "pickup-wide or trajectory\n",
        stderr
    )
    exit(2)
}
guard (cameraProfile == "trajectory") == (CommandLine.arguments.count == 6),
      CommandLine.arguments.count != 6 || CommandLine.arguments[4] == "--framing-list" else {
    fputs("render_cloth_obj.swift: trajectory requires --framing-list FILE\n", stderr)
    exit(2)
}

do {
    let (vertices, fruits, grip, bench) = try parseOBJ(at: CommandLine.arguments[1])
    let framing = cameraProfile == "trajectory"
        ? try trajectoryBounds(listPath: CommandLine.arguments[5]) : nil
    try render(
        vertices: vertices,
        fruits: fruits,
        grip: grip,
        bench: bench,
        cameraProfile: cameraProfile,
        framing: framing,
        output: CommandLine.arguments[2]
    )
    print("rendered vertices=\(vertices.count) fruits=\(fruits.count) grip=\(grip != nil) camera=\(cameraProfile) output=\(CommandLine.arguments[2])")
} catch {
    fputs("render_cloth_obj.swift: \(error.localizedDescription)\n", stderr)
    exit(1)
}

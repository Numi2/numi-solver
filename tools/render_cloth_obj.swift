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

private enum PrimitiveKind {
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

private func parseOBJ(at path: String) throws -> ([Vec3], [Fruit], Grip?) {
    let source = try String(contentsOfFile: path, encoding: .utf8)
    var vertices: [Vec3] = []
    var fruits: [Fruit] = []
    var grip: Grip?
    for line in source.split(separator: "\n") {
        let fields = line.split(separator: " ")
        guard let first = fields.first else { continue }
        if first == "v", fields.count >= 4,
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
    return (vertices, fruits, grip)
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
    cameraProfile: String,
    output: String
) throws {
    let pickupCamera = cameraProfile == "pickup" ||
        cameraProfile == "pickup-wide"
    let width = pickupCamera ? 960 : (grip == nil ? 1200 : 800)
    let height = 800
    let yaw = -0.62
    let pitch = 0.19
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
    let scale: Double
    let centerX: Double
    let centerY: Double
    if cameraProfile == "pickup-wide" {
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

    let ground = project(Vec3(x: 0, y: 0, z: 0)).point
    context.saveGState()
    context.setFillColor(color(69, 52, 34, 0.12))
    context.fillEllipse(in: CGRect(
        x: ground.x - CGFloat(0.52 * scale),
        y: ground.y - CGFloat(0.065 * scale),
        width: CGFloat(1.04 * scale),
        height: CGFloat(0.13 * scale)
    ))
    context.restoreGState()

    var primitives: [Primitive] = []
    primitives.reserveCapacity(6_900)
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

guard CommandLine.arguments.count == 3 || CommandLine.arguments.count == 4 else {
    fputs(
        "usage: render_cloth_obj.swift INPUT.obj OUTPUT.png " +
        "[pickup|pickup-wide]\n",
        stderr
    )
    exit(2)
}

let cameraProfile = CommandLine.arguments.count == 4
    ? CommandLine.arguments[3]
    : "automatic"
guard cameraProfile == "automatic" || cameraProfile == "pickup" ||
      cameraProfile == "pickup-wide" else {
    fputs(
        "render_cloth_obj.swift: camera profile must be pickup or " +
        "pickup-wide\n",
        stderr
    )
    exit(2)
}

do {
    let (vertices, fruits, grip) = try parseOBJ(at: CommandLine.arguments[1])
    try render(
        vertices: vertices,
        fruits: fruits,
        grip: grip,
        cameraProfile: cameraProfile,
        output: CommandLine.arguments[2]
    )
    print("rendered vertices=\(vertices.count) fruits=\(fruits.count) grip=\(grip != nil) camera=\(cameraProfile) output=\(CommandLine.arguments[2])")
} catch {
    fputs("render_cloth_obj.swift: \(error.localizedDescription)\n", stderr)
    exit(1)
}

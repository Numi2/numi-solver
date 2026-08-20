#!/usr/bin/env swift

import Foundation
import ImageIO
import UniformTypeIdentifiers

guard CommandLine.arguments.count >= 4,
      let delay = Double(CommandLine.arguments[2]),
      delay > 0.0 else {
    fputs("usage: compose_cloth_gif.swift OUTPUT.gif DELAY_SECONDS FRAME.png...\n", stderr)
    exit(2)
}

let outputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let framePaths = Array(CommandLine.arguments.dropFirst(3))
guard let destination = CGImageDestinationCreateWithURL(
    outputURL as CFURL,
    UTType.gif.identifier as CFString,
    framePaths.count,
    nil
) else {
    fputs("compose_cloth_gif.swift: could not create GIF destination\n", stderr)
    exit(1)
}

let loopProperties = [
    kCGImagePropertyGIFDictionary: [
        kCGImagePropertyGIFLoopCount: 0,
    ],
] as CFDictionary
CGImageDestinationSetProperties(destination, loopProperties)

let frameProperties = [
    kCGImagePropertyGIFDictionary: [
        kCGImagePropertyGIFDelayTime: delay,
        kCGImagePropertyGIFUnclampedDelayTime: delay,
    ],
] as CFDictionary

for path in framePaths {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        fputs("compose_cloth_gif.swift: could not read frame \(path)\n", stderr)
        exit(1)
    }
    CGImageDestinationAddImage(destination, image, frameProperties)
}

guard CGImageDestinationFinalize(destination) else {
    fputs("compose_cloth_gif.swift: failed to finalize GIF\n", stderr)
    exit(1)
}

print("composed frames=\(framePaths.count) delay=\(delay) output=\(outputURL.path)")

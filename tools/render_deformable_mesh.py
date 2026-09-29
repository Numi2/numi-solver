#!/usr/bin/env python3
"""Rasterize exact native mesh-drop OBJ vertices with a fixed orthographic camera."""
import argparse
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def render(path, output, frame):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'v':
            vertices.append(tuple(map(float, fields[1:])))
        elif fields and fields[0] == 'f':
            faces.append(tuple(int(x) - 1 for x in fields[1:]))
    if len(vertices) != 13 or len(faces) != 20 or not all(math.isfinite(v) for p in vertices for v in p):
        raise ValueError('expected a finite thirteen-node twenty-face mesh')
    size, scale = 960, 2400
    image = Image.new('RGB', (size, 800), '#142b2b')
    draw = ImageDraw.Draw(image)
    def project(p):
        x, y, z = p
        return (480 + scale * (.86*x - .5*y), 652 + scale * (.24*x + .41*y - .866*z))
    floor = [project(p) for p in [(-.14,-.14,0),(.14,-.14,0),(.14,.14,0),(-.14,.14,0)]]
    draw.polygon(floor, fill='#244541')
    for i in range(-7, 8):
        t = i*.02
        draw.line([project((t,-.14,0)),project((t,.14,0))], fill='#355952', width=1)
        draw.line([project((-.14,t,0)),project((.14,t,0))], fill='#355952', width=1)
    # A projected native boundary silhouette provides a world-aligned shadow.
    for face in faces:
        draw.polygon([project((vertices[n][0],vertices[n][1],0)) for n in face], fill='#183630')
    camera = (.433,.75,.5)
    def depth(face):
        return sum(sum(vertices[n][k]*camera[k] for k in range(3)) for n in face)/3
    for face in sorted(faces, key=depth):
        points = [vertices[n] for n in face]
        a = tuple(points[1][k]-points[0][k] for k in range(3))
        b = tuple(points[2][k]-points[0][k] for k in range(3))
        normal = (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
        length = math.sqrt(sum(x*x for x in normal))
        illumination = max(0,sum(normal[k]*(-.3,.2,.93)[k] for k in range(3))/length)
        shade = .57+.43*illumination
        color = tuple(int(x*shade) for x in (246,169,82))
        polygon = [project(p) for p in points]
        draw.polygon(polygon, fill=color)
        draw.line(polygon+[polygon[0]], fill='#b48750', width=2)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    large = ImageFont.truetype(font_path,32)
    regular = ImageFont.truetype(font_path,20)
    small = ImageFont.truetype(font_path,17)
    draw.text((42,34), 'NATIVE ELASTIC MESH DROP', font=large, fill='#edf4eb')
    draw.text((42,80), 'Apple Metal  /  13 shared nodes  /  20 tetrahedra',font=regular,fill='#b9d1c4')
    draw.text((42,115), 'Exact solver boundary; authored material, frictionless inelastic plane',font=small,fill='#a7c4b7')
    draw.text((42,746), f't = {frame*.005:.3f} s     0.5 simulated seconds / 6.7x slow playback',font=regular,fill='#d7e8dc')
    draw.line((42,719,918,719),fill='#46695f',width=2)
    draw.line((42,719,42+876*frame/100,719),fill='#ebb979',width=4)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix',type=Path)
    parser.add_argument('output_directory',type=Path)
    args=parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=True)
    for frame in range(101):
        render(Path(str(args.prefix)+f'-{frame}.obj'),args.output_directory/f'frame-{frame:03}.png',frame)


if __name__ == '__main__':
    main()

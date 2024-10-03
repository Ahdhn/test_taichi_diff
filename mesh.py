#ref https://github.com/alecjacobson/libigl-autodiff-example/tree/main
import taichi as ti
import meshtaichi_patcher as Patcher
import argparse
import numpy as np

import igl

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="./beetle.obj")
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

mesh = Patcher.load_mesh(args.model, relations=['FV'])

# mesh.verts.place({'U': ti.math.vec3, 'normal': ti.math.vec3}, needs_grad=False)

mesh.verts.place({'U': ti.math.vec3}, needs_grad=True)
mesh.verts.U.from_numpy(mesh.get_position_as_numpy())

mesh.verts.place({'dfdU': ti.math.vec3}, needs_grad=True)
mesh.faces.place({'area': ti.f32}, needs_grad=True)

total_area = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def double_area():
    for f in mesh.faces:
        face_area = ti.f32(0.0)
        for x in range(3):
            y = (x+1) % 3
            rx = f.verts[0].U[x] - f.verts[2].U[x]
            sx = f.verts[1].U[x] - f.verts[2].U[x]
            ry = f.verts[0].U[y] - f.verts[2].U[y]
            sy = f.verts[1].U[y] - f.verts[2].U[y]
            face_area += rx*sy - ry*sx
        f.area = face_area
        ti.atomic_add(total_area[None], face_area)



total_area[None] = 0
total_area.grad[None] = 1

with ti.ad.Tape(total_area):
    double_area()          

#double_area()
#double_area().grad()

print(total_area)

grad_U = mesh.verts.U.grad.to_numpy()

print(grad_U[1:10])

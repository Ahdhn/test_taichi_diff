#https://docs.taichi-lang.org/docs/differentiable_programming
import taichi as ti
import taichi.math as tm

#ti.init(arch=ti.gpu, debug=True)
ti.init(arch=ti.gpu)



N = 8
dt = 1e-5

#particle positions 
x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)

#particle velocities 
v = ti.Vector.field(2, dtype=ti.f32, shape=N) 

#potential energy
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        U[None] += -1 / r.norm(1e-3)

@ti.kernel
def advance():
    for i in x:
         # dv/dt = -dU/dx
        v[i] += dt * -x.grad[i] 
    for i in x:
        # dx/dt = v
        x[i] += dt * v[i]   
        
def substep():
    with ti.ad.Tape(loss=U, validation=True):
        compute_U()
        advance()

@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]

init()
gui = ti.GUI('Autodiff gravity')
while gui.running:
    for i in range(50):
        substep()
    gui.circles(x.to_numpy(), radius=0.1)
    gui.show()
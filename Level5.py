import taichi as ti    
import numpy as np
import time
from taichi.ui.gui import rgb_to_hex


ti.init(arch=ti.gpu)

## determine whether the player succeed passing the game
intersect_ratio=0.5
n_particles_base=9000

quality = 2  # Use a larger value for higher-res simulations                                                             
n_particles, n_grid = n_particles_base * quality**2, 128 * quality                                                                   
dx, inv_dx = 1 / n_grid, float(n_grid)                                                                                   
dt = 1e-4 / quality
p_vol,p_rho = (dx * 0.5)**2,1
p_mass = p_vol * p_rho                                                                                                   


x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position                                                       
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity                                                       
C = ti.Matrix.field(2, 2, dtype=float,shape=n_particles)  # affine velocity field                                                          
F = ti.Matrix.field(2, 2, dtype=float,shape=n_particles)  # deformation gradient                                                           
material = ti.field(dtype=int, shape=n_particles)  # material id                           
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation                                                     
grid_v = ti.Vector.field(2, dtype=float,shape=(n_grid, n_grid))  # grid node momentum/velocity                                          
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass                                                 


attractor_strength = ti.field(dtype=float, shape=())


attractor_pos_np= np.array([[0.5,0.5],[0.7,0.4]]).astype(np.float32)
attractor_pos = ti.Vector.field(2, dtype=float, shape=(attractor_pos_np.shape[0]))
attractor_pos.from_numpy(attractor_pos_np)
attractor_on = ti.field(dtype=float, shape=(attractor_pos_np.shape[0]))
attractor_colors = [0xFC2604,0xFCFC04,0x17FC04,]



drag_damping = ti.field(dtype=ti.f32, shape=())



E_np = np.array([5e3,5e3,5e3]),   # Young's modulus and Poisson's ratio
nu_np = np.array([0.2,0.3,0.2])
mu_0_np = (E_np / (2 * (1 + nu_np))).flatten()
lambda_0_np = (E_np * nu_np / ((1 + nu_np) * (1 - 2 * nu_np))).flatten()  # Lame parameters 
mu_0_0,mu_0_1,mu_0_2 = mu_0_np
lambda_0_0,lambda_0_1,lambda_0_2 = lambda_0_np
colors = [0x068587, 0xED553B, 0xEEEEF0]

#set Target Shape------------------------------------------------------------

target_polys_np_list = [
    np.array([[0.2,0.4],[0.5,0.1],[0.8,0.4],[0.5,0.7]]).astype(np.float32),
    ]
target_bounds_material = [2]
target_polys_vec_list,target_bound_xs,target_bound_ys = [],[],[]
for i in range(len(target_polys_np_list)):
    target_poly_temp = ti.Vector.field(2, dtype=float,shape=(target_polys_np_list[i].shape[0],))
    target_poly_temp.from_numpy(target_polys_np_list[i])
    target_polys_vec_list.append(target_poly_temp)
    target_bound_xs.append(target_polys_np_list[i])
    target_bound_ys.append(np.concatenate([target_polys_np_list[i][1:],target_polys_np_list[i][0].reshape(1,-1)]))

target_bound_width = 2
is_in = ti.field(dtype=ti.int32,shape=x.shape[0])

wall_polys_np_list = [
    np.array([[0.1,0.9],[0.2,0.9],[0.3,0.7],[0.1,0.6]]).astype(np.float32),
] 
wall_polys_vec_list,wall_bound_xs,wall_bound_ys = [],[],[]
for i in range(len(wall_polys_np_list)):
    wall_poly_temp = ti.Vector.field(2, dtype=float,shape=(wall_polys_np_list[i].shape[0],))
    wall_poly_temp.from_numpy(wall_polys_np_list[i])
    wall_polys_vec_list.append(wall_poly_temp)
    wall_bound_xs.append(wall_polys_np_list[i])
    wall_bound_ys.append(np.concatenate([wall_polys_np_list[i][1:],wall_polys_np_list[i][0].reshape(1,-1)]))
wall_bound_width = 2
@ti.kernel

def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3

        mu_0,lambda_0 = mu_0_0,lambda_0_0
        if material[p] == 1:
            mu_0,lambda_0 = mu_0_1,lambda_0_1
        elif material[p] == 2:
            mu_0,lambda_0 = mu_0_2,lambda_0_2
            
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            # grid_v[i, j] += dt * gravity[None] * 30  # gravity
            for attr_idx in range(attractor_pos.shape[0]):
                dist = attractor_pos[attr_idx] - dx * ti.Vector([i, j])
                grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100 * (attractor_on[attr_idx] % 3 - 1)
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
            if is_pt_in_poly(pt=(i/n_grid,j/n_grid),poly=wall_polys_vec_list[0]) == 1:
                grid_v[i, j][0] = 0
                grid_v[i, j][1] = 0


    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        v[p] *= ti.exp(-dt * drag_damping[None])
        x[p] += dt * v[p]  # advection  
                                                                                                                                                                                                                                                                                                
@ti.kernel                                                                                                             
def reset():                                                                                                             
    group_size = n_particles // 3                                                                                        
    for i in range(n_particles):  
        ## initial positions-------------------------------------------------
        group_n=i // group_size
        if group_n==0:
            x[i] = [                                                                                                         
            ti.random() * 0.98 + 0.01,                                                          
            ti.random() * 0.05+0.01                                                          
        ]  
        elif group_n==1:
            x[i] = [                                                                                                         
            ti.random() * 0.98+0.01,                                                          
            ti.random() * 0.05+0.94                                                         
        ]  
        elif group_n==2:
            x[i] = [                                                                                                         
            ti.random() * 0.05,                                                          
            ti.random() * 0.9+0.05                                                          
        ]  
        else:
            x[i] = [                                                                                                         
            ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),                                                          
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)                                                          
        ]  
        ## initial positions-------------------------------------------------                                                                                                              
        # material[i] = i // group_size  # 0: fluid 1: jelly 2: snow     
        material[i] = 2

        v[i] = [0, 0]                                                                                                    
        F[i] = ti.Matrix([[1, 0], [0, 1]])                                                                               
        Jp[i] = 1                                                                                                        
        C[i] = ti.Matrix.zero(float, 2, 2) 
                                                                                     
@ti.func
def is_pt_in_poly(pt,poly):
    nvert = poly.shape[0]
    j = nvert - 1
    res = -1
    for i in range(poly.shape[0]): 
        if (poly[j][1] - poly[i][0]) == 0:
            j = i
            continue
        xx = (poly[j][0] - poly[i][0]) * \
            (pt[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + \
            poly[i][0]
        if ((poly[i][1] > pt[1]) != (poly[j][1] > pt[1])) and (pt[0] < xx):
            res = - res 
        j = i 
    return res

@ti.kernel
def update_isin():
    for pt in x:
        in_target_bound_0 = is_pt_in_poly(x[pt],target_polys_vec_list[0])
        # in_target_bound_1 = is_pt_in_poly(x[pt],target_polys_vec_list[1])
        is_in[pt] = 0
        if in_target_bound_0 == 1 and target_bounds_material[0] == material[pt]:
            is_in[pt] = 1
        # elif in_target_bound_1 == 1 and target_bounds_material[1] == material[pt]:
        #     is_in[pt] = 1
def is_click(mouse_pos,aim_pos,r=15/720):
    return ((mouse_pos - aim_pos)** 2).sum() < r**2
def level5_main():   
    gui = ti.GUI("Level1", res=720, background_color=0x112F41)    
    # Show the score and time -----------------------------------------------------
    score = gui.label('Score')
    time_record = gui.label('Time(s)')   
    attract_scale = gui.slider('attaction_scale', 0, 100, step=5)
    damping_scale = gui.slider('damping scale', 0, 100, step=5)    
    score.value=0
    time_record.value=0
    start_time=time.time()
    X_border,Y_border=0.605,0.835
    reset()
    win_flag = False 
    
                                       
    frame=0
    while gui.running:   
        ## Target Shape 
        for taregt_bound_x,taregt_bound_y,material_id in zip(target_bound_xs,target_bound_ys,target_bounds_material):
            gui.lines(begin=taregt_bound_x, end=taregt_bound_y, radius=target_bound_width, color=colors[material_id])   
        for wall_bound_x,wall_bound_y in zip(wall_bound_xs,wall_bound_ys):
            gui.lines(begin=wall_bound_x, end=wall_bound_y, radius=wall_bound_width, color=0xFFFFFF)
        for attr_idx in range(attractor_pos_np.shape[0]):            
            gui.circle(attractor_pos_np[attr_idx], color=attractor_colors[int(attractor_on[attr_idx] % 3)], radius=15) 
        
        
        if gui.get_event(ti.GUI.PRESS):                                                                                      
            if gui.event.key == 'r':                                                                                         
                reset()    
                start_time=time.time()
                score.value=0    
                frame=0                                                                                     
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:                                                              
                break                                                                                                        
                                                                                              
        mouse = gui.get_cursor_pos()                                                                                         
        gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)                                                          
        # attractor_pos[None] = [mouse[0], mouse[1]] 
                                                                       
        attractor_strength[None] = (14/(np.e-1)*np.exp(attract_scale.value/100) + 2-14/(np.e-1))                                                                             
        if gui.is_pressed(ti.GUI.LMB):
            for attr_idx in range(attractor_pos_np.shape[0]): 
                if is_click(mouse,attractor_pos_np[attr_idx]):
                    attractor_on[attr_idx] += 1                                                                      
        if gui.is_pressed(ti.GUI.RMB): 
            for attr_idx in range(attractor_pos_np.shape[0]): 
                if is_click(mouse,attractor_pos_np[attr_idx]):
                    attractor_on[attr_idx] -= 1      
        drag_damping[None] = damping_scale.value                                                                          
        for s in range(int(2e-3 // dt)):                                                                                     
             substep()                                                                                                       
        gui.circles(x.to_numpy(),                                                                                            
                    radius=1.5,                                                                                              
                    palette=colors,                                                                  
                    palette_indices=material) 

        # update time--------------------------------------------------------------
        end_time = time.time()
        current_time = end_time-start_time
        time_record.value = current_time
        # update time--------------------------------------------------------------
        update_isin()
        score.value = is_in.to_numpy().sum() / is_in.to_numpy().shape[0] / intersect_ratio
        if score.value >= 1.0:
            win_flag = True
            gui.text("Congratulations!",pos=np.array([0.14,0.5]),font_size=60,color=rgb_to_hex([100,100,100]))  
            gui.text("You pass the game!",pos=np.array([0.18,0.35]),font_size=45,color=rgb_to_hex([100,100,100])) 
        frame+=1
        gui.show()
    return True

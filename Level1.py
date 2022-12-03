import taichi as ti    
import numpy as np
import time
from taichi.ui.gui import rgb_to_hex
##attract stength value setting
attract_Strength_Value=0.25
## determine whether the player succeed passing the game
intersectRatio=0.667
PointsNum=9000

quality = 1  # Use a larger value for higher-res simulations                                                             
n_particles, n_grid = PointsNum * quality**2, 128 * quality                                                                   
dx, inv_dx = 1 / n_grid, float(n_grid)                                                                                   
dt = 1e-4 / quality                                                                                                      
p_vol,p_rho = (dx * 0.5)**2,1                                                                                                                                                                                                                           
ti.init(arch=ti.gpu)  # Try to run on GPU                                                                                                                                                                                                         quality = 1  # Use a larger value for higher-res simulations                                                             n_particles, n_grid = 9000 * quality**2, 128 * quality                                                                   dx, inv_dx = 1 / n_grid, float(n_grid)                                                                                   dt = 1e-4 / quality                                                                                                      p_vol, p_rho = (dx * 0.5)**2, 1                                                                                          
p_mass = p_vol * p_rho                                                                                                   
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio                                                                  
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters                                                                          
                                                                                                                         
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position                                                       
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity                                                       
C = ti.Matrix.field(2, 2, dtype=float,shape=n_particles)  # affine velocity field                                                          
F = ti.Matrix.field(2, 2, dtype=float,shape=n_particles)  # deformation gradient                                                           
material = ti.field(dtype=int, shape=n_particles)  # material id                                                         
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation                                                     
grid_v = ti.Vector.field(2, dtype=float,shape=(n_grid, n_grid))  # grid node momentum/velocity                                          
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass                                                 
# gravity = ti.Vector.field(2, dtype=float, shape=())                                                                      
attractor_strength = ti.field(dtype=float, shape=())                                                                     
attractor_pos = ti.Vector.field(2, dtype=float, shape=())  

#set Target Shape------------------------------------------------------------
Point1=[0.5,0.2]
Point2=[0.75,0.4]
Point3=[0.5,0.8]
Point4=[0.3,0.4]
targetLineColor = 0x068587
linesWidth=2
X = np.array([Point1,Point2,Point3,Point4])
Y = np.array([Point2,Point3,Point4,Point1])
#set Target Shape------------------------------------------------------------
#detect if inside the Target Shape-------------------------------------------
a1,a2=(Point2[0] - Point1[0]),(Point2[1] - Point1[1])
b1,b2=(Point3[0] - Point2[0]),(Point3[1] - Point2[1])
c1,c2=(Point4[0] - Point3[0]),(Point4[1] - Point3[1])
d1,d2=(Point1[0] - Point4[0]),(Point1[1] - Point4[1])

#detect if inside the Target Shape-------------------------------------------
#Set the Threshodl-----------------------------------------------------------
RatioLst=np.zeros(20)
PointsThreshold=PointsNum*intersectRatio
#Set the Threshodl-----------------------------------------------------------
 


@ti.kernel                                                  
def substep():    
    #grid_m,grid_v,x,inv_dx,dt,dx,material,lambda_0,mu_0,gravity,\
    #    attractor_pos,attractor_strength,n_grid,v,C,F,Jp,p_vol,p_mass                                                                                                       
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
            dist = attractor_pos[None] - dx * ti.Vector([i, j])                                                          
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100                                        
            if i < 3 and grid_v[i, j][0] < 0:                                                                            
                grid_v[i, j][0] = 0  # Boundary conditions                                                               
            if i > n_grid - 3 and grid_v[i, j][0] > 0:                                                                   
                grid_v[i, j][0] = 0                                                                                      
            if j < 3 and grid_v[i, j][1] < 0:                                                                            
                grid_v[i, j][1] = 0                                                                                      
            if j > n_grid - 3 and grid_v[i, j][1] > 0:                                                                   
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
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow                                                       
        v[i] = [0, 0]                                                                                                    
        F[i] = ti.Matrix([[1, 0], [0, 1]])                                                                               
        Jp[i] = 1                                                                                                        
        C[i] = ti.Matrix.zero(float, 2, 2) 
                                                                                     
def isPointInRect(x, y):
    a = a1*(y - Point1[1]) - a2*(x - Point1[0])
    b = b1*(y - Point2[1]) - b2*(x - Point2[0])
    c = c1*(y - Point3[1]) - c2*(x - Point3[0])
    d = d1*(y - Point4[1]) - d2*(x - Point4[0])
    if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    return False
 
def CalculateIntersect(x):
    pointsSum=0
    for point in x:
        [p1,p2]=point
        if isPointInRect(p1,p2):
            pointsSum+=1
    if pointsSum>PointsThreshold:
        return True
    return False


def level1_main():   
    gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)    
    #Show the score and time -----------------------------------------------------
    score = gui.label('Score')
    timeRecord = gui.label('Time(s)')   
    Gravity_Scale = gui.slider('Gravity_Scale', 10, 100, step=5)    
    score.value=0
    timeRecord.value=0
    startTime=time.time()
    X_border,Y_border=0.605,0.835
    reset()  #n_particles,x,material,v,F,Jp,C                                                                                                             
    #Show the score and time -----------------------------------------------------
                                              
    currentCheck,lastCheck=False,False   
    frame=0
    while gui.running:   
        ##Target Shape------------------------------------------------------------
        gui.lines(begin=X, end=Y, radius=linesWidth, color=targetLineColor)    
        ##Target Shape------------------------------------------------------------                                                                                          
        if gui.get_event(ti.GUI.PRESS):                                                                                      
            if gui.event.key == 'r':                                                                                         
                reset()    
                startTime=time.time()
                score.value=0    
                frame=0                                                                                     
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:                                                              
                break                                                                                                        
                                                                                              
        mouse = gui.get_cursor_pos()                                                                                         
        gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)                                                          
        attractor_pos[None] = [mouse[0], mouse[1]] 
        # print([mouse[0], mouse[1]] )                                                                          
        attractor_strength[None]=0                                                                               
        if gui.is_pressed(ti.GUI.LMB):   
            if mouse[0]<X_border or mouse[1]<Y_border:                                                                                    
                attractor_strength[None] = Gravity_Scale.value/200                                                                                     
        if gui.is_pressed(ti.GUI.RMB): 
            if mouse[0]<X_border or mouse[1]<Y_border:                                                                                         
                attractor_strength[None] = -Gravity_Scale.value/200                                                                               
        for s in range(int(2e-3 // dt)):                                                                                     
             substep()                                                                                                       
        gui.circles(x.to_numpy(),                                                                                            
                    radius=1.5,                                                                                              
                    palette=[0x068587, 0xED553B, 0xEEEEF0],                                                                  
                    palette_indices=material) 

        # update time--------------------------------------------------------------
        endTime=time.time()
        currentTime=endTime-startTime
        timeRecord.value=currentTime
        # update time--------------------------------------------------------------

        if frame%40==0:
            if CalculateIntersect(x.to_numpy()):
                currentCheck=True
            else:
                currentCheck=False
        if frame%40==20:
            lastCheck=currentCheck

        if lastCheck and currentCheck:
            gui.text("Congratulations!",pos=np.array([0.14,0.5]),font_size=60,color=rgb_to_hex([100,100,100]))  
            gui.text("You pass the game!",pos=np.array([0.18,0.35]),font_size=45,color=rgb_to_hex([100,100,100]))  
        frame+=1
        gui.show()
    return True

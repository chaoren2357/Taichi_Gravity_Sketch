import taichi as ti
import numpy as np
from taichi.ui.gui import rgb_to_hex
from Level1 import level1_main
from Level2 import level2_main
from Level3 import level3_main
from Level4 import level4_main
gui = ti.GUI('Gravity Sketch',(720, 720),background_color=0x112F41)
level1 = gui.button('Level 1')
level2 = gui.button('Level 2')
level3 = gui.button('Level 3')
level4 = gui.button('Level 4')

while gui.running:
    gui.text("Welcome to Gravity Sketch!",pos=np.array([0.14,0.6]),font_size=60,color=rgb_to_hex([100,100,100]))  
    for e in gui.get_events(gui.PRESS):
        if e.key == level1:
            level1_main()
        elif e.key==level2:
            level2_main()
        elif e.key==level3:
            level3_main()
        elif e.key==level4:
            level4_main()
    gui.show()

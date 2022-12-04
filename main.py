import taichi as ti
import numpy as np
from taichi.ui.gui import rgb_to_hex

gui = ti.GUI('Gravity Sketch',(800, 480),background_color=0x112F41)
level1 = gui.button('Level 1')
level2 = gui.button('Level 2')
level3 = gui.button('Level 3')
level4 = gui.button('Level 4')
level5 = gui.button('Level 5')
while gui.running:
    gui.text("Welcome to Gravity Sketch!",pos=np.array([0.1,0.5]),font_size=60,color=0xFFFFF)  
    for e in gui.get_events(gui.PRESS):
        if e.key == level1:
            from Level1 import level1_main
            level1_main()
        elif e.key==level2:
            from Level2 import level2_main
            level2_main()
        elif e.key==level3:
            from Level3 import level3_main
            level3_main()
        elif e.key==level4:
            from Level4 import level4_main
            level4_main()
        elif e.key==level5:
            from Level5 import level5_main
            level5_main()
    gui.show()

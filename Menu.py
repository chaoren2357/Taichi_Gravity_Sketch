import taichi as ti
import numpy as np
from taichi.ui.gui import rgb_to_hex
from Level1 import level1_main
from Level2 import main
gui = ti.GUI('Gravity Sketch',(640, 480))
level1 = gui.button('Level 1')
level2 = gui.button('Level 2')
level3 = gui.button('Level 3')
level4 = gui.button('Level 4')

while gui.running:
    gui.text("Welcome to",pos=np.array([0.14,0.5]),font_size=60,color=rgb_to_hex([100,100,100]))  
    gui.text("Gravity Sketch!",pos=np.array([0.14,0.38]),font_size=60,color=rgb_to_hex([100,100,100]))
    for e in gui.get_events(gui.PRESS):
        if e.key == level1:
            is_return = level1_main()
        elif e.key==level2:
            main()
        elif e.key==level3:
            print('Level3 clicked')
        elif e.key==level4:
            print('Level4 clicked')
    gui.show()

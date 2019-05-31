# from plotteraxi import PlotterAxi

# p = PlotterAxi()

# p.up()
# p.sprint(-10, -10)
# p.down()

# p.move(0, 0)
# p.move_rel(5, 5)
# p.move_rel(5, 0)

# p.runfile('~/output_0002.ngc')



import axi
from axi import Device

# p = Plotter()
d = Device()

d.enable_motors()

d.pen_up()
d.move(5, 5)
d.pen_down()
d.move(6, 6)
d.move_rel(2, 2)
d.move_rel(2, 0)
d.pen_up()

d.goto(0, 0)

d.disable_motors()



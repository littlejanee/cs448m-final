from plotter import Plotter

p = Plotter()

p.up()
p.sprint(-10, -10)
p.down()

p.move(0, 0)
p.move_rel(5, 5)
p.move_rel(5, 0)

p.runfile('~/output_0002.ngc')



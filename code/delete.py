import model as m3
import random
random.seed(42)

route1 = [1,3,2,6,4,5,9,7,8]
route2 = [3,7,8,1,4,9,2,5,6]
m3.Sol.set_stations([1])
sol1 = m3.Sol(route1)
sol2 = m3.Sol(route2)
sol1.reproduce(sol2)
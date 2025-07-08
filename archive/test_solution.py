from gurobipy import *

model = Model("dual")
model.modelSense = GRB.MAXIMIZE

K = [0,1]

a = [model.addVar(obj=0, lb=0, name='a'+str(k)) for k in K]
u = [model.addVar(obj=1, lb=0, name='u'+str(i)) for i in range(2)]
w = model.addVar(obj=-1, lb=0, name='w')

model.update()

model.addConstr(a - u[0] <= 0)
model.addConstr(u[0] - u[1] <= 0)
model.addConstr(u[1] - w <= 0)

model.update()
model.optimize()
model.computeIIS()
#model.write('ts.ilp')
# Was a wrapper for CPLEX and GUROBI, but I've removed CPLEX
# This entire class will be removed at some point
from gurobipy import Env, GRB, Model

##
## Abstraction for gurobi
##
class Solver(object):
    """Abstract layer for Gurobi"""
    __slots__ = ['model']

    def __init__(self, minimize=True, quiet=True, use_callback=True, env=None):
        self.model = Model("model_name", env=env if env is not None else Env(""))
        self.model.modelSense = GRB.MINIMIZE if minimize else GRB.MAXIMIZE
        self.model.setParam('OutputFlag', not quiet)

    def set_gap(self, gap):
        self.model.setParam(GRB.param.MIPGap, gap)

    def set_timelimit(self, timelimit):
        self.model.setParam(GRB.param.TimeLimit, timelimit)

    def set_threads(self, val):
        self.model.setParam(GRB.param.Threads, val)

    def set_aggressive_cuts(self):
        self.model.setParam(GRB.param.MIPFocus, 2)
        self.model.setParam(GRB.param.PrePasses, 3)

    # update variables (batch mode for gurobi & cplex)
    def update(self):
        self.model.update()

    def write(self, file):
        self.model.write(file)

    def optimize(self, callback=None):
        if callback:
            def opt(model, where):
                if where == GRB.callback.MIP:
                    if not callback(model.cbGet(GRB.callback.MIP_OBJBST), model.cbGet(GRB.callback.MIP_OBJBND)):
                        self.model.terminate()

            self.model.optimize(opt)
        else:
            self.model.optimize()

    def presolve(self):
        return self.model.presolve()

    def is_optimal(self):
        return self.model.status == GRB.status.OPTIMAL

    def is_abort(self):
        return self.model.status in [GRB.status.TIME_LIMIT, GRB.status.INTERRUPTED]


    def objVal(self) -> float:
        return self.model.objVal

    def objBound(self) -> float:
        return self.model.objBound

    def val(self, var):
        return var.x

    def vals(self, vars):
        return [x.x for x in vars] #self.model.cbGetSolution(vars)


    #
    # add variable & useful constants
    #
    def inf(self):
        return GRB.INFINITY
    def integer(self):
        return GRB.INTEGER
    def binary(self):
        return GRB.BINARY
    def continuous(self):
        return GRB.CONTINUOUS

    def addVar(self, obj, lb, ub, type=None, name = None):
        return self.model.addVar(obj=obj, lb=lb, ub=ub, vtype=(type or GRB.CONTINUOUS))#, name=name[:255]) 

    def removeVar(self, var):
        self.model.remove(var)

    def getVars(self):
        return self.model.getVars()

    @property
    def NumVars(self):
        return self.model.NumVars

    @property
    def PresolveNumVars(self):
        return len(self.model.presolve().getVars())

    @property
    def NumConstrs(self):
        return self.model.NumConstrs

    @property
    def PresolveNumConstrs(self):
        return len(self.model.presolve().getConstrs())

    #
    # add constraints
    #
    def addConstr(self, cons, name=None):
        return self.model.addConstr(cons)#,name[:255])
    
    def addConstrs(self, generator):
        return self.model.addConstrs(generator)

    def chgCoeff(self, cons, var, val):
        self.model.chgCoeff(cons, var, val)

    def getConstrs(self):
        return self.model.getConstrs()

    def removeCons(self, cons):
        self.model.remove(cons)

    def set_rhs(self, cons, rhs):
        cons.setAttr(GRB.Attr.RHS, rhs) 


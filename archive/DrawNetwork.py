from math import *
from gurobipy import *
import networkx as nx
import subprocess
import pydot
import os

##
## Draws network/time window (similar to DrawLaTeX - but without any external class references)
##
class DrawNetwork(object):
    """Outputs figures in LaTeX and pdf"""
    __slots__ = ['network', 'commodities', 'figures', 'latex']

    def __init__(self, filename):
        self.read_file(filename)

        self.figures = []
        self.latex = ''

    # loads network and commodities from file
    def read_file(self, filename):
        self.commodities = []
        tmp_network = {}

        with open(filename, "r") as file:
            while not file.readline().startswith("ARCS"):
                pass

            line = file.readline() # skip header
            line = file.readline()

            while not line.startswith("COMMODITIES"):
                tmp = line.split(',')

                if int(tmp[1]) not in tmp_network:
                    tmp_network[int(tmp[1])] = {}

                tmp_network[int(tmp[1])][int(tmp[2])] = float(tmp[6])
                line = file.readline()

            line = file.readline() # skip header
            line = file.readline()

            while len(line) > 0:
                tmp = line.split(',')
                self.commodities.append({'a': (int(tmp[1]), float(tmp[4])), 'b': (int(tmp[2]), float(tmp[5])), 'q': float(tmp[3])})
                line = file.readline()

        # build graph
        self.network = nx.DiGraph()

        for a, destinations in tmp_network.items():
            for b, transit_time in destinations.items():
                self.network.add_edge(a, b, weight=transit_time, cost=transit_time)


    # draws problem data
    def save(self, filename, clean=True):
        with open(filename + '.tex', "w") as text_file:
            text_file.write(self.latex)

        self.latex = ''

        proc = subprocess.Popen(['pdflatex', filename + '.tex', '-quiet'])
        proc.communicate()

        if clean:
            os.remove(filename + '.tex')
            os.remove(filename + '.aux')
            os.remove(filename + '.log')


    ##
    ## Defines temporary latex document for drawing figures
    ##
    def generate_latex(self, portrait=False, columns=1):
        output = []
        output.append('\\documentclass{article}\n' +
                      '\\usepackage{tikz,amsmath, amssymb,bm,color}\n' +
                      #'\\usepackage[margin=0cm,nohead]{geometry}\n' +
                      '\\usepackage[active,tightpage]{preview}\n' +
                      '\\usetikzlibrary{shapes,arrows}\n' +
                      '\\usetikzlibrary{calc}\n' +
                      '\\usepackage{pdflscape}\n' +
                      '\\usepackage[left=1cm,top=0.5cm,right=1cm,bottom=0.5cm,bindingoffset=0.5cm]{geometry}\n' +
                     #'\\PreviewEnvironment{tikzpicture}\n' +
                      '\\usepackage{graphicx}\n' + 
                      '\\usepackage{caption}\n' + 
                      '\\usepackage{subcaption}\n' +
                      '\\usepackage{booktabs}\n' + 
                      '\\begin{document}\n' +
                      ('\\begin{landscape}\n' if portrait == False else ''))
       
        for i, f in enumerate(self.figures):
            if i % columns == 0:
                output.append('\\begin{figure}[H]\n'+                  
                              '\\centering\n')

            output.append('{0}'.format(f))
                          
            if (i + 1) % columns == 0:
                output.append('\\end{figure}\n\\vspace{2mm}\n')

                if (i + 1) % 16 == 0: # avoid errors for having too many figures
                    output.append('\\clearpage\n')

        if (i + 1) % columns != 0:
            output.append('\\end{figure}\n\\vspace{2mm}\n')

        output.append(('\\end{landscape}\n' if portrait == False else '') + '\\end{document}\n')
        self.latex = ''.join(output)
        self.figures = []

    ##
    ## Draw table
    ##
    def draw_table(self, header, data):
        output = ['\\begin{{tabular}}{{{0}}}\n'.format('r' * len(header)) + 
                  '\\toprule\n']

        for i,h in enumerate(header):
            output.append("\\textbf{{{0}}} {1} ".format(h, '&' if i + 1 < len(header) else '\\\\\n'))

        output.append('\\midrule\n')

        for d in data:
            for i,h in enumerate(header):
                output.append("{0} {1} ".format(d[h], '&' if i + 1 < len(header) else '\\\\\n'))

        output.append('\\bottomrule\n\\end{tabular}\n')

        self.figures.append(''.join(output))

    ##
    ## Draw Network figure
    ##
    def draw_network(self, scale=2):
        network = self.network
        position = nx.pygraphviz_layout(network, prog='neato')

        output = ['\\begin{subfigure}[b]{10cm}\n' + 
                  '\\centering\n' +
                  '\\begin{{tikzpicture}} [>=latex\',line join=bevel,thick,scale={0}]\n'.format(scale) +
                  '\\tikzstyle{n} = [shape=circle, thick, draw=black!55, align=center]\n']
    
        for n in network.nodes():
            output.append('\\draw ({0}bp,{1}bp) node [n] (n{2}) {{{2}}};\n'.format(position[n][0], position[n][1], n))

        edges = network.edges()

        for a,b in edges:
            if (b,a) in edges:
                if a > b: # simplify edges
                    output.append('\\draw [<->] (n{0}) edge node [above] {{{2}}} (n{1});\n'.format(a, b, network[a][b]['weight']))
            else:
                output.append('\\draw [-latex, color=orange] (n{0}) edge node [above] {{{2}}} (n{1});\n'.format(a, b, network[a][b]['weight']))

        output.append('\\end{tikzpicture}\n\\caption*{Network}\n\\end{subfigure}\n')
        self.figures.append(''.join(output))

    ##
    ## Draw Commodities
    ##
    def draw_commodities(self, scale=2):
        network = self.network
        position = nx.pygraphviz_layout(network, prog='neato')

        output = ['\\begin{subfigure}[b]{10cm}\n' + 
                  '\\centering\n' +
                  '\\begin{{tikzpicture}} [>=latex\',line join=bevel,thick,scale={0}]\n'.format(scale) +
                  '\\tikzstyle{n} = [shape=circle, thick, draw=black!55, align=center]\n']
    
        for n in network.nodes():
            output.append('\\draw ({0}bp,{1}bp) node [n] (n{2}) {{{2}}};\n'.format(position[n][0], position[n][1], n))

        processed = []

        for k,c in enumerate(self.commodities):
            a,b = (c['a'][0], c['b'][0])
            output.append('\\draw [-latex,dashed] (n{0}) edge {3} node [pos=0.2,above,align=center,font={{\\tiny\\bfseries}},color=blue] {{{2}}} (n{1});\n'.format(a, b, "${0}$\\\\$[{1},{2}]$".format(c['q'], c['a'][1], c['b'][1]), '[bend right]' if (a,b) not in processed else ''))
            processed.append((a,b))

        output.append('\\end{tikzpicture}\n\\caption*{Commodities}\n\\end{subfigure}\n')
        self.figures.append(''.join(output))

    ##
    ## Draw timeline arc figure for each commodity
    ##
    def draw_timeline(self, nodes, arcs, solution=None, portrait=False):
        stretch = 15 if portrait else 24
        commodities = self.commodities

        T = max([c[2] for c in nodes])
        vScale = 0.75

        def scale(t):
            return t / float(T) * stretch

        for k in range(len(commodities)):
            tmp = ['\\begin{subfigure}[b]{\\linewidth}\n' +
                   '\\centering\n' +  
                   '\\begin{tikzpicture}\n' + 
                   '\\tikzstyle{w} = [{[-)}, color=blue!65, semithick]\n' + 
                   '\\tikzstyle{wd} = [{[-]}, color=black!50!green!75, thick]\n' + 
                   '\\tikzstyle{a} = [-latex, color=black!75, semithick]\n' + 
                   '\\tikzstyle{b} = [-latex, color=black!15]\n'
                   '\\tikzstyle{c} = [-latex, color=orange]\n'
                   '\\tikzstyle{n} = [ font={\\tiny\\bfseries}, shape=circle,inner sep=2pt, text=black, thin, draw=black, fill=white, align=center]\n' + 
                   '\\tikzstyle{s} = [ shape=circle,draw=black, inner sep=1.2pt, fill=orange!60]\n']
    
            physical_nodes = self.network.nodes()

            # draw commodity information (name, origin->destination)
            tmp.append('\\draw (-1.5,{0}) node {{$ k_{{{1}}} $}};\n'.format((len(physical_nodes) - 1) * vScale, k))
            tmp.append('\\draw (-1.5,{0}) node {{$ t:[{1},{2}] $}};\n'.format((len(physical_nodes) - 2) * vScale, commodities[k]['a'][1], commodities[k]['b'][1]))
            tmp.append('\\draw (-1.5,{0}) node {{$ q:{1} $}};\n'.format((len(physical_nodes) - 3) * vScale, commodities[k]['q']))

            # draw a line for each node (and label)
            for n in range(len(physical_nodes)):
                tmp.append('\\draw (-0.5,{0}) node [n] (n{1}) {{$ {1} $}};\n'.format(n*vScale, n))
                tmp.append('\\draw (0,{0}) -- ({1},{0});\n'.format(n * vScale, scale(T)))
    
            # dodgey, but it looks good for small numbers
            def get_angles(a, b):
                c,d = [(90,-90), (120,240), (113,247), (110,250), (110,250), (107,253)][min(abs(a-b)-1, 5)]
                return (c,d) if a < b else (d,c)

            a,b = commodities[k]['a'], commodities[k]['b']
            ta,tb = get_angles(a[0],b[0])

            tmp.append('\\draw [->] (n{0}) to[out={2},in={3}] (n{1});'.format(a[0], b[0], ta, tb))

            # draw time marks
            step = int(max(1, ceil(T / (stretch / (log10(T) * 0.3)))))

            for t in range(0,int(T+1),step):
                tmp.append('\\draw ({0},0) node[below=2pt,font=\\tiny] {{$ {1} $}};\n'.format(scale(t), t))

            # draw time windows
            for n in nodes:
                tmp.append('\\draw ({0},{2}) -- ({1},{2}) [w];\n'.format(scale(n[1]), scale(n[2]), n[0]*vScale))

            shortest_paths = nx.shortest_path_length(self.network, weight='weight')
            tmp.append('\\draw ({0},{2}) -- ({1},{2}) [wd];\n'.format(scale(a[1] + shortest_paths[a[0]][b[0]]), scale(b[1]), b[0]*vScale))

            # draw time arcs (above windows) - not including solution
            for a,b in arcs[k]:
                if solution == None or not len(solution.select(k, (a,b))) > 0:
                    #if a[0] != b[0]:
                    #    tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[2]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'b'))
                    #else:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[1]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'b'))

            # draw solution arcs
            if solution != None:
                for k,(a,b) in solution.select(k, '*'):
                    #if a[0] != b[0]:
                    #    tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[2]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'a'))
                    #else:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[1]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'a'))

            tmp.append('\\end{tikzpicture}\n\\end{subfigure}\n')
            self.figures.append(''.join(tmp))

    #
    # simplifies common discretization 
    #
    def discretize(self, step):
        nodes = self.create_node_intervals(self.discretization_network(step))               # Create node/interval pairs
        arcs = {k: self.lower_bound_arcs(c, nodes) for k,c in enumerate(self.commodities)}  # Create arc/interval pairs

        return (nodes, arcs)


    # creates arcs/nodes for unit time discretization transit times are cost function
    def discretization_network(self, step=1):
        T = max([c['b'][1] for c in self.commodities]) + 1

        # Create nodes (n,t) pairs
        return tuplelist([(n,t) for n in self.network.nodes() for t in range(0, int(T+step), step)])


    # creates node/interval pairs from node/time pairs
    def create_node_intervals(self, nodes):
        output = tuplelist()

        for n in self.network.nodes():
            time_points = sorted(t for _,t in nodes.select(n, '*'))

            for t1,t2 in zip(time_points, time_points[1:]):
                output.append((n, t1, t2))

        return output

    # creates a lower bound network from node/interval pairs
    def lower_bound_arcs(self, commodity, nodes):
        arcs = tuplelist()

        # setup storage arcs
        for n in self.network.nodes():
            next = nodes.select(n, 0, '*')[0]
            while True:
                tmp = nodes.select(n, next[2], '*')

                if not tmp:
                    break

                # setup storage arcs - only if overlaps with commodity availability
                if next[1] <= commodity['b'][1] and next[2] > commodity['a'][1]:
                    arcs.append((next,tmp[0]))
                next = tmp[0]


        shortest_paths = nx.shortest_path_length(self.network, weight='weight')

        # setup transport arcs
        for n in nodes:
            if commodity['b'][0] in shortest_paths[n[0]] and n[0] in shortest_paths[commodity['a'][0]]:
                if n[1] <= commodity['b'][1] - shortest_paths[n[0]][commodity['b'][0]] and n[2] > commodity['a'][1] + shortest_paths[commodity['a'][0]][n[0]]:
                    for e in self.network.edges(n[0]):
                        # no flow into origin or from destination
                        if e[1] == commodity['a'][0] or n[0] == commodity['b'][0]:
                            continue

                        # find closest node/time pair
                        n2 = self.find_interval(nodes, e[1], n[1] + self.network[n[0]][e[1]]['weight'])

                        if n2 != None and n2[1] <= commodity['b'][1] and n2[2] > commodity['a'][1]:
                            arcs.append((n, n2))

        return arcs

    # finds the appropriate interval to connect arc
    def find_interval(self, nodes, node, time):
        for n in nodes.select(node, '*', '*'):
            if time >= n[1] and time < n[2]:
                return n

        return None

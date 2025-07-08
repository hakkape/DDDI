from math import *
import networkx as nx
import subprocess

try:
    import pydot
    pydot_loaded = True
except ImportError:
    pydot_loaded = False

import os

class DrawLaTeX(object):
    """Outputs figures in LaTeX and pdf"""
    __slots__ = ['commodities', 'network', 'figures', 'latex', 'output_directory']

    def __init__(self, commodities, network, output_directory=None ):
        self.commodities = commodities
        self.network =  network

        self.output_directory = output_directory if output_directory != None else os.getcwd() + '/output/'

        self.figures = []
        self.latex = ''

    # draws problem data
    def save(self, filename, clean=True):      
        with open(self.output_directory + filename + '.tex', "w") as text_file:
            text_file.write(self.latex)

        self.latex = ''

        FNULL = open(os.devnull, 'w')
        proc = subprocess.Popen(['pdflatex', filename + '.tex', '-quiet'], cwd=self.output_directory, stdout=FNULL)
        proc.communicate()

        if clean:
            #if os.path.isfile(self.output_directory + filename + '.tex'):
            #    os.remove(self.output_directory + filename + '.tex')
 
            if os.path.isfile(self.output_directory + filename + '.aux'):
                os.remove(self.output_directory + filename + '.aux')

            if os.path.isfile(self.output_directory + filename + '.log'):
                os.remove(self.output_directory + filename + '.log')


    ##
    ## Defines temporary latex document for drawing figures
    ##
    def draw_latex(self, portrait=False, columns=1):
        output = []
        output.append('\\batchmode\n'
                       '\\documentclass{article}\n' +
                      '\\usepackage{tikz,amsmath, amssymb,bm,color}\n' +
                      #'\\usepackage[margin=0cm,nohead]{geometry}\n' +
                      #'\\usepackage[active,tightpage]{preview}\n' +
                      '\\usetikzlibrary{shapes,arrows}\n' +
                      '\\usetikzlibrary{calc}\n' +
                      '\\usepackage{pdflscape}\n' +
                      '\\usepackage[a2paper,margin=0cm,nohead]{geometry}\n' +
                     #'\\PreviewEnvironment{tikzpicture}\n' +
                      '\\usepackage{graphicx}\n' + 
                      '\\usepackage{caption}\n' + 
                      '\\usepackage{subcaption}\n' +
                      '\\usepackage{booktabs}\n' + 
                      '\\begin{document}\n' +
                      ('\\begin{landscape}\n' if portrait == False else ''))
       
        for i, f in enumerate(self.figures):
            if i % columns == 0:
                output.append('\\begin{figure}[h]\n'+                  
                              '\\centering\n')

            output.append('{0}'.format(f))
                          
            if (i + 1) % columns == 0:
                output.append('\\end{figure}\n\\vspace{2mm}\n')

                if (i + 1) % 16 == 0: # avoid errors for having too many figures
                    output.append('\\clearpage\n')

        if (len(self.figures) + 1) % columns != 0:
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

    def get_position(self):
        return nx.nx_pydot.pydot_layout(self.network, prog='neato') if pydot_loaded else None

    def scale_position(self, position, scale=1):
        scale *= 250

        if position == None:
            return None

        # scale positions 0->1
        mn = float(min([min(p) for p in (position.values() if isinstance(position, dict) else position)]))
        mx = float(max([max(p) for p in (position.values() if isinstance(position, dict) else position)]))

        return {k: ((v[0] - mn)/(mx-mn) * scale, (v[1] - mn)/(mx-mn) * scale) for k,v in (position.items() if isinstance(position, dict) else enumerate(position))}


    ##
    ## Draw Network figure
    ##
    def draw_network(self, scale=1, position=None, font='tiny'):
        network = self.network
        position = self.scale_position(position if position != None else self.get_position())

        if position == None:
            return

        output = ['\\begin{subfigure}[b]{20cm}\n' + 
                  '\\centering\n' +
                  '\\begin{{tikzpicture}} [>=latex\',line join=bevel,thick,scale={0}]\n'.format(scale) +
                  '\\tikzstyle{n} = [shape=circle, thick, inner sep=1pt, draw=black!55, align=center, font={\\' + font + '}, minimum size=' + ('0.5' if font == 'tiny' else '0.75') + 'cm]\n' + 
                  '\\tikzstyle{e} = [fill=white,inner sep=1pt, align=center, font={\\' + font + '\\bfseries}]\n']
    
        for n in network.nodes():
            output.append('\\draw ({0}bp,{1}bp) node [n] (n{2}) {{{2}}};\n'.format(position[n][0], position[n][1], n))

        edges = network.edges()

        for a,b in edges:
            if (b,a) in edges:
                if a > b: # simplify edges
                    output.append('\\draw [<->] (n{0}) edge node [e] {{{2}}} (n{1});\n'.format(a, b, int(network[a][b]['weight'])))
            else:
                output.append('\\draw [-latex, color=orange] (n{0}) edge node [e] {{{2}}} (n{1});\n'.format(a, b, int(network[a][b]['weight'])))

        output.append('\\end{tikzpicture}\n\\caption*{Network}\n\\end{subfigure}\n')
        self.figures.append(''.join(output))

    ##
    ## Draw Commodities
    ##
    def draw_commodities(self, scale=1, position=None, font='normal'):
        network = self.network
        position = self.scale_position(position if position != None else self.get_position())

        if position == None:
            return

        output = ['\\begin{subfigure}[b]{20cm}\n' + 
                  '\\centering\n' +
                  '\\begin{{tikzpicture}} [>=latex\',line join=bevel,thick,scale={0}]\n'.format(scale) +
                  '\\tikzstyle{n} = [shape=circle, thick, inner sep=1pt, draw=black!55, align=center, font={\\' + font + '}, minimum size=' + ('0.5' if font == 'tiny' else '0.75') + 'cm]\n']
    
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
    ## Draw Solution figure
    ##
    def draw_solution_network(self, solution, cycle, scale=1):
        for G_ in nx.weakly_connected_components(solution):
            G = solution.subgraph(G_)
            nx.drawing.nx_agraph.write_dot(G, 'test.dot')

            position = nx.nx_pydot.pydot_layout(G, prog='dot' if not cycle else 'neato') if pydot_loaded else None

            if position == None:
                return

            cycle = map(set,cycle)

            output = ['\\begin{subfigure}[b]{\\linewidth}\n' + 
                      '\\centering\n' +
                      '\\begin{{tikzpicture}} [>=latex\',line join=bevel,thick,scale={0}]\n'.format(scale) +
                      '\\tikzstyle{n} = [shape=rectangle, thick, inner sep=3pt, rounded corners, draw=black!55, align=center, minimum size=0.5cm]\n' + 
                      '\\tikzstyle{C} = []' +
                      '\\tikzstyle{e} = [fill=white,inner sep=1pt, align=center]']
    
            get = lambda n,d: str(d[n]) if n in d else ""
            nid = lambda x: str(x[0]) + str(x[1]) if type(x[1]) is not frozenset else str(x[0][0]) + str(x[0][1]) + str().join(map(str,x[1]))

            tlabel = lambda x: "$t^{{k_{{{0}}}}}_{{{1}}}$ {3}{2}".format(
                        n[0], n[1], 
                        "\\\\\\textcolor{{{2}}}{{$[{0},{1}]$}}".format(G.nodes[x]['early'], G.nodes[x]['late'], 'black' if G.nodes[x]['early'] <= G.nodes[x]['late'] else 'red') if 'early' in G.nodes[x] else "",
                        "" if 'tw' not in G.nodes[x] else "\\textcolor{{{1}}}{{${0}$}}".format(G.nodes[x]['tw'], 'black' if G.nodes[x]['tw'][0] <= G.nodes[x]['tw'][1] else 'red'))

            slabel = lambda x: "$s^{{{1}}}_{{{0}}}$ {3}{2}{4}".format(
                        n[0], 
                        str().join(map(str("k_{{{0}}},").format,n[1]))[0:-1], 
                        "\\\\\\textcolor{{{2}}}{{$[{0},{1}]$}}".format(G.nodes[x]['early'], G.nodes[x]['late'], 'black' if G.nodes[x]['early'] <= G.nodes[x]['late'] else 'red') if 'early' in G.nodes[x] else "",
                        "" if 'tw' not in G.nodes[x] else "\\textcolor{{{1}}}{{${0}$}}".format(G.nodes[x]['tw'], 'black' if G.nodes[x]['tw'][0] <= G.nodes[x]['tw'][1] else 'red'),
                        ": " + str(G.nodes[x]['diff']) if 'diff' in G.nodes[x] else '')

            nlabel = lambda x: slabel(x) if type(x[1]) is frozenset else tlabel(x)

            dot = "strict digraph "" {"

            for n in G.nodes():
                output.append('\\draw ({0}bp,{1}bp) node [n] (n{2}) {{{3}}};\n'.format(position[n][0], position[n][1], nid(n), nlabel(n)))
                dot += '"{0}" [];'.format(n)

            for a in G.edges():
                output.append('\\draw [-latex, color={4}] (n{0}) edge node [{3}] {{{2}}} (n{1});\n'.format(nid(a[0]), nid(a[1]), get('weight', G.edges[a]), 'e' if 'weight' in G.edges[a] else 'C', 'orange' if any([C.issuperset(a) for C in cycle]) else 'blue'))
                dot += '"{0}" -> "{1}" [];'.format(a[0], a[1], get('weight', G.edges[a]))

            dot += "}"
            
            # write out string to file
            with open('test2.dot', 'w') as f:
                f.write(dot)

            output.append('\\end{tikzpicture}\n\\caption*{Solution}\n\\end{subfigure}\n')
            self.figures.append(''.join(output))

    ##
    ## Draw timeline arc figure for each commodity
    ##
    ## Arc comes from middle of interval
    def draw_timeline_paper(self, nodes, arcs, stretch, solution=None, K=[]):
        commodities = self.commodities
        if not K:
            K = range(len(commodities))

        T = max([c[2] for c in nodes])
        vScale = 0.75

        def scale(t):
            return t / float(T) * stretch

        for k in K:
            tmp = ['\\begin{subfigure}[b]{\\linewidth}\n' +
                   '\\centering\n' +  
                   '\\begin{tikzpicture}\n' + 
                   '\\tikzstyle{w} = [{[-)}, color=blue!65, semithick]\n' + 
                   '\\tikzstyle{wd} = [{[-]}, color=black!50!green!75, thick]\n' + 
                   '\\tikzstyle{a} = [-latex, color=black!75, very thick]\n' + 
                   '\\tikzstyle{b} = [-latex, color=orange!75]\n'
                   '\\tikzstyle{c} = [-latex, color=orange]\n'
                   '\\tikzstyle{n} = [ font={\\tiny\\bfseries}, shape=circle,inner sep=2pt, text=black, thin, draw=black, fill=white, align=center]\n' + 
                   '\\tikzstyle{s} = [ shape=circle,draw=black, inner sep=1.2pt, fill=orange!60]\n']
    
            physical_nodes = self.network.nodes()

            # draw commodity information (name, origin->destination)
            tmp.append('\\draw (-1.75,{0}) node {{$ k_{{{1}}} $}};\n'.format((len(physical_nodes) - 1) * vScale, k))
            tmp.append('\\draw (-1.75,{0}) node {{$ t:[{1},{2}] $}};\n'.format((len(physical_nodes) - 2) * vScale, commodities[k]['a'][1], commodities[k]['b'][1]))
            tmp.append('\\draw (-1.75,{0}) node {{$ q:{1} $}};\n'.format((len(physical_nodes) - 3) * vScale, commodities[k]['q']))
            #text_file.write('\\draw (-1.5,{0}) node {{$ {1} \\rightarrow {2} $}};\n'.format((len(self.network_paths) - 2) * 0.5, self.commodities[k]['n'][0], self.commodities[k]['n'][1]))

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

            for t in range(0,int(T),step):
                tmp.append('\\draw ({0},0) node[below=2pt,font=\\tiny] {{$ {1} $}};\n'.format(scale(t + step/2.0), t))

            # draw time windows
            for n in nodes:
                tmp.append('\\fill ({0},{2} - 0.102) -- ({0},{2} + 0.102) -- ({1} - 0.1,{2} + 0.102) to[out=0,in=0,looseness=1.35] ({1} - 0.1,{2} - 0.102) -- ({0},{2} - 0.102) -- cycle [fill=transparent!5,fill opacity=1,draw=blue!65, opacity=0.9, semithick];\n'.format(scale(n[1]), scale(n[2]), n[0]*vScale))

            shortest_paths = dict(nx.shortest_path_length(self.network, weight='weight'))
            
            # destination window
            tmp.append('\\draw ({0} + 0.1,{2} - 0.2) -- ({0},{2} - 0.2) -- ({0},{2} + 0.2) -- ({0} + 0.1,{2} + 0.2)  [color=black!50!green!75, thick];\n'.format(
                scale(a[1] + shortest_paths[a[0]][b[0]]), None, b[0]*vScale))

            tmp.append('\\draw ({1} - 0.1,{2} - 0.2) -- ({1},{2} - 0.2) -- ({1},{2} + 0.2) -- ({1} - 0.1,{2} + 0.2)  [color=black!50!green!75, thick];\n'.format(
                None, scale(b[1]+step), b[0]*vScale))

            # origin window
            tmp.append('\\draw ({0} + 0.1,{2} - 0.2) -- ({0},{2} - 0.2) -- ({0},{2} + 0.2) -- ({0} + 0.1,{2} + 0.2)  [color=black!50!green!75, thick];\n'.format(
                scale(a[1]), None, a[0]*vScale))

            tmp.append('\\draw ({1} - 0.1,{2} - 0.2) -- ({1},{2} - 0.2) -- ({1},{2} + 0.2) -- ({1} - 0.1,{2} + 0.2)  [color=black!50!green!75, thick];\n'.format(
                None, scale(b[1]- shortest_paths[a[0]][b[0]]), a[0]*vScale))

            # draw time arcs (above windows) - not including solution
            for a,b,d in arcs[k]:
                if solution == None or not len(solution.select(k, (a,b))) > 0:
                    if a[0] != b[0]:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format((scale(a[2]) + scale(a[1])) / 2.0, a[0]*vScale, (scale(b[2]) + scale(b[1])) / 2.0, b[0]*vScale, 'b' if 'x' in d else 'b,color=red,thick'))
                    else:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format((scale(a[2]) + scale(a[1])) / 2.0, a[0]*vScale, (scale(b[2]) + scale(b[1])) / 2.0, b[0]*vScale, 'b'))

            # draw solution arcs
            if solution != None:
                for k,(a,b) in solution.select(k, '*'):
                    if a[0] != b[0]:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format((scale(a[2]) + scale(a[1])) / 2.0, a[0]*vScale, (scale(b[2]) + scale(b[1])) / 2.0, b[0]*vScale, 's'))
                    else:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format((scale(a[2]) + scale(a[1])) / 2.0, a[0]*vScale, (scale(b[2]) + scale(b[1])) / 2.0, b[0]*vScale, 's'))

            tmp.append('\\end{tikzpicture}\n\\end{subfigure}\n')
            self.figures.append(''.join(tmp))


    ##
    ## Draw timeline arc figure for each commodity
    ##
    def draw_timeline(self, nodes, arcs, stretch, solution=None, K=[]):
        commodities = self.commodities
        if not K:
            K = range(len(commodities))

        T = max([c[2] for c in nodes])
        vScale = 0.75

        def scale(t):
            return t / float(T) * stretch

        for k in K:
            tmp = ['\\begin{subfigure}[b]{\\linewidth}\n' +
                   '\\centering\n' +  
                   '\\begin{tikzpicture}\n' + 
                   '\\tikzstyle{w} = [{[-)}, color=blue!65, semithick]\n' + 
                   '\\tikzstyle{wd} = [{[-]}, color=black!50!green!75, thick]\n' + 
                   '\\tikzstyle{a} = [-latex, color=black!75, very thick]\n' + 
                   '\\tikzstyle{b} = [-latex, color=orange!75]\n'
                   '\\tikzstyle{c} = [-latex, color=orange]\n'
                   '\\tikzstyle{n} = [ font={\\tiny\\bfseries}, shape=circle,inner sep=2pt, text=black, thin, draw=black, fill=white, align=center]\n' + 
                   '\\tikzstyle{s} = [ shape=circle,draw=black, inner sep=1.2pt, fill=orange!60]\n']
    
            physical_nodes = self.network.nodes()

            # draw commodity information (name, origin->destination)
            tmp.append('\\draw (-1.75,{0}) node {{$ k_{{{1}}} $}};\n'.format((len(physical_nodes) - 1) * vScale, k))
            tmp.append('\\draw (-1.75,{0}) node {{$ t:[{1},{2}] $}};\n'.format((len(physical_nodes) - 2) * vScale, commodities[k]['a'][1], commodities[k]['b'][1]))
            tmp.append('\\draw (-1.75,{0}) node {{$ q:{1} $}};\n'.format((len(physical_nodes) - 3) * vScale, commodities[k]['q']))
            #text_file.write('\\draw (-1.5,{0}) node {{$ {1} \\rightarrow {2} $}};\n'.format((len(self.network_paths) - 2) * 0.5, self.commodities[k]['n'][0], self.commodities[k]['n'][1]))

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

                ##if len(xb.keys()) == 1:
                ##    tmp.append('\\draw ({0},{1}) node [s] {{ }};\n'.format(scale(xb.keys()[0]), a*vScale))  # only one point means must dispatch at this time

                ##sorted_times = sorted(xb.keys())

                ##if len(sorted_times) > 1:
                ##    tmp.append('\\draw ({0},{1}) node [s] (e{2}_{3}_{4}) {{ }};\n'.format(scale(sorted_times[-1]), a*vScale, k, a, b))  # only one point means must dispatch at this time
                ##    tmp.append('\\draw ({0},{1}) -- (e{2}_{3}_{4}) [w];\n'.format(scale(sorted_times[-2]), a*vScale, k, a, b))

                ##for t1,t2 in zip(sorted_times, sorted_times[1:])[:-1]:
                ##    tmp.append('\\draw ({0},{2}) -- ({1},{2}) [w];\n'.format(scale(t1), scale(t2), a*vScale))

            shortest_paths = nx.shortest_path_length(self.network, weight='weight')
            tmp.append('\\draw ({0},{2}) -- ({1},{2}) [wd];\n'.format(scale(a[1] + shortest_paths[a[0]][b[0]]), scale(b[1]), b[0]*vScale))

            # draw time arcs (above windows) - not including solution
            for a,b,d in arcs[k]:
                if solution == None or not len(solution.select(k, (a,b))) > 0:
                    if a[0] != b[0]:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[2]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'b' if 'x' in d else 'b,color=red,thick'))
                    else:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[1]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'b'))
                #tmp.append('\\draw ({0},{1}) [dashed] -> ({2},{3}) [a];\n'.format(scale(t2), scale(t1 + self.G[a][b]['weight']), a*vScale, b*vScale))

            # draw solution arcs
            if solution != None:
                for k,(a,b) in solution.select(k, '*'):
                    if a[0] != b[0]:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[2]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'a'))
                    else:
                        tmp.append('\\draw ({0},{1}) -> ({2},{3}) [{4}];\n'.format(scale(a[1]), a[0]*vScale, scale(b[1]), b[0]*vScale, 'a'))
                #tmp.append('\\draw ({0},{1}) [dashed] -> ({2},{3}) [a];\n'.format(scale(t2), scale(t1 + self.G[a][b]['weight']), a*vScale, b*vScale))


                #if len(xb.keys()) == 1:
                #    tmp.append('\\draw ({0},{2}) -> ({1},{3}) [c];\n'.format(scale(xb.keys()[0]), scale(xb.keys()[0] + self.G[a][b]['weight']), a*vScale, b*vScale)) # only one point means must dispatch at this time

                #sorted_times = sorted(xb.keys())
                #for t1,t2 in zip(sorted_times, sorted_times[1:])[:-1]:
                #    tmp.append('\\draw ({0},{2}) -> ({1},{3}) [b];\n'.format(scale(t1), scale(t1 + self.G[a][b]['weight']), a*vScale, b*vScale))
                #    tmp.append('\\draw ({0},{2}) [dashed] -> ({1},{3}) [a];\n'.format(scale(t2), scale(t1 + self.G[a][b]['weight']), a*vScale, b*vScale))

                #if len(sorted_times) > 1:
                #    tmp.append('\\draw (e{2}_{3}_{4}) -> ({0},{1}) [c];\n'.format(scale(sorted_times[-1] + self.G[a][b]['weight']), b*vScale, k, a, b))

                #    tmp.append('\\draw ({0},{2}) -> ({1},{3}) [b];\n'.format(scale(sorted_times[-2]), scale(sorted_times[-2] + self.G[a][b]['weight']), a*vScale, b*vScale))
                #    tmp.append('\\draw ({0},{2}) [dashed] -> ({1},{3}) [a];\n'.format(scale(sorted_times[-1]) - 1.11/stretch, scale(sorted_times[-2] + self.G[a][b]['weight']), a*vScale, b*vScale))

            tmp.append('\\end{tikzpicture}\n\\end{subfigure}\n')
            self.figures.append(''.join(tmp))


    ##
    ## Draw the solution (time slice)
    ##
#    def draw_timeslice(self, lp1):
#        lp = lp1.model
#        G = self.network

#        position = nx.pygraphviz_layout(G, prog='neato')

#        #invalid_nodes = [(self.consolidations[a][i],a[0][0]) for (a,i,p),v in self.x.items() if v.x != 0]
#        invalid_nodes = [(k,p) for (k,p),v in lp1.x.items() if v.x != 0]

#        times = sorted(set([v.x for yk in lp1.y for v in yk.values()] + 
#                           [c['a'][1] for c in self.commodities] + 
#                           [lp1.y[k][path[n]].x + lp1.transit_time(k, n, n+1) for k, path in enumerate(lp1.solution_paths) for n in range(len(path)-1)] + 
#                           [c['a'][1] for c in self.commodities]))

#        edges = [zip(path,path[1:]) for k, path in enumerate(lp1.solution_paths)]

#        # unique t
#        for time in range(len(times)):
#            output = ['\\begin{subfigure}[b]{.5\\linewidth}\n' + 
#                      '\\caption{{$t={0}$}}\n'.format(time) + 
#                      '\\begin{tikzpicture}\n' + 
#                      '\\tikzstyle{w} = [{[-)}, color=blue!65, semithick]\n' + 
#                      '\\tikzstyle{wd} = [{[-]}, color=black!50!green!75, thick]\n' + 
#                      '\\tikzstyle{a} = [-latex, color=black!75, semithick]\n' + 
#                      '\\tikzstyle{b} = [-latex, color=black!15]\n'
#                      '\\tikzstyle{c} = [-latex, color=orange]\n'
#                      '\\tikzstyle{n} = [shape=circle, thick, draw=black!55, align=center]\n'
##                      '\\tikzstyle{n} = [ font={\\tiny\\bfseries}, shape=circle,inner sep=2pt, text=black, thin, draw=black, fill=white, align=center]\n' + 
#                      '\\tikzstyle{s} = [ shape=circle,draw=black, inner sep=1.2pt, fill=orange!60]\n']

#            node_time = [self.node_at_time(lp1, k, times[time]) for k in range(len(lp1.solution_paths))]

#            for n in G.nodes():
#                items_at_node = [k for k,node in enumerate(node_time) if n == node]
#                output.append('\\draw ({0}bp,{1}bp) node [n] (n{2}) {{{3}}};\n'.format(position[n][0], position[n][1], n, str(n)))# str(items_at_node) if len(items_at_node) > 0 else ' '))

#            # get all commodities that flow at this time (including slack variables)
#            tmp = {a: [group[i] for i in range(len(group)) if lp1.y[group[i]][a[0][0]].x == times[time]] for a, group in lp1.consolidations.items()}
           
#            for a,g in tmp.items():
#                if len(g) > 0:
#                    if [(k,a[0][0]) in invalid_nodes for k in g]:
#                        output.append('\\draw [-latex, color=red] (n{0}) edge node [above] {{{2}}} (n{1});\n'.format(a[0][0], a[1][0], '{0}\\\\{1}'.format(str(g), sum([self.commodities[k]['q'] for k in g]))))
#                    else:
#                        output.append('\\draw [-latex, color=blue] (n{0}) edge node [above] {{{2}}} (n{1});\n'.format(a[0][0], a[1][0], '{0}\\\\{1}'.format(str(g), sum([self.commodities[k]['q'] for k in g]))))

#            output.append('\\end{tikzpicture}\n\\end{subfigure}\n')
#            self.figures.append(''.join(output))


    # find which node the commodity is at given time
    def node_at_time(self, lp1, k, time):
        path = lp1.solution_paths[k]
        commodities = self.commodities

        # outside of time window
        if time < commodities[k]['a'][1] or time > commodities[k]['b'][1]:
            return None

        # added to origin
        if time >= commodities[k]['a'][1] and time <= lp1.y[k][path[0]].x:
            return commodities[k]['a'][0]

        # between dispatches
        if len(path) > 1:
            for i in range(len(path) - 2):
                if time >= lp1.y[k][path[i]].x + lp1.transit_time(k, i, i+1) and time <= lp1.y[k][path[i+1]].x:
                    return path[i]

        # arrived at destination
        if time >= lp1.y[k][path[-2]].x + lp1.transit_time(k, len(path)-2, len(path)-1):
            return commodities[k]['b'][0]

        # at last arc (between dispatches)
        #if time > self.sy[k][path[-1]].x and time < self.sy[k][path[-1]].x + lookup_transit_time(path[-1]):
        #    return None

        return None  # shouldn't actually get here

from random import random

from pm4py.objects.process_tree import util as pt_util
from pm4py.objects.process_tree import semantics as pt_semantics
from pm4py.objects.process_tree import pt_operator as pt_opt
from pm4py.objects.process_tree import state as pt_st
#from pm4py.objects.process_tree import process_tree as pt
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.transition_system import utils as ts_util
from pm4py.visualization.transition_system.util import visualize_graphviz as visual_ts
from pm4py.visualization.transition_system import factory as visual_ts_factory


#def construct(pt, trace):

   #Constructs the synchronous product tree net of a given process tree and a trace


class Move(object):

    def __init__(self, log, model):
        self._log = log
        self._model = model

    def _set_log(self, log):
        self._log = log

    def _get_log(self):
        return self._log

    def _set_model(self, model):
        self._model= model

    def _get_model(self):
        return self._model

    def __repr__(self):
        if self.log == None and self.model == None:
            return '(>>, >>)'
        elif self.log == None:
            return '(>>, ' + str(self._model) + ')'
        elif self.model == None:
            return '(' + str(self._log) + ', >>)'
        else:
            return '(' + str(self._log) + ', ' + str(self._model) + ')'

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return True if str(other) == self.__repr__() else False

    log = property(_get_log, _set_log)
    model = property(_get_model, _set_model)


class State(object):
    def __init__(self, log, model, node=None):
        self._log = log
        self._model = model
        self._node = node

    def _set_log(self, log):
        self._log = log

    def _get_log(self):
        return self._log

    def _set_model(self, model):
        self._model = model

    def _get_model(self):
        return self._model

    def _get_node(self):
        return self._node

    def __repr__(self):
        string = "(" + str(self._log) + ", ("

        for i in range(0, len(self._model)):
            if self._model[i] == pt_st.State.ENABLED:
                string += "E"
            elif self._model[i] == pt_st.State.FUTURE_ENABLED:
                string += "F"
            elif self._model[i] == pt_st.State.OPEN:
                string += "O"
            elif self._model[i] == pt_st.State.CLOSED:
                string += "C"
        string += "))"
        return string

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return True if str(other) == self.__repr__() else False

    log = property(_get_log,_set_log)
    model = property(_get_model, _set_model)
    node = property(_get_node)


def execute(pt, trace):
    """
    Execute the process tree, returning an execution sequence

    Parameters
    -----------
    pt
        Process tree

    Returns
    -----------
    exec_sequence
        Execution sequence on the process tree
    """
    # TODO: add trace as parameter
    enabled, f_enabled, open, closed = list(), list(), list(), list()
    enabled.append(pt)
    populate_closed(pt.children, closed) # set all child nodes to closed
    execution_sequence = list()

    print('Start closed', closed)



    # procces tree config
    pt_config = list()
    pt_config.append(pt_st.State.ENABLED)
    i_nodes = tree.index_nodes(0)  # index process tree nodes, returns the number of nodes
    for i in range(1, i_nodes):  # set list to start configuration
        pt_config.append(pt_st.State.CLOSED)
    root = ts.TransitionSystem.State(State(0, pt_config))
    ts_system = ts.TransitionSystem('Sync_Net', None, None)
    ts_system.states.add(root)
    init_state = State(0, pt_config, root)
    all_states = list()
    all_states.append(init_state)

    dummy_node = root
    #log moves for init state
    for i in range(0, len(trace)):
        dummy_node = add_node_syn_net(all_states, ts_system,dummy_node ,trace,i,pt_config, None, False)

    print("---",all_states)

    execute_enabled(enabled, f_enabled, open, closed, pt_config,ts_system,root, trace,0, all_states, execution_sequence,)
    print(enabled)


    graph = visual_ts.visualize(ts_system)
    visual_ts_factory.view(graph)

    return execution_sequence


def add_node_syn_net(all_states, ts_system, old_node,trace, new_node_i_trace, new_node_config, new_node_operator, model_move):
    ts_new_node = None


    if (State(new_node_i_trace, new_node_config) in all_states and model_move == True) or State(new_node_i_trace+1, new_node_config) in all_states and model_move == False  :
        edge_exists = False
        #if (model_move == False and len(trace) > new_node_i_trace) or model_move:
        for outgoing in old_node.outgoing:
            new_edge = None

            new_edge = Move(trace[new_node_i_trace], new_node_operator)


            if outgoing.name == new_edge:
                print("Help")
                # edge exists
                edge_exists = True
                ts_new_node = outgoing.to_state


        if edge_exists == False:
            if model_move:
                ts_new_node = all_states[all_states.index(State(new_node_i_trace, new_node_config))].node
            else:
                ts_new_node = all_states[all_states.index(State(new_node_i_trace+1, new_node_config))].node

            if ts_new_node.name.model[0] == pt_st.State.CLOSED:
                ts_util.add_arc_from_to(Move(None, None), old_node, ts_new_node, ts_system, None)


            #model move
            elif model_move:
                ts_util.add_arc_from_to(Move(None, new_node_operator), old_node, ts_new_node, ts_system, None)
            # log move
            else:
                ts_util.add_arc_from_to(Move(trace[new_node_i_trace], None), old_node, ts_new_node, ts_system, None)


    else:
        work_pt_config = new_node_config.copy()
        new_state = None


        if new_node_config[0] == pt_st.State.CLOSED:
            ts_new_node = ts.TransitionSystem.State(State(new_node_i_trace, work_pt_config), None, None, None)
            ts_system.states.add(ts_new_node)
            new_state = State(new_node_i_trace, work_pt_config, ts_new_node)
            ts_util.add_arc_from_to(Move(None, None), old_node, ts_new_node, ts_system, None)

        # model move
        if model_move:
            ts_new_node = ts.TransitionSystem.State(State(new_node_i_trace, work_pt_config), None, None, None)
            ts_system.states.add(ts_new_node)
            new_state = State(new_node_i_trace, work_pt_config, ts_new_node)
            ts_util.add_arc_from_to(Move(None, new_node_operator), old_node, ts_new_node, ts_system, None)

        elif len(trace) > new_node_i_trace:
            # log move
            ts_new_node = ts.TransitionSystem.State(State(new_node_i_trace+1, work_pt_config), None, None, None)
            ts_system.states.add(ts_new_node)
            new_state = State(new_node_i_trace+1, work_pt_config, ts_new_node)

            ts_util.add_arc_from_to(Move(trace[new_node_i_trace], None), old_node, ts_new_node, ts_system, None)
        all_states.append(new_state)

    # sync move
    if len(trace) > new_node_i_trace and new_node_operator == trace[new_node_i_trace] and model_move == True:
        print(new_node_operator, " sync")
        ts_new_node_sync = ts.TransitionSystem.State(State(new_node_i_trace + 1, new_node_config.copy()), None, None, None)
        all_states.append(State(new_node_i_trace + 1, new_node_config.copy(), ts_new_node_sync))
        ts_system.states.add(ts_new_node_sync)
        ts_util.add_arc_from_to(Move(trace[new_node_i_trace], new_node_operator), old_node, ts_new_node_sync,ts_system, None)






    if ts_new_node == None:
        print("hilfe", edge_exists)
    return ts_new_node




def execute_enabled(enabled, f_enabled, open, closed , pt_config ,ts_system ,from_ts, trace, i_trace, all_states ,execution_sequence=None):



    # TODO Connect States in ty_system and add trace with log moves, sync moves
    # Todo add CCCC node with tau transition



    """
    Execute an enabled node of the process tree

    Parameters
    -----------
    enabled
        Enabled nodes
    open
        Open nodes
    closed
        Closed nodes
    execution_sequence
        Execution sequence

    Returns
    -----------
    execution_sequence
        Execution sequence
    """
   # execution_sequence = list() if execution_sequence is None else execution_sequence

    length_enabled = len(enabled)
    for v in range(0, length_enabled):
        #Todo all states auch fuer states nicht in ts ? -> test ig node = none
        #todo save following nodes from a visited node ? to speed up process
        work_enabled = enabled.copy()
        work_f_enabled = f_enabled.copy()
        work_open = open.copy()
        work_pt_config = pt_config.copy()
        work_closed = closed.copy()
        work_i_trace = i_trace
        vertex = enabled[v]
        work_from_ts = from_ts #todo real copy ?
        del work_enabled[v]  #  enabled.remove(vertex) todo check the delete
        work_open.append(vertex)
        work_pt_config[vertex.index_c] = pt_st.State.OPEN
        if State(i_trace, work_pt_config) in all_states:
            # check if current node already exists
            if len(all_states) != len(trace)+1:
                print("not knoe")
                add_node_syn_net(all_states,ts_system, from_ts, trace, i_trace,work_pt_config, vertex.label)
        else:
      #  execution_sequence.append((vertex, pt_st.State.OPEN)) #
            if len(vertex.children) > 0:
                # sequence
                if vertex.operator is pt_opt.Operator.SEQUENCE:
                    c = vertex.children[0]
                    work_enabled.append(c)
                    work_pt_config[c.index_c] = pt_st.State.ENABLED
                    work_closed.remove(c)
                    # set rest to future_enabled
                    for i in range(1, len(vertex.children)):
                        work_f_enabled.append(vertex.children[i])
                        work_pt_config[vertex.children[i].index_c] = pt_st.State.FUTURE_ENABLED
                    execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system,
                                    work_from_ts,trace, work_i_trace, all_states, execution_sequence)
                elif vertex.operator is pt_opt.Operator.LOOP:   # loop
                    c = vertex.children[0]
                    work_enabled.append(c)
                    work_pt_config[c.index_c] = pt_st.State.ENABLED
                    work_closed.remove(c)

                    execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system, # No redo
                                    work_from_ts,trace, work_i_trace, all_states, execution_sequence)

                    for i in range(1, len(vertex.children)):  # all redo possibilities
                        c = vertex.children[i]
                        copy_work_f_enabled = work_f_enabled.copy()
                        copy_work_f_enabled.append(c)
                        copy_work_closed = work_closed.copy()
                        copy_work_closed.remove(c)
                        copy_work_pt_config = work_pt_config.copy()
                        copy_work_pt_config[vertex.children[i].index_c] = pt_st.State.FUTURE_ENABLED
                        execute_enabled(work_enabled, copy_work_f_enabled, work_open, copy_work_closed, copy_work_pt_config,
                                        ts_system, work_from_ts,trace, work_i_trace, all_states,  execution_sequence)
                # parallel
                elif vertex.operator is pt_opt.Operator.PARALLEL:
                    work_enabled.extend(vertex.children)
                    for child in vertex.children:
                        work_closed.remove(child)

                    for i in range(0, len(vertex.children)): # set children to enabled in config
                        work_pt_config[vertex.children[i].index_c] = pt_st.State.ENABLED
                    execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system,
                                                     work_from_ts, trace, work_i_trace, all_states, execution_sequence)
                # xor
                elif vertex.operator is pt_opt.Operator.XOR:
                    vc = vertex.children

                    for i in range(0, len(vc)):
                        c = vc[i]
                        copy_work_enabled = work_enabled.copy()
                        copy_work_enabled.append(c)
                        copy_work_closed = work_closed.copy()
                        copy_work_open = work_open.copy()
                        copy_work_closed.remove(c)
                        copy_work_pt_config = work_pt_config[:]
                        copy_work_pt_config[vertex.children[i].index_c] = pt_st.State.ENABLED
                        execute_enabled(copy_work_enabled, work_f_enabled, copy_work_open, copy_work_closed, copy_work_pt_config, ts_system,
                                        work_from_ts, trace, work_i_trace, all_states, execution_sequence)

                  #  execution_sequence.append((c, pt_st.State.ENABLED))



            else:
                # Todo was ist bei tau ? das kein Move
                # Todo alle endknoten muessen zu dem knoten n, CCCCC fueren da der zielknoten für a* ist (oder setzt zielknoten auf n, ein state davor (schlechte idee wenn mehrere knoten n knoten sind))
                # model move
                old_node = from_ts
                dummy_new_node = add_node_syn_net(all_states, ts_system, from_ts, trace, i_trace,  work_pt_config, vertex.label, True)
                ts_new_node = dummy_new_node

                print(all_states)

                if(i_trace != 0):
                    print("i_trace nicht gleich 0")

                # log move
                for i in range(0,len(trace)):
                    #print(all_states)
                    #print(vertex.label)
                    dummy_new_node = add_node_syn_net(all_states, ts_system, dummy_new_node , trace, i,  work_pt_config,vertex.label, False) # log move
                    dummy_old_node = all_states[all_states.index(State(i+1,old_node.name.model))].node # getting the log move from node #todo or going from edge from old node ?
                    add_node_syn_net(all_states, ts_system, dummy_old_node, trace, i+1,  work_pt_config, vertex.label, True) # connect from the logmove from old node to the new logmove (so an model move)




                #redo loop by setting loop node to enabled and children to closed
                parent = vertex.parent
                if parent.operator is pt_opt.Operator.LOOP:
                    if parent.children[0] != vertex:
                        work_copy_enabled = work_enabled.copy()
                        work_copy_open = work_open.copy()
                        work_copy_closed = work_closed.copy()

                        work_copy_enabled.append(parent)
                        work_copy_open.remove(parent)

                        work_copy_closed.append(vertex)
                        work_copy_open.remove(vertex)

                        execute_enabled(work_copy_enabled, work_f_enabled, work_copy_open, work_copy_closed, work_pt_config, ts_system,
                                        work_from_ts, trace, work_i_trace, all_states, execution_sequence)

                close(vertex, work_enabled,work_f_enabled, work_open, work_closed,work_pt_config,execution_sequence)


                # transition to end configuration

                if (len(work_enabled)+len(work_f_enabled) + len(work_open)) == 0:
                    print(all_states)
                    j = all_states.index(State(len(trace), ts_new_node.name.model))
                    add_node_syn_net(all_states, ts_system, all_states[j].node, trace, len(trace), work_pt_config, None, True)




                execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system,
                                ts_new_node, trace, work_i_trace,all_states,  execution_sequence)


    return execution_sequence

def populate_closed(nodes, closed):
    """
    Populate all closed nodes of a process tree

    Parameters
    ------------
    nodes
        Considered nodes of the process tree
    closed
        Closed nodes
    """
    for child in nodes:
        closed.append(child)
    for node in nodes:
        populate_closed(node.children, closed)


def close(vertex,  enabled, f_enabled, open, closed,pt_config,  execution_sequence):
    """
    Close a given vertex of the process tree

    Parameters
    ------------
    vertex
        Vertex to be closed
    enabled
        Set of enabled nodes
    open
        Set of open nodes
    closed
        Set of closed nodes
    execution_sequence
        Execution sequence on the process tree
    """

    open.remove(vertex)
    closed.append(vertex)
    pt_config[vertex.index_c] = pt_st.State.CLOSED
    # execution_sequence.append((vertex, pt_st.State.CLOSED))
    process_closed(vertex, enabled, f_enabled, open, closed, pt_config, execution_sequence)


def process_closed(closed_node, enabled, f_enabled, open, closed,pt_config, execution_sequence):

    """
    Process a closed node, deciding further operations

    Parameters
    -------------
    closed_node
        Node that shall be closed
    enabled
        Set of enabled nodes
    open
        Set of open nodes
    closed
        Set of closed nodes
    execution_sequence
        Execution sequence on the process tree
    """
    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node, enabled, f_enabled):
            close(vertex, enabled, f_enabled, open, closed,pt_config,execution_sequence)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                enable = vertex.children[vertex.children.index(closed_node) + 1]
                f_enabled.remove(enable)
            elif vertex.operator is pt_opt.Operator.LOOP: # sets future enabled to enabled
                for i in range(1,len(vertex.children)):
                    if vertex.children[i] in f_enabled:
                        if enable is None:
                            enable = vertex.children[i]
                            f_enabled.remove(enable)
                            pt_config[enable.index_c] = pt_st.State.ENABLED

            if enable is not None:
                enabled.append(enable)
                # execution_sequence.append((enable, pt_st.State.ENABLED))


def should_close(vertex, closed, child, enabled, f_enabled):
    """
    Decides if a parent vertex shall be closed based on
    the processed child

    Parameters
    ------------
    vertex
        Vertex of the process tree
    closed
        Set of closed nodes
    child
        Processed child

    Returns
    ------------
    boolean
        Boolean value (the vertex shall be closed)
    """

    if vertex.children is None:
        return True
    elif vertex.operator is pt_opt.Operator.LOOP:
        close = True
        for i in range(1, len(vertex.children)):
            if vertex.children[i] in enabled or vertex.children[i] in f_enabled:
                close = False

        return close



    elif vertex.operator is pt_opt.Operator.SEQUENCE:
        return vertex.children.index(child) == len(vertex.children) - 1
    else:

        close = True
        for i in range(0, len(vertex.children)):
            if vertex.children[i] not in closed:
                close = False
        return close

def config_in_search_tree(pt_config,i_trace , ex_tree):
    # checks if condig is already created if not add the config to the tree
    current_node = ex_tree.copy()
    edges = current_node.outgoing.copy()
    new_entry = True
    for i in range(0, len(edges)):
        if i_trace == edges.pop():
            new_entry = False

    if new_entry == False:
        for i in range(0, len(pt_config)):
            edges = current_node.outgoing.copy()
            for j in range(0, len(edges)):
                if pt_config[i] == edges.pop():
                    new_entry == False

    def is_state_in_net(pt_config, i_trace, ex_tree):
            i_trace = ex_tree




trace = list()
#for i in range(0,3):
#    trace.append('a')
trace.append('a')
trace.append('b')
trace.append('c')

#tree = pt_util.parse("+('a','b','c')")
#tree =  pt_util.parse("->(*('a','d'),'b','c')")
tree = pt_util.parse("->('a','b','c')")

print('test label: ', tree.label, ' operator: ', tree.operator)

test = execute(tree, trace)

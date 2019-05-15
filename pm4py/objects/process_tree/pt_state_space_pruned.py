from pm4py.objects.process_tree import pt_operator as pt_opt
from pm4py.objects.process_tree import state as pt_st
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.transition_system import utils as ts_util
from pm4py.objects.process_tree import util as pt_util
from pm4py.visualization.transition_system.util import visualize_graphviz as visual_ts
from pm4py.visualization.transition_system import factory as visual_ts_factory

SKIP = ">>"
TAU = '\u03C4'


class Move(object):

    def __init__(self, log, model, derive = None):
        self._log = log
        self._model = model
        # self._derive = derive

    def _set_log(self, log):
        self._log = log

    def _get_log(self):
        return self._log

    def _set_model(self, model):
        self._model = model

    def _get_model(self):
        return self._model

    '''
    def _get_derive(self):
        return self._derive
    '''

    def __repr__(self):
        if self.log is TAU or self.model is TAU:
            return TAU   # ''(>>, >>)'
        elif self.log is SKIP:
            return '(>>, ' + str(self._model) + ')'
        elif self.model is SKIP:
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
    # derive = property(_get_derive)


class State(object):
    def __init__(self, log, model, node=None, state_set=None):
        self._log = log
        self._model = model
        self._node = node
        self._state_set = state_set

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

    def _get_state_set(self):
        return self._state_set

    def _set_state_set(self, state_set):
        self._state_set = state_set

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

    log = property(_get_log, _set_log)
    model = property(_get_model, _set_model)
    node = property(_get_node)
    state_set = property(_get_state_set, _set_state_set)

class StateSet(object):
    def __init__(self, states):
        if isinstance(states, State):
            self._states = [states]

        else:
            self._states = states

    def states(self):
        return self._states

    def add_state(self, state):
        self._states.append(state)

    def __repr__(self):
        if len(self._states) > 1:
            string = "(" + self._states + ", ("

            for i in range(1, len(self._states)):
                string += ', ' + str(self._states[i])

            string += ")"
        else:
            string = str(self._states[0])

        return string

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):

        if isinstance(other, State):
            if len(self._states) > 1:
                return False

            else:
                return self._states[0] == other

        else:
            for i in self._states:
                exists = False
                for j in other:
                    if i == j:
                        exists = True
                if exists is False:
                    return False
            return True


def execute(pt, trace):
    """
    Calculates the State Net of Process tree and a trace

    Parameters
    -----------
    pt
        Process tree
    trace
        Trace
    Returns
    -----------
    exec_sequence
        Execution sequence on the process tree
    """
    enabled, f_enabled, open, closed = list(), list(), list(), list()
    enabled.append(pt)

    # set all child nodes to closed
    populate_closed(pt.children, closed)

    print('Start closed', closed)

    # process tree config
    pt_config = list()
    pt_config.append(pt_st.State.ENABLED)
    i_nodes = pt.index_nodes(0)  # index process tree nodes, returns the number of nodes
    for i in range(1, i_nodes):    # set list to start configuration
        pt_config.append(pt_st.State.CLOSED)
    root = ts.TransitionSystem.State(State(0, pt_config))
    ts_system = ts.TransitionSystem('Sync_Net', None, None)
    ts_system.states.add(root)
    init_state = State(0, pt_config, root)
    all_states = list()
    all_states.append(init_state)

    dummy_node = root
    # log moves for init state
    for i in range(0, len(trace)):
        dummy_node = add_node_syn_net(all_states, ts_system, dummy_node, trace, i, pt_config, None, False)

    execute_enabled(enabled, f_enabled, open, closed, pt_config, ts_system, root, trace, 0, all_states)
    print(enabled)

    return ts_system


def cal_derive(old_config, new_config):
    derive = list()
    for i in range(0, len(old_config)):
        if old_config[i] == new_config[i]:
            derive.append(None)
        else:
            derive.append((old_config[i], new_config[i]))

    return derive


def add_node_syn_net(all_states, ts_system, old_node, trace, new_node_i_trace,
                     new_node_config, new_node_operator, model_move):
    ts_new_node = None

    # difference between old and new node
    data = dict()
    data['derive'] = cal_derive(old_node.name.model, new_node_config)

    if(State(new_node_i_trace, new_node_config) in all_states and model_move is True) \
            or State(new_node_i_trace+1, new_node_config) in all_states and model_move is False:
        edge_exists = False

        # TODO Check this case
        """"
        new_edge = Move(trace[new_node_i_trace], None)
        for outgoing in old_node.outgoing:
            print("name ",outgoing.name, " edge ", new_edge )
            if outgoing.name == new_edge:
                # todo what if same edge but different nodes
                print(old_node,' to ' ,State(new_node_i_trace, new_node_config) ," Help")
                # edge exists
                edge_exists = True
                ts_new_node = outgoing.to_state
        """

        if edge_exists is False:

            # find the to node
            if model_move:
                ts_new_node = all_states[all_states.index(State(new_node_i_trace, new_node_config))].node
            else:
                ts_new_node = all_states[all_states.index(State(new_node_i_trace+1, new_node_config))].node

            # model move
            if model_move:
                ts_util.add_arc_from_to(Move(SKIP, new_node_operator), old_node, ts_new_node, ts_system, data)
            # log move
            else:
                ts_util.add_arc_from_to(Move(trace[new_node_i_trace], SKIP), old_node, ts_new_node, ts_system, data)

    else:
        work_pt_config = new_node_config.copy()
        new_state = None
        # model move
        if model_move:
            ts_new_node = ts.TransitionSystem.State(State(new_node_i_trace, work_pt_config), None, None, None)
            ts_system.states.add(ts_new_node)
            new_state = State(new_node_i_trace, work_pt_config, ts_new_node)
            ts_util.add_arc_from_to(Move(SKIP, new_node_operator), old_node, ts_new_node, ts_system, data)

        elif len(trace) > new_node_i_trace:
            # log move
            ts_new_node = ts.TransitionSystem.State(State(new_node_i_trace+1, work_pt_config), None, None, None)
            ts_system.states.add(ts_new_node)
            new_state = State(new_node_i_trace+1, work_pt_config, ts_new_node)

            ts_util.add_arc_from_to(Move(trace[new_node_i_trace], SKIP), old_node, ts_new_node, ts_system, data)
        all_states.append(new_state)

    # sync move
    if len(trace) > new_node_i_trace and new_node_operator == trace[new_node_i_trace] \
            and new_node_operator is not None and model_move is True:
        ts_new_node_sync = ts.TransitionSystem.State(State(new_node_i_trace + 1,
                                                           new_node_config.copy()), None, None, None)
        all_states.append(State(new_node_i_trace + 1, new_node_config.copy(), ts_new_node_sync))
        ts_system.states.add(ts_new_node_sync)
        ts_util.add_arc_from_to(Move(trace[new_node_i_trace], new_node_operator),
                                old_node, ts_new_node_sync, ts_system, data)

    return ts_new_node


def execute_enabled(enabled, f_enabled, open, closed, pt_config, ts_system, from_ts, trace, i_trace, all_states):

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

    """
    length_enabled = len(enabled)
    for v in range(0, length_enabled):
        # Todo all states auch fuer states nicht in ts ? -> test ig node = none
        # todo save following nodes from a visited node ? to speed up process
        work_enabled = enabled.copy()
        work_f_enabled = f_enabled.copy()
        work_open = open.copy()
        work_pt_config = pt_config.copy()
        work_closed = closed.copy()
        work_i_trace = i_trace
        vertex = enabled[v]
        work_from_ts = from_ts
        work_enabled.remove(vertex)
        work_open.append(vertex)
        work_pt_config[vertex.index_c] = pt_st.State.OPEN

        if State(i_trace, work_pt_config) in all_states:
            # check if current node already exists
            # simple connect is like a model move with a already existing new node
            # connect for each log move

            # do not recognize init nodes as  visited nodes
            if len(all_states) != len(trace)+1:
                # new_from_node = from_ts
                for t in range(i_trace, len(trace)+1):
                    new_from_node = all_states[all_states.index(State(t, from_ts.name.model))].node
                    add_node_syn_net(all_states, ts_system, new_from_node, trace, t, work_pt_config, vertex.label, True)

        else:
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
                                    work_from_ts, trace, work_i_trace, all_states)
                # loop
                elif vertex.operator is pt_opt.Operator.LOOP:

                    if len(vertex.children) != 3:
                        raise ValueError("Loop requires exact 3 children!")

                    # No redo
                    c = vertex.children[0]
                    work_enabled.append(c)
                    work_f_enabled.append(vertex.children[2])
                    work_pt_config[c.index_c] = pt_st.State.ENABLED
                    work_pt_config[vertex.children[2].index_c] = pt_st.State.FUTURE_ENABLED

                    work_closed.remove(vertex.children[2])
                    work_closed.remove(c)

                    execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system,
                                    work_from_ts, trace, work_i_trace, all_states)

                    # Redo
                    c = vertex.children[1]
                    copy_work_f_enabled = f_enabled.copy()
                    copy_work_f_enabled.append(c)

                    copy_work_closed = closed.copy()
                    copy_work_closed.remove(c)
                    copy_work_closed.remove(vertex.children[0])

                    copy_work_pt_config = pt_config.copy()
                    copy_work_pt_config[vertex.children[0].index_c] = pt_st.State.ENABLED
                    copy_work_pt_config[vertex.index_c] = pt_st.State.OPEN
                    copy_work_pt_config[c.index_c] = pt_st.State.FUTURE_ENABLED

                    execute_enabled(work_enabled, copy_work_f_enabled, work_open, copy_work_closed,
                                    copy_work_pt_config, ts_system, work_from_ts, trace, work_i_trace, all_states)
                # parallel
                elif vertex.operator is pt_opt.Operator.PARALLEL:
                    work_enabled.extend(vertex.children)
                    for child in vertex.children:
                        work_closed.remove(child)
                    # set children to enabled in config
                    for i in range(0, len(vertex.children)):
                        work_pt_config[vertex.children[i].index_c] = pt_st.State.ENABLED
                    execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system,
                                    work_from_ts, trace, work_i_trace, all_states)
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
                        copy_work_pt_config = work_pt_config.copy()
                        copy_work_pt_config[vertex.children[i].index_c] = pt_st.State.ENABLED

                        execute_enabled(copy_work_enabled, work_f_enabled, copy_work_open, copy_work_closed,
                                        copy_work_pt_config, ts_system, work_from_ts, trace, work_i_trace, all_states)

            # encountered leaf
            else:
                # model move
                old_node = from_ts
                dummy_new_node = add_node_syn_net(all_states, ts_system, from_ts, trace,
                                                  i_trace,  work_pt_config, vertex.label, True)
                # new node after model move
                ts_new_node = dummy_new_node

                # log move
                for i in range(0, len(trace)):
                    # log move
                    dummy_new_node = add_node_syn_net(all_states, ts_system, dummy_new_node,
                                                      trace, i, work_pt_config, vertex.label, False)
                    # getting the log move from node
                    dummy_old_node = all_states[all_states.index(State(i+1, old_node.name.model))].node

                    # connect from the log move from old log move node to the new log move (so an model move)
                    add_node_syn_net(all_states, ts_system, dummy_old_node, trace, i+1,
                                     work_pt_config, vertex.label, True)

                close(vertex, work_enabled, work_f_enabled, work_open, work_closed, work_pt_config,
                      ts_system, all_states, from_ts, i_trace)

                if vertex.parent.operator == pt_opt.Operator.LOOP and vertex.parent in work_f_enabled:

                    c_work_enabled = work_enabled.copy()
                    c_work_f_enabled = work_f_enabled.copy()
                    c_work_pt_config = work_pt_config

                    c_work_enabled.append(vertex.parent)
                    c_work_f_enabled.remove(vertex.parent)
                    c_work_pt_config[vertex.parent.index_c] = pt_st.State.ENABLED
                    print(work_pt_config, work_enabled, work_f_enabled, work_closed, ts_new_node, old_node)

                    if len(all_states) != len(trace) + 1:
                        # new_from_node = ts_new_node
                        for t in range(i_trace, len(trace) + 1):
                            new_from_node = all_states[all_states.index(State(t, ts_new_node.name.model))].node
                            add_node_syn_net(all_states, ts_system, new_from_node, trace, t, c_work_pt_config, TAU,
                                             # vertex label
                                             True)



                else:
                    # transition to end configuration

                    if (len(work_enabled) + len(work_enabled) + len(work_open) + len(work_f_enabled)) == 0:
                        j = all_states.index(State(len(trace), ts_new_node.name.model))
                        add_node_syn_net(all_states, ts_system, all_states[j].node, trace, len(trace)
                                         , work_pt_config, TAU, True)

                    execute_enabled(work_enabled, work_f_enabled, work_open, work_closed, work_pt_config, ts_system,
                                    ts_new_node, trace, work_i_trace,all_states)




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


def close(vertex,  enabled, f_enabled, open, closed,pt_config, ts_system, all_states, from_ts, i_trace):
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
    """
    open.remove(vertex)
    closed.append(vertex)
    pt_config[vertex.index_c] = pt_st.State.CLOSED
    process_closed(vertex, enabled, f_enabled, open, closed, pt_config, ts_system, all_states, from_ts, i_trace)


def process_closed(closed_node, enabled, f_enabled, open, closed, pt_config, ts_system, all_states, from_ts, i_trace):

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
    """
    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node, enabled, f_enabled):

            if vertex.operator is pt_opt.Operator.LOOP and closed_node == vertex.children[1]:

                f_enabled.append(vertex)
                open.remove(vertex)
                pt_config[vertex.index_c] = pt_st.State.FUTURE_ENABLED

            else:
                if vertex.operator is pt_opt.Operator.XOR:
                    vertex = vertex
                    # if all_states[vertex.index_c].state_set is not None:

                    #else:

                    # todo implement pruning
                elif vertex.operator is pt_opt.Operator.PARALLEL:
                    vertex = vertex
                    # todo implement pruning
                close(vertex, enabled, f_enabled, open, closed, pt_config, ts_system, all_states, from_ts, i_trace)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE:

                # sets future enabled to enabled
                enable = vertex.children[vertex.children.index(closed_node) + 1]
                f_enabled.remove(enable)

            elif vertex.operator is pt_opt.Operator.LOOP:

                for i in range(0, len(vertex.children)):
                    if vertex.children[i] in f_enabled:
                        if enable is None:
                            enable = vertex.children[i]
                            f_enabled.remove(enable)
                            pt_config[enable.index_c] = pt_st.State.ENABLED

            if enable is not None:
                enabled.append(enable)



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
        for i in range(0, len(vertex.children)): #TODO instead 0 index of closed node
            if vertex.children[i] not in closed:
                close = False
        return close




trace = list()
# for i in range(0,3):
#    trace.append('a')
trace.append('a')
#trace.append('b')
#trace.append('c')

tree = pt_util.parse("X('a',->('b','c'))")
# tree =  pt_util.parse("->(*('a','d'),'b','c')")
# tree = pt_util.parse("+('a','b','c')")
#tree = pt_util.parse("+('a','b')")

ts_system = execute(tree, trace)

graph = visual_ts.visualize(ts_system)
visual_ts_factory.view(graph)

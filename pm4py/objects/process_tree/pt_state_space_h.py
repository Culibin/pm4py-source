from pm4py.objects.process_tree import pt_operator as pt_opt
from pm4py.objects.process_tree import state as pt_st
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.transition_system import utils as ts_util
from pm4py.objects.process_tree import util as pt_util
from pm4py.visualization.transition_system.util import visualize_graphviz as visual_ts
from pm4py.visualization.transition_system import factory as visual_ts_factory
from pm4py.objects.process_tree.pt_state_space import Move as Move
from pm4py.objects.process_tree.pt_state_space import SbState as SbState
from enum import Enum
from pm4py.objects.process_tree.alignment import apply_cost_function as apply_cost_function
import random
import heapq

SKIP = ">>"
TAU = '\u03C4'

# used for heap to help compare tuples
counter = 0

class Action(Enum):
    # open prcess tree
    START = 1
    # close prcess tree
    CLOSE = 2


def execute(pt, trace):
    # todo heurisik einbauen
    #
    open_list = []
    closed_list = set()

    global counter

    ts_system = ts.TransitionSystem('Sync_Net', None, None)
    i_nodes = pt.index_nodes(0)  # index process tree nodes, returns the number of nodes

    init_sb_config = [pt_st.State.ENABLED]
    goal_config = [pt_st.State.CLOSED]
    for i in range(1, i_nodes):  # set list to start configuration
        init_sb_config.append(pt_st.State.CLOSED)
        goal_config.append(pt_st.State.CLOSED)
    init_sb_node = SbState(0, init_sb_config, pt)
    init_ts_node = ts.TransitionSystem.State(init_sb_node)
    init_sb_node.node = init_ts_node
    ts_system.states.add(init_ts_node)

    heapq.heappush(open_list, (0, counter, init_ts_node))
    counter += 1
    all_states = [init_sb_node]

    goal_sb = SbState(len(trace), goal_config, None)

    while not len(open_list) == 0:
        current_node = heapq.heappop(open_list)

        # path found
        if current_node[2] == goal_sb:
            return goal_sb

        closed_list.add(current_node[2])

        new_list = list()
        expand_node(current_node[2], open_list, closed_list)

    # no Path found
    print('no path found')
    return 0


def add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, model_label, trace, trace_i,
                   list_action):
    # todo: actions in die kanten ? besonders in hinbick auf loop redo kante zu alten knoten
    new_sb_state = SbState(trace_i, new_sb_config)
    data = dict()
    data['action'] = list_action.copy
    list_action.clear()

    if new_sb_state in all_states:  # todo can all_states durch closed list ersaetzt werden ?
        sb_state = all_states[all_states.index[new_sb_state]]
        ts_util.add_arc_from_to(Move(SKIP, model_label), from_ts_node, sb_state, ts_system, data)
    else:

        new_ts_node = ts.TransitionSystem.State(new_sb_state)
        new_sb_state.node = new_ts_node
        ts_system.states.add(new_ts_node)
        ts_util.add_arc_from_to(Move(SKIP, model_label), from_ts_node, new_ts_node, ts_system, data)

        new_list.append(new_sb_state)

    if len(trace) > trace_i + 1 and model_label == trace[trace_i]:
        new_sb_state = SbState(trace_i + 1, new_sb_config)

        if new_sb_state in all_states:

            sb_state = all_states[all_states.index[new_sb_state]]
            ts_util.add_arc_from_to(Move(trace[trace_i], model_label), from_ts_node, sb_state, ts_system, data)

        else:
            new_ts_node = ts.TransitionSystem.State(new_sb_state)
            new_sb_state.node = new_ts_node
            ts_system.states.add(new_ts_node)
            ts_util.add_arc_from_to(Move(trace[trace_i], model_label), from_ts_node, new_ts_node, ts_system, data)

            new_list.append(new_sb_state)


def states_to_config(open, enabled, f_enabled, closed):
    config = list()
    for i in open:
        config.insert(i.index_c, i)
    for i in enabled:
        config.insert(i.index_c, i)
    for i in f_enabled:
        config.insert(i.index_c, i)
    for i in closed:
        config.insert(i.index_c, i)
    return config


def explore_model(fire_enabled, open, enabled, f_enabled, closed, new_list, from_ts_node, trace, trace_i
                  , ts_system, all_states, list_actions):
    configs = list()

    for vertex in fire_enabled:
        temp_open = open.copy()
        temp_enabled = enabled.copy()
        temp_f_enabled = f_enabled.copy()
        temp_closed = closed.copy()
        temp_fire_enabled = set()

        temp_open.append(vertex)



        if len(vertex.children) > 0:

            v_list_actions = list_actions.copy()
            v_list_actions.extend(list_actions)
            v_list_actions.append((vertex.index_c, Action.Start))

            # sequence
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                temp_enabled.add(vertex.children[0])
                temp_closed.remove(vertex.children[0])
                for i in range(1, len(vertex.children)):
                    temp_f_enabled.add(vertex.children[i])
                temp_fire_enabled.add(vertex.children[0])

                configs.append(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed, new_list,
                                  from_ts_node, trace, trace_i
                                  , ts_system, all_states, v_list_actions))

            # xor
            elif vertex.operator is pt_opt.Operator.XOR:
                for child in vertex.children:
                    temp_enabled = enabled.copy()
                    temp_closed = closed.copy()

                    temp_enabled.add(child)
                    temp_closed.remove(child)

                    temp_fire_enabled.add(child)

                    configs.append(
                        explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed, new_list,
                                      from_ts_node, trace, trace_i
                                      , ts_system, all_states, v_list_actions))


            # parallel
            elif vertex.operator is pt_opt.Operator.PARALLEL:
                for child in vertex.children:
                    temp_enabled.add(child)
                    temp_closed.remove(child)
                    temp_fire_enabled.add(child)

                configs.append(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed, new_list,
                                  from_ts_node, trace, trace_i
                                  , ts_system, all_states, v_list_actions))


            # loop
            elif vertex.operator is pt_opt.Operator.LOOP:

                if len(vertex.children) != 3:
                    raise ValueError("Loop requires exact 3 children!")

                temp_enabled.add(vertex.children[0])
                temp_closed.remove(vertex.children[0])

                temp_fire_enabled.add(vertex.children[0])

                configs.append(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed, new_list,
                                  from_ts_node, trace, trace_i
                                  , ts_system, all_states, v_list_actions))

        else:
            new_sb_config = states_to_config(temp_open, temp_enabled, temp_f_enabled, temp_closed)

            add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, vertex.label, trace,
                           trace_i, list_actions)

            close(vertex, temp_enabled, temp_open, temp_closed, list_actions)

            configs.append((temp_open, temp_enabled, temp_enabled, temp_closed, list_actions))

            # todo add CCCC as closing node (espacially to add the close action to this node)

    return configs


def close(vertex, enabled, open, closed, list_actions):

    if vertex.operator is not None:
        list_actions.append((vertex.index_c, Action.CLOSE))
    open.remove(vertex)
    closed.add(vertex)
    process_closed(vertex, enabled, open, closed)


def process_closed(closed_node, enabled, open, closed):
    # todo wie speichere ich den node wohin die transition beim redo hingeht oder auch nicht ?
    # todo alternative einfach 1 kind auf enabled. kanten werden nur noch mal gezogen und heurisik vehindert
    # todo endlosschliefe
    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node):
            close(vertex, enabled, open, closed)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                enable = vertex.children[vertex.children.index(closed_node) + 1]  # zwischen
            elif vertex.operator is pt_opt.Operator.LOOP:
                if vertex.children.index(closed_node) == 0:
                    enable = vertex.children[1]
                    enable.add(vertex.children[2])
                else:
                    enable = vertex.children[0]
            if enable is not None:
                enabled.add(enable)


def should_close(vertex, closed, child):
    if vertex.children is None:
        return True
    elif vertex.operator is pt_opt.Operator.LOOP or vertex.operator is pt_opt.Operator.SEQUENCE:
        return vertex.children.index(child) == len(vertex.children) - 1
    else:
        return set(vertex.children) <= closed


def explore_log(pt, new_list, from_ts_node, all_states, ts_system, trace, trace_i, list_action):
    sb_new_node = SbState(trace_i + 1, from_ts_node.name.model)
    data = dict()
    data['action'] = list_action.copy

    if len(trace) != trace_i:
        if sb_new_node in all_states:
            sb_state = all_states[all_states.index[sb_new_node]]
            ts_util.add_arc_from_to(Move(trace[trace_i], SKIP), from_ts_node, sb_state, ts_system, data)
        else:
            ts_new_node = ts.TransitionSystem.State(sb_new_node)
            sb_new_node.node = ts_new_node
            ts_util.add_arc_from_to(Move(trace[trace_i], SKIP), from_ts_node, ts_new_node, ts_system, data)

            new_list.append(sb_new_node)

    return None

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
import random

SKIP = ">>"
TAU = '\u03C4'


class Action(Enum):
    # open prcess tree
    START = 1
    # close prcess tree
    CLOSE = 2


def execute(pt, trace):
    # todo heurisik einbauen
    #
    return None


def add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, model_label, trace, trace_i,
                   list_action):
    new_sb_state = SbState(trace_i, new_sb_config)
    data = dict()
    data['action'] = list_action.copy
    list_action.clear()

    if new_sb_state in all_states:  # todo can das durch closed list ersaetzt werden ?
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


def explore_model(open, enabled, f_enabled, closed, new_list, closed_list, from_ts_node, trace, trace_i
                  , ts_system, all_states, list_actions):
    configs = list()

    for vertex in enabled:
        temp_open = open.copy()
        temp_enabled = enabled.copy()
        temp_f_enabled = f_enabled.copy()
        temp_closed = closed.copy()

        temp_open.append(vertex)

        v_list_actions = list_actions.copy()
        v_list_actions.extend(list_actions)
        v_list_actions.append((vertex.index_c, Action.Start))

        if len(vertex.children) > 0:

            # sequence
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                temp_enabled.add(vertex.children[0])
                temp_closed.remove(vertex.children[0])
                for i in range(1, len(vertex.children)):
                    temp_f_enabled.add(vertex.children[i])

                configs.append((temp_open, temp_enabled, temp_f_enabled, temp_closed, v_list_actions))
            # xor
            elif vertex.operator is pt_opt.Operator.XOR:
                for child in vertex.children:
                    temp_enabled = enabled.copy()
                    temp_closed = closed.copy()

                    temp_enabled.add(child)
                    temp_closed.remove(child)

                    configs.append((temp_open, temp_enabled, temp_f_enabled, temp_closed, v_list_actions))

            # parallel
            elif vertex.operator is pt_opt.Operator.PARALLEL:
                for child in vertex.children:
                    temp_enabled.add(child)
                    temp_closed.remove(child)

                configs.append((temp_open, temp_enabled, temp_f_enabled, temp_closed, v_list_actions))

            # loop
            elif vertex.operator is pt_opt.Operator.LOOP:

                if len(vertex.children) != 3:
                    raise ValueError("Loop requires exact 3 children!")

                temp_enabled.add(vertex.children[0])
                temp_closed.remove(vertex.children[0])
                configs.append((temp_open, temp_enabled, temp_f_enabled, temp_closed, v_list_actions))


        else:
            new_sb_config = states_to_config(temp_open, temp_enabled, temp_f_enabled, temp_closed)

            add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, vertex.label, trace,
                           trace_i, list_actions)

            close(vertex, temp_enabled, temp_open, temp_closed, list_actions)

            configs.append((temp_open, temp_enabled, temp_enabled, temp_closed, v_list_actions))

            # todo add CCCC as closing node (espacially to add the close action to this node)

    return configs


def close(vertex, enabled, open, closed, list_actions):
    if vertex.operator is not None:
        list_actions.append((vertex.index_c, Action.CLOSE))
    open.remove(vertex)
    closed.add(vertex)
    process_closed(vertex, enabled, open, closed)


def process_closed(closed_node, enabled, open, closed):
    # todo wie speichere ich den node wohin die transition beim redo hingeht
    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node):
            close(vertex, enabled, open, closed)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                enable = vertex.children[vertex.children.index(closed_node) + 1]
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


def explore_log():
    return None


def explore_sync():
    return None

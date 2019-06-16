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
from pm4py.objects.process_tree.alignment import apply_cost_function_ts_node_outgoing as apply_cost_function
from pm4py.objects.process_tree import alignment as a_stern
import random
from pm4py.objects.process_tree.semantics import populate_closed as populate_closed
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


def calculate_h(node):
    # toDO implement heuristic
    return 0


def update_node_key(heap, node, value):
    copy_heap = []
    while len(copy_heap) != 0:
        if node[0].name == heapq.heappop(heap)[2][0].name:
            global counter
            heapq.heappush(copy_heap, (value, counter, node))
            counter += 1
        else:
            heapq.heappush(copy_heap, node)
    heap = copy_heap
    return heap


def is_node_in_heap(heap, node):
    copy_heap = heap.copy()
    while len(copy_heap) != 0:
        if node == heapq.heappop(copy_heap)[2][0]:
            return True
    return False


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
    init_sb_node._node = init_ts_node
    ts_system.states.add(init_ts_node)

    enabled, open, closed, f_enabled = set(), set(), set(), set()
    enabled.add(pt)
    populate_closed(pt.children, closed)

    config = (init_ts_node, open, enabled, f_enabled, closed, list())

    heapq.heappush(open_list, (0, counter, config))
    counter += 1
    all_states = [init_sb_node]

    goal_sb = SbState(len(trace), goal_config, None)
    trace_i = 0

    init_ts_node.data['g'] = 0
    init_ts_node.data['h'] = 0
    init_ts_node.data['f'] = 0
    init_ts_node.data['predecessor'] = None

    while not len(open_list) == 0:
        current_node = heapq.heappop(open_list)[2]

        # path found
        if current_node[0].name == goal_sb:
            print('path found')
            graph = visual_ts.visualize(ts_system)
            visual_ts_factory.view(graph)

            return goal_sb

        closed_list.add(current_node[0].name)

        new_listt = list()

        # expand node

        # 0 = ts_node 1 = open, 2 = enabled , 3 = f_enabled, 4 = closed . 5 list_action
        configs = explore_model(current_node[2], current_node[1], current_node[2], current_node[3], current_node[4]
                                , new_listt, current_node[0], trace, current_node[0].name.log
                                , ts_system, all_states, current_node[5])

        if len(trace) > current_node[0].name.log:

            log_config_node = explore_log(new_listt, current_node[0], all_states, ts_system, trace,
                                          current_node[0].name.log, current_node[5])
            if log_config_node is not None:
                configs.append(
                    (log_config_node, current_node[1], current_node[2], current_node[3], current_node[4], list()))
        print('--counter', counter)
        print('configs', configs)
        print('openList', open_list)
        print('closedList', closed_list)

        # graph = visual_ts.visualize(ts_system)
        # visual_ts_factory.view(graph)

        # heurisik to the explored nodes

        apply_cost_function(current_node[0], 10, 10, 1, 0)

        for config in configs:
            edge = None
            successor = config[0]
            for incoming in successor.incoming:
                if incoming.from_state.name == current_node[0].name:
                    edge = incoming


            if successor.name in closed_list:
                continue

            new_g = current_node[0].data.get('g') + edge.data.get('cost')
            if is_node_in_heap(open_list, successor.name):
                print('-*- open list', open_list, 'current', current_node[0].name, 'succname', successor.name, 'data',
                      successor.data)

            if is_node_in_heap(open_list, successor.name) and new_g >= successor.data.get('g'):
                continue

            successor.data['predecessor'] = current_node[0]
            successor.data['g'] = new_g

            f = new_g + calculate_h(successor)

            if is_node_in_heap(open_list, successor.name):
                open_list = update_node_key(open_list, config, f)
            else:
                heapq.heappush(open_list, (f, counter, config))

            counter += 1




    # no Path found
    print('no path found')

    graph = visual_ts.visualize(ts_system)
    visual_ts_factory.view(graph)
    return 0


def add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, model_label, trace, trace_i,
                   list_action):
    # todo: actions in die kanten ? besonders in hinbick auf loop redo kante zu alten knoten
    new_sb_state = SbState(trace_i, new_sb_config)
    data = dict()
    data['action'] = list_action.copy
    list_action.clear()
    list_new_nodes = list()


    if new_sb_state in all_states:  # todo can all_states durch closed list ersaetzt werden ?
        sb_state = all_states[all_states.index(new_sb_state)]
        ts_util.add_arc_from_to(Move(SKIP, model_label), from_ts_node, sb_state.node, ts_system, data)
        new_ts_node = sb_state.node  # todo das zurüclgeben ?
    else:

        new_ts_node = ts.TransitionSystem.State(new_sb_state)
        new_sb_state._node = new_ts_node
        ts_system.states.add(new_ts_node)
        ts_util.add_arc_from_to(Move(SKIP, model_label), from_ts_node, new_ts_node, ts_system, data)
        all_states.append(new_sb_state)

    list_new_nodes.append(new_ts_node)

    if len(trace) > trace_i and model_label == trace[trace_i]:
        new_sb_state = SbState(trace_i + 1, new_sb_config)

        if new_sb_state in all_states:

            sb_state = all_states[all_states.index(new_sb_state)]
            ts_util.add_arc_from_to(Move(trace[trace_i], model_label), from_ts_node, sb_state.node, ts_system, data)
            new_ts_node = sb_state.node  # todo das zurüclgeben ?
        else:
            new_ts_node = ts.TransitionSystem.State(new_sb_state)
            new_sb_state._node = new_ts_node
            ts_system.states.add(new_ts_node)
            ts_util.add_arc_from_to(Move(trace[trace_i], model_label), from_ts_node, new_ts_node, ts_system, data)
            all_states.append(new_sb_state)

        list_new_nodes.append(new_ts_node)

    return list_new_nodes


def states_to_config(open, enabled, f_enabled, closed):

    config = list()

    all = list()
    all.extend(open)
    all.extend(enabled)
    all.extend(f_enabled)
    all.extend(closed)

    all.sort(key=lambda r: r.index_c)

    for i in all:
        if i in open:
            config.append(pt_st.State.OPEN)
        elif i in enabled:
            config.append(pt_st.State.ENABLED)
        elif i in f_enabled:
            config.append(pt_st.State.FUTURE_ENABLED)
        elif i in closed:
            config.append(pt_st.State.CLOSED)
        else:
            raise ValueError("i has no state")

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

        temp_open.add(vertex)
        temp_enabled.remove(vertex)

        if len(vertex.children) > 0:

            v_list_actions = list_actions.copy()
            v_list_actions.extend(list_actions)
            v_list_actions.append((vertex.index_c, Action.START))

            # sequence
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                temp_enabled.add(vertex.children[0])
                temp_closed.remove(vertex.children[0])
                for i in range(1, len(vertex.children)):
                    temp_f_enabled.add(vertex.children[i])
                    temp_closed.remove(vertex.children[i])
                temp_fire_enabled.add(vertex.children[0])

                configs.extend(
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

                    configs.extend(
                        explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed, new_list,
                                      from_ts_node, trace, trace_i
                                      , ts_system, all_states, v_list_actions))


            # parallel
            elif vertex.operator is pt_opt.Operator.PARALLEL:
                for child in vertex.children:
                    temp_enabled.add(child)
                    temp_closed.remove(child)
                    temp_fire_enabled.add(child)

                configs.extend(
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

                configs.extend(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed, new_list,
                                  from_ts_node, trace, trace_i
                                  , ts_system, all_states, v_list_actions))

        else:
            new_sb_config = states_to_config(temp_open, temp_enabled, temp_f_enabled, temp_closed)

            new_ts_nodes = add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, vertex.label,
                                          trace,
                                          trace_i, list_actions)

            close(vertex, temp_enabled, temp_open, temp_closed, temp_f_enabled, list_actions)

            for ts_node in new_ts_nodes:
                configs.append((ts_node, temp_open, temp_enabled, temp_f_enabled, temp_closed, list_actions))

            # todo add CCCC as closing node (espacially to add the close action to this node)

                if (len(temp_open) + len(temp_enabled) + len(temp_f_enabled)) == 0:
                    new_sb_config = states_to_config(temp_open, temp_enabled, temp_f_enabled, temp_closed)

                    new_ts_nodes = add_node_to_ts(all_states, new_list, ts_node, new_sb_config, ts_system,
                                                  SKIP,
                                                  trace,
                                                  ts_node.name.log, list_actions)


    return configs


def close(vertex, enabled, open, closed, f_enabled, list_actions):

    if vertex.operator is not None:
        list_actions.append((vertex.index_c, Action.CLOSE))
    open.remove(vertex)
    closed.add(vertex)
    process_closed(vertex, enabled, open, closed, f_enabled, list_actions)


def process_closed(closed_node, enabled, open, closed, f_enabled, list_actions):
    # todo wie speichere ich den node wohin die transition beim redo hingeht oder auch nicht ?
    # todo alternative einfach 1 kind auf enabled. kanten werden nur noch mal gezogen und heurisik vehindert
    # todo endlosschliefe
    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node):
            close(vertex, enabled, open, closed, f_enabled, list_actions)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                enable = vertex.children[vertex.children.index(closed_node) + 1]
                f_enabled.remove(vertex.children[vertex.children.index(closed_node) + 1])
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


def explore_log(new_list, from_ts_node, all_states, ts_system, trace, trace_i, list_action):
    sb_new_node = SbState(trace_i + 1, from_ts_node.name.model)
    data = dict()
    data['action'] = list_action.copy

    if len(trace) != trace_i:
        if sb_new_node in all_states:
            ts_new_node = all_states[all_states.index(sb_new_node)].node
            ts_util.add_arc_from_to(Move(trace[trace_i], SKIP), from_ts_node, ts_new_node, ts_system, data)
            # todo auch diese kante zurckgeben ?
        else:
            ts_new_node = ts.TransitionSystem.State(sb_new_node)
            sb_new_node.node = ts_new_node
            ts_util.add_arc_from_to(Move(trace[trace_i], SKIP), from_ts_node, ts_new_node, ts_system, data)
            all_states.append(sb_new_node)
            new_list.append(ts_new_node)
            return ts_new_node
    return None

trace = list()
trace.append('a')
trace.append('b')
trace.append('c')
tree = pt_util.parse("->('a','b','c')")
execute(tree, trace)

import heapq
from enum import Enum

from pm4py.algo.conformance.alignments import utils as align_utils
from pm4py.algo.conformance.alignments.versions import state_equation_a_star  as a_star
from pm4py.objects.conversion.process_tree.versions import to_petri_net as conversion
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.util.xes import DEFAULT_NAME_KEY
from pm4py.objects.petri import incidence_matrix as inicence_m
from pm4py.objects.petri import petrinet as petrinet
from pm4py.objects.petri import synchronous_product as sync_p
from pm4py.objects.petri import utils as pnet_util
from pm4py.objects.process_tree import pt_operator as pt_opt
from pm4py.objects.process_tree import state as pt_st
from pm4py.objects.process_tree import util as pt_util
from pm4py.objects.process_tree.alignment import apply_cost_function_ts_node_outgoing as apply_cost_function
from pm4py.objects.process_tree.pt_state_space import Move as Move
from pm4py.objects.process_tree.pt_state_space import SbState as SbState
from pm4py.objects.process_tree.semantics import populate_closed as populate_closed
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.transition_system import utils as ts_util
from pm4py.visualization.transition_system import factory as visual_ts_factory
from pm4py.visualization.transition_system.util import visualize_graphviz as visual_ts
from pm4py.visualization.petrinet import factory as vi_petri
from pm4py.visualization.petrinet.common import visualize as vi_petri2
from pm4py.objects.conversion.log import factory as log_converter

SKIP = ">>"
# TAU = '\u03C4'
TAU = 'None'

# used for heap to help compare tuples
counter = 0


class Action(Enum):
    # open process tree
    START = 1
    # close process tree
    CLOSE = 2


def create_place_tree_list(subtree, subtree_marking):
    subtree_marking.append((subtree.data['start_place'], subtree.data['final_place'], subtree))

    for i in range(0, len(subtree._children)):
        child = subtree._children[i]
        create_place_tree_list(child, subtree_marking)


def get_parallel_parents(subtree, parallel_list):
    if subtree.operator == pt_opt.Operator.PARALLEL:
        parallel_list.append(subtree)
    if subtree.parent is not None:
        get_parallel_parents(subtree.parent, parallel_list)
    else:
        return parallel_list


def get_passed_children(tree_list, closed_list, open_list, passed_children):
    for tree in tree_list:
        for child in tree.children:
            if child in closed_list:
                passed_children.append(child)
            if child in open_list:
                passed_children.append(child)


def calculate_h(tree, enabled, i_trace, n, i, f, v, t, imatrx, tree_final_marking,
                closed, open):  # (tree, trace, subtree, i_trace):

    complete_marking = petrinet.Marking()
    complete_marking[t[i_trace][0]] = 1

    # print('§§enabled', enabled[1])


    if len(enabled[1]) > 0:
        for i in enabled[1]:
            complete_marking[i.data['start_place']] = 1

            parallel_parent = list()
            passed_children = list()
            get_parallel_parents(i, parallel_parent)
            get_passed_children(parallel_parent, closed, open, passed_children)
            for child in passed_children:
                complete_marking[child.data['final_place']] = 1

    else:
        complete_marking[tree.data['final_place']] = 1

    #print("complete", complete_marking)
    # gviz = vi_petri.apply(n, complete_marking, f)
    # gviz = vi_petri2.graphviz_visualization(n, debug=True)
    # vi_petri.view(gviz)

    cost_function = align_utils.construct_standard_cost_function(n, ">>")

    ini_vec, fin_vec, cost_vec = a_star.__vectorize_initial_final_cost(imatrx, complete_marking, f, cost_function)
    h, x = a_star.__compute_exact_heuristic(n, imatrx, complete_marking, cost_vec, fin_vec)
    # print('heuristic ', h)
    if h > 9000000000000:
        ValueError('Wrong Marking')
    return h


def update_node_key(heap, node, value):
    copy_heap = []
    while len(heap) != 0:
        current_node = heapq.heappop(heap)
        if node[0].name == current_node[2][0].name:
            global counter
            heapq.heappush(copy_heap, (value, counter, node))
            counter += 1
        else:
            heapq.heappush(copy_heap, current_node)
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
    init_sb_node = SbState(0, init_sb_config, pt, vertex=({pt}, {pt}))
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

    all_loop_nodes = list()

    activity_key = DEFAULT_NAME_KEY

    net1, init1, final1, trace_place_list = pnet_util.construct_trace_net_marking(trace, activity_key=activity_key)
    net, init, final = conversion.apply(pt)

    dummy_marking = petrinet.Marking()
    marking_vector_tree = list()
    create_place_tree_list(pt, marking_vector_tree)
    #marking_vector_tree.append((set(), final))

    net, i_marking, f_marking, pt_marking_list, trace_marking_list = sync_p.construct_place_aware(net, dummy_marking,
                                                                                                  final, net1,
                                                                                                  dummy_marking, final1,
                                                                                                  ">>",
                                                                                                  marking_vector_tree,
                                                                                                  trace_place_list)
    # gviz = vi_petri2.graphviz_visualization(net, debug=True)
    #vi_petri.view(gviz)

    imatrx = inicence_m.construct(net)

    visited = 1
    queued = 1
    traversed_arcs = 0

    # rename trace
    trace = list(map(lambda event: event["concept:name"], trace))

    while not len(open_list) == 0:

        top = heapq.heappop(open_list)

        current_node = top[2]
        visited += 1
        # print('-s-current', current_node[0], 'o: ', len(open_list), 'h:', top[0],'+', top[1], '              -|-', current_node)

        # print('pre openlist')
        # for o in open_list:
        #    print('                         ', o)

        # path found
        if 'end' in current_node[0].data and current_node[0].data['end'] is True:
            # print('path found')
            # graph = visual_ts.visualize(ts_system)
            # visual_ts_factory.view(graph)
            # print_path(current_node[0])

            return reconstruct_alignment(current_node[0], visited, queued, traversed_arcs)

        closed_list.add(current_node[0].name)

        loop_list = list()

        # 0 = ts_node 1 = open, 2 = enabled , 3 = f_enabled, 4 = closed . 5 list_action 6 vertex label, 7 node seen
        configs = explore_model(current_node[2], current_node[1], current_node[2], current_node[3], current_node[4]
                                , loop_list, current_node[0], trace, current_node[0].name.log
                                , ts_system, all_states, current_node[5])

        if len(trace) > current_node[0].name.log:
            log_config_node, saw_node = explore_log(loop_list, current_node[0], all_states, ts_system, trace,
                                          current_node[0].name.log, current_node[5])
            if log_config_node is not None:
                configs.append(
                    (log_config_node, current_node[1], current_node[2], current_node[3],
                     current_node[4], list(), None, saw_node))

        # print('current keys', current_node[0].data)
        # print('configs', configs)
        #print('closedList', closed_list)

        # heurisic to the explored nodes

        apply_cost_function(current_node[0], 1100, 1100, 1, 0)

        for l in all_loop_nodes:

            if l[0] == current_node[0]:
                configs.append(l[1])

        for config in configs:
            # print('---*---', len(open_list))
            # print('*config', config)
            traversed_arcs += 1

            edge = None
            successor = config[0]
            for incoming in successor.incoming:
                if incoming.from_state.name == current_node[0].name:
                    edge = incoming

            if successor.name in closed_list:
                continue

            if edge is None:
                print('// edge is none')

            new_g = current_node[0].data.get('g') + edge.data.get('cost')
            if is_node_in_heap(open_list, successor.name) and new_g >= successor.data.get('g'):
                continue

            successor.data['predecessor'] = current_node[0]
            successor.data['g'] = new_g

            #print('succ', current_node[0], '-->', successor)
            f = new_g + calculate_h(pt, successor.name.vertex, successor.name.log, net, i_marking, f_marking,
                                    pt_marking_list, trace_marking_list, imatrx, final,
                                    config[4], config[1])
            # print('heurisik', f, new_g, successor.name)

            if config[7] is True:
                open_list = update_node_key(open_list, config, f)
            elif is_node_in_heap(open_list, successor.name):  # todo can i replace this with config[7]
                open_list = update_node_key(open_list, config, f)
                if config[7] is False:
                    raise ValueError('tes')
            else:
                heapq.heappush(open_list, (f, counter, config))
                queued += 1

            counter += 1
        # print('post openList      top: ' 'h:', top[0],'+', top[1])
        # for o in open_list:
        #    print('                         ', o)

    print('no path found')

    graph = visual_ts.visualize(ts_system)
    visual_ts_factory.view(graph)
    return 0


def add_node_to_ts(all_states, new_list, from_ts_node, new_sb_config, ts_system, model_label, trace, trace_i,
                   list_action, vertex):
    # todo: actions in die kanten ? besonders in hinblick auf loop redo kante zu alten knoten
    new_sb_state = SbState(trace_i, new_sb_config, vertex=vertex)
    data = dict()
    data['action'] = list_action.copy
    list_action.clear()
    list_new_nodes = list()

    if new_sb_state in all_states:  # todo can all_states durch closed list ersetzt werden ?
        sb_state = all_states[all_states.index(new_sb_state)]
        ts_util.add_arc_from_to(Move(SKIP, model_label), from_ts_node, sb_state.node, ts_system, data)
        new_ts_node = sb_state.node
        list_new_nodes.append((new_ts_node, True))
    else:

        new_ts_node = ts.TransitionSystem.State(new_sb_state)
        new_sb_state._node = new_ts_node
        ts_system.states.add(new_ts_node)
        ts_util.add_arc_from_to(Move(SKIP, model_label), from_ts_node, new_ts_node, ts_system, data)
        all_states.append(new_sb_state)
        list_new_nodes.append((new_ts_node, False))

    if len(trace) > trace_i and model_label == trace[trace_i]:
        data = dict()
        data['action'] = list_action.copy
        new_sb_state = SbState(trace_i + 1, new_sb_config, vertex=vertex)

        if new_sb_state in all_states:

            sb_state = all_states[all_states.index(new_sb_state)]
            ts_util.add_arc_from_to(Move(trace[trace_i], model_label), from_ts_node, sb_state.node, ts_system, data)
            list_new_nodes.append((new_ts_node, True))
        else:
            new_ts_node = ts.TransitionSystem.State(new_sb_state)
            new_sb_state._node = new_ts_node
            ts_system.states.add(new_ts_node)
            ts_util.add_arc_from_to(Move(trace[trace_i], model_label), from_ts_node, new_ts_node, ts_system, data)
            all_states.append(new_sb_state)
            list_new_nodes.append((new_ts_node, False))

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


def explore_model(fire_enabled, open, enabled, f_enabled, closed, loop_config_list, from_ts_node, trace, trace_i
                  , ts_system, all_states, list_actions):
    # todo remove loop_config_node
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
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed,
                                  loop_config_list, from_ts_node, trace, trace_i,
                                  ts_system, all_states, v_list_actions))

            # xor
            elif vertex.operator is pt_opt.Operator.XOR:

                for child in vertex.children:
                    temp2_enabled = temp_enabled.copy()
                    temp_closed = closed.copy()

                    temp2_enabled.add(child)
                    temp_closed.remove(child)

                    temp_fire_enabled.add(child)

                    configs.extend(
                        explore_model(temp_fire_enabled, temp_open, temp2_enabled, temp_f_enabled.copy(), temp_closed,
                                      loop_config_list,
                                      from_ts_node, trace, trace_i, ts_system, all_states, v_list_actions.copy()))
                    temp_fire_enabled = set()

            # parallel
            elif vertex.operator is pt_opt.Operator.PARALLEL:
                for child in vertex.children:
                    temp_enabled.add(child)
                    temp_closed.remove(child)
                    temp_fire_enabled.add(child)

                configs.extend(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed,
                                  loop_config_list, from_ts_node, trace, trace_i,
                                  ts_system, all_states, v_list_actions))

            # loop
            elif vertex.operator is pt_opt.Operator.LOOP:

                if len(vertex.children) != 3:
                    raise ValueError("Loop requires exact 3 children!")

                temp_enabled.add(vertex.children[0])
                temp_closed.remove(vertex.children[0])
                temp_fire_enabled.add(vertex.children[0])

                configs.extend(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed,
                                  loop_config_list, from_ts_node, trace, trace_i,
                                  ts_system, all_states, v_list_actions))

                temp_f_enabled.add(vertex.children[1])
                temp_closed.remove(vertex.children[1])

                configs.extend(
                    explore_model(temp_fire_enabled, temp_open, temp_enabled, temp_f_enabled, temp_closed,
                                  loop_config_list, from_ts_node, trace, trace_i,
                                  ts_system, all_states, v_list_actions))

        else:

            new_sb_config = states_to_config(temp_open, temp_enabled, temp_f_enabled, temp_closed)

            new_ts_nodes = add_node_to_ts(all_states, loop_config_list, from_ts_node, new_sb_config,
                                          ts_system, vertex.label, trace, trace_i,
                                          list_actions, (enabled, temp_enabled))

            close(vertex, temp_enabled, temp_open, temp_closed, temp_f_enabled, list_actions)

            for ts_node in new_ts_nodes:
                configs.append(
                    (ts_node[0], temp_open, temp_enabled, temp_f_enabled, temp_closed,
                     list_actions, vertex.label, ts_node[1]))
                ts_node = ts_node[0]
            # todo add CCCC as closing node (espacially to add the close action to this node)

                if (len(temp_open) + len(temp_enabled) + len(temp_f_enabled)) == 0:

                    ts_node.data['end'] = False

                    if ts_node.name.log == len(trace):
                        ts_node.data['end'] = True
            if len(configs) == 0 and len(new_ts_nodes) != 0:
                graph = visual_ts.visualize(ts_system)
                visual_ts_factory.view(graph)
                raise ValueError('error')

    return configs


def close(vertex, enabled, open, closed, f_enabled, list_actions):

    if vertex.operator is not None:
        list_actions.append((vertex.index_c, Action.CLOSE))
    open.remove(vertex)
    closed.add(vertex)
    return process_closed(vertex, enabled, open, closed, f_enabled, list_actions)


def process_closed(closed_node, enabled, open, closed, f_enabled, list_actions):

    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node):
            return close(vertex, enabled, open, closed, f_enabled, list_actions)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE:
                enable = vertex.children[vertex.children.index(closed_node) + 1]
                f_enabled.remove(vertex.children[vertex.children.index(closed_node) + 1])
            elif vertex.operator is pt_opt.Operator.LOOP:
                if vertex.children.index(closed_node) == 0:

                    if vertex.children[1] in f_enabled:
                        enable = vertex.children[1]
                        f_enabled.remove(vertex.children[1])
                    else:
                        enable = vertex.children[2]
                        closed.remove(vertex.children[2])

                elif vertex.children.index(closed_node) == 1:
                    enabled.add(vertex)
                    open.remove(vertex)
                else:
                    print('wtf')
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
    sb_new_node = SbState(trace_i + 1, from_ts_node.name.model, vertex=from_ts_node.name.vertex)
    data = dict()
    data['action'] = list_action.copy
    list_action.clear()

    if len(trace) != trace_i:
        if sb_new_node in all_states:
            ts_new_node = all_states[all_states.index(sb_new_node)].node
            ts_util.add_arc_from_to(Move(trace[trace_i], SKIP), from_ts_node, ts_new_node, ts_system, data)
            return ts_new_node, True
        else:
            ts_new_node = ts.TransitionSystem.State(sb_new_node)
            sb_new_node.node = ts_new_node
            ts_util.add_arc_from_to(Move(trace[trace_i], SKIP), from_ts_node, ts_new_node, ts_system, data)
            all_states.append(sb_new_node)
            new_list.append(ts_new_node)

            if "end" in from_ts_node.data:
                if trace_i + 1 == len(trace):
                    ts_new_node.data["end"] = True
                else:
                    ts_new_node.data["end"] = False

            return ts_new_node, False

    return None


def print_path(node):
    answer = str(node) + "   <-"

    while node.data['predecessor'] is not None:

        for edge in node.incoming:
            if edge.from_state == node.data['predecessor']:
                answer += str(edge) + "--  "

        answer += str(node.data['predecessor']) + "   <-"

        node = node.data['predecessor']
    print(answer)
    return answer


def reconstruct_alignment(node, visited, queued, traversed):
    answer = list()
    g = node.data['g']

    while node.data['predecessor'] is not None:

        for edge in node.incoming:
            if edge.from_state == node.data['predecessor']:
                answer.append(edge)

        node = node.data['predecessor']
    answer.reverse()
    return {'alignment': answer, 'cost': g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed}


def apply():
    log = xes_importer.import_log("/Users/Ralf/PycharmProjects/pm4py-source/tests/input_data/ababac.xes")
    trace = list()
    trace.append('a')
    trace.append('b')
    trace.append('a')
    trace.append('b')
    trace.append('a')
    trace.append('c')
    trace = log_converter.apply(log, {}, log_converter.TO_EVENT_LOG)
    print(trace)
    tree = pt_util.parse("*('a','b','c')")
    tree = pt_util.parse("->('a','b','a','b','a','c')")
    align = list(map(lambda trace: execute(tree, trace), log))
    # print('result', execute(tree, log))


def race(tree, log):
    align = list()  # list(map(lambda trace: execute(tree, trace), log))
    for i in log:
        #   print(i)
        align.append(execute(tree, i))
    return align


'''   
try:
    import faulthandler

    faulthandler.enable()
    print('Faulthandler enabled')
except Exception:
    print('Could not enable faulthandler')

#apply()

'''

from pm4py.objects.process_tree import pt_state_space
from pm4py.objects.process_tree import util as pt_util
import heapq
from pm4py.visualization.transition_system.util import visualize_graphviz as visual_ts
from pm4py.visualization.transition_system import factory as visual_ts_factory

SKIP = ">>"
# TAU = '\u03C4'
TAU = None

# used for heap to help compare tuples
counter = 0


def apply_cost_function_ts_system(ts_system, lm_cost, mm_cost, tau_cost, sync_cost):

    for edge in ts_system.transitions:
        # tau
        if edge.name.log is TAU or edge.name.model is TAU:
            edge.data['cost'] = tau_cost
        # model move
        elif edge.name.log is SKIP:
            edge.data['cost'] = mm_cost
        # log move
        elif edge.name.model is SKIP:
            edge.data['cost'] = lm_cost
        # sync move
        else:
            edge.data['cost'] = sync_cost


def apply_cost_function_ts_node_outgoing(ts_node, lm_cost, mm_cost, tau_cost, sync_cost):
    for edge in ts_node.outgoing:
        # tau
        if edge.name.log == TAU or edge.name.model == TAU:
            edge.data['cost'] = tau_cost
        # model move
        elif edge.name.log == SKIP:
            edge.data['cost'] = mm_cost
        # log move
        elif edge.name.model == SKIP:
            edge.data['cost'] = lm_cost
        # sync move
        else:
            edge.data['cost'] = sync_cost


def a_star_search(ts_system, root, goal):

    open_list = []
    closed_list = set()

    global counter

    heapq.heappush(open_list, (0, counter, root))
    counter += 1

    while not len(open_list) == 0:

        current_node = heapq.heappop(open_list)

        # path found
        if current_node[2] == goal:
            return goal

        closed_list.add(current_node[2])

        expand_node(current_node[2], open_list, closed_list)

    # no Path found
    print('no path found')
    return 0


def expand_node(c_node, open_list, closed_list):

    for outgoing in c_node.outgoing:
        successor = outgoing.to_state

        if successor in closed_list:
            continue

        new_g = c_node.data.get('g') + outgoing.data.get('cost')

        if is_node_in_heap(open_list, successor) and new_g >= successor.data.get('g'):
            continue

        successor.data['predecessor'] = c_node
        successor.data['g'] = new_g

        f = new_g + calculate_h(successor)

        global counter
        if is_node_in_heap(open_list, successor):
            open_list = update_node_key(open_list, successor, f)
        else:
            heapq.heappush(open_list, (f, counter, successor))

        counter += 1


def is_node_in_heap(heap, node):
    copy_heap = heap.copy()
    while len(copy_heap) != 0:
        if node == heapq.heappop(copy_heap)[2]:
            return True
    return False


def update_node_key(heap, node, value):
    copy_heap = []
    while len(copy_heap) != 0:
        if node == heapq.heappop(heap)[2]:
            global counter
            heapq.heappush(copy_heap, (value,counter,node))
            counter += 1
        else:
            heapq.heappush(copy_heap, node)
    heap = copy_heap
    return heap


def calculate_h(node):
    # toDO implement heuristic
    return 0


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


'''
trace = list()
# for i in range(0,3):
#    trace.append('a')
trace.append('a')
#trace.append('b')
# trace.append('c')

# tree = pt_util.parse("->( X('a','b'), 'c' ")
# tree =  pt_util.parse("->(*('a','d'),'b','c')")
# tree = pt_util.parse("+('a','b','c')")
#tree = pt_util.parse("+('a','b')")
tree = pt_util.parse("*('a','b','c')")

ts_system = pt_state_space.execute(tree, trace)

apply_cost_function(ts_system, 100, 100, 1, 0)

# get root and goal node
root = None
goal = None
for node in ts_system.states:
    if len(node.incoming) == 0:
        root = node
    if len(node.outgoing) == 0:
        goal = node

root.data['g'] = 0
root.data['h'] = 0
root.data['f'] = 0
root.data['predecessor'] = None
if a_star_search(ts_system, root, goal) != 0:
    print_path(goal)

graph = visual_ts.visualize(ts_system)
visual_ts_factory.view(graph)

#book example
#tree = pt_util.parse("->('As', X( ->(+('Fa', *( ->('SSo', 'Ro'), 'Co')) , X(->('Ao', 'Aan'), ->('Do', 'Da2')))) ,'Da1') ,'Af')")
'''

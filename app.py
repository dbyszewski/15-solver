import argparse
import time
import copy


# globals
ORDER = ['L', 'R', 'U', 'D']
MAX_DEPTH = 20

SOLVED_PUZZLE = []
START_PUZZLE = []
FIELD_0 = {}


class Node:
    def __init__(self, puzzle, parent_node, last_move, moves, score=0):
        self.puzzle = puzzle
        self.parent_node = parent_node
        self.children = {}
        self.moves = moves.copy()
        self.moves.append(last_move)
        self.possible_moves = ORDER.copy()
        self.score = score

    def copy_puzzle(self):
        return copy.deepcopy(self.puzzle)

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return len(self.moves) + self.score

    def create_child(self, puzzle_after_move, move):
        child = Node(puzzle_after_move, self, move, self.moves)
        self.children[move] = child

    def make_move(self, move):
        y1, x1 = self.find_field_0()
        y2, x2 = y1, x1
        if move == 'L':
            x2 = x2 - 1
        elif move == 'R':
            x2 = x2 + 1
        elif move == 'U':
            y2 = y2 - 1
        elif move == 'D':
            y2 = y2 + 1

        swap_puzzle = self.copy_puzzle()
        swap_puzzle[y1][x1], swap_puzzle[y2][x2] = swap_puzzle[y2][x2], swap_puzzle[y1][x1]
        child = Node(swap_puzzle, self, move, self.moves)
        self.children[move] = child

    def remove_edge_directions(self):
        y, x = self.find_field_0()
        if x == 0:
            self.possible_moves.remove('L')
        if x == len(SOLVED_PUZZLE[0])-1:
            self.possible_moves.remove('R')
        if y == 0:
            self.possible_moves.remove('U')
        if y == len(SOLVED_PUZZLE)-1:
            self.possible_moves.remove('D')

    def remove_last_visited_direction(self):
        if self.moves[-1] == 'R':
            self.possible_moves.remove('L')
        elif self.moves[-1] == 'L':
            self.possible_moves.remove('R')
        elif self.moves[-1] == 'U':
            self.possible_moves.remove('D')
        elif self.moves[-1] == 'D':
            self.possible_moves.remove('U')

    def find_field_0(self):
        return find_indexes(self.puzzle, 0)

    def is_solved(self):
        if self.puzzle == SOLVED_PUZZLE:
            return True


# Functions
def find_indexes(puzzle, value):
    for j in range(len(puzzle)):
        if value in puzzle[j]:
            return j, puzzle[j].index(value)


def create_solution_files(s_time, mvs, p_nodes, v_nodes, d_level, solution_file, statistic_file):

    file = open(solution_file, 'w+')
    if mvs != -1:
        mvs.remove(mvs[0])
        solution_length = len(mvs)
        file.write(str(solution_length))
        file.write('\n')
        file.write(''.join(mvs))
    else:
        solution_length = -1
        file.write(str(solution_length))
    file.close()

    file = open(statistic_file, 'w+')
    file.write(str(solution_length))
    file.write('\n')
    file.write(str(v_nodes))
    file.write('\n')
    file.write(str(p_nodes))
    file.write('\n')
    file.write(str(d_level))
    file.write('\n')
    file.write(str(round((time.time() - s_time) * 1000, 3)))
    file.close()


# Algoritms
def bfs():
    processed_nodes, visited_nodes = 1, 1
    current_node = Node(START_PUZZLE, None, None, [])
    current_node.remove_edge_directions()
    queue = []
    counter = 0
    while True:
        counter += 1
        if current_node.is_solved():
            return current_node.moves, processed_nodes, visited_nodes, len(current_node.moves) - 1
        else:
            if current_node.parent_node:
                current_node.remove_edge_directions()
                current_node.remove_last_visited_direction()
            for move in current_node.possible_moves:
                processed_nodes += 1
                current_node.make_move(move)
                child_node = current_node.children[move]
                queue.append(child_node)
            queue.pop(0)
            current_node = queue[0]
            visited_nodes += 1


def dfs():
    processed_nodes, visited_nodes, depth = 1, 1, 0
    current_node = Node(START_PUZZLE, None, None, [])
    moved_back = False
    max_depth_visited = False
    current_node.remove_edge_directions()

    while True:
        if current_node.is_solved():
            if max_depth_visited:
                depth = MAX_DEPTH
            else:
                depth = len(current_node.moves) - 1
            return current_node.moves, processed_nodes, visited_nodes, depth
        elif len(current_node.moves) == MAX_DEPTH:
            current_node = current_node.parent_node
            moved_back, max_depth_visited = True, True
        elif len(current_node.possible_moves) != 0:
            if current_node.parent_node and not moved_back:
                current_node.remove_edge_directions()
                current_node.remove_last_visited_direction()
            if len(current_node.possible_moves) != 0:
                move = current_node.possible_moves.pop(0)
                current_node.make_move(move)
                current_node = current_node.children[move]
                moved_back = False
                visited_nodes += 1
                processed_nodes += 1
            else:
                if current_node.parent_node is None:
                    return -1, processed_nodes, visited_nodes, depth
                else:
                    current_node = current_node.parent_node
                    moved_back = True
        else:
            if current_node.parent_node is None:
                return -1, processed_nodes, visited_nodes, depth
            else:
                current_node = current_node.parent_node
                moved_back = True


def astr(heuristic):
    visited_nodes, processed_nodes = 0, 1
    nodes_pool = []
    nodes_history = []
    start_node = Node(START_PUZZLE, None, None, [])
    nodes_pool.append(start_node)
    nodes_history.append(start_node.copy_puzzle())

    while len(nodes_pool):
        visited_nodes += 1
        current_node = nodes_pool.pop(0)
        if current_node.is_solved():
            return current_node.moves, processed_nodes, visited_nodes, len(current_node.moves) - 1
        else:
            current_node.remove_edge_directions()
            for move in current_node.possible_moves:
                processed_nodes += 1
                current_node.make_move(move)
                children_node = current_node.children[move]
                if children_node.copy_puzzle in nodes_history:
                    continue
                children_node.set_score(heuristic(children_node.puzzle))
                lo = 0
                hi = len(nodes_pool)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if children_node.get_score() < nodes_pool[mid].get_score():
                        hi = mid
                    else:
                        lo = mid + 1
                nodes_pool.insert(lo, children_node)


# Heuristics
def manh(current_board):
    distance = 0
    for y1 in range(len(current_board)):
        for x1 in range(len(current_board)):
            y2, x2 = find_indexes(SOLVED_PUZZLE, current_board[y1][x1])
            distance += abs(y1 - y2) + abs(x1 - x2)
    return distance


def hamm(current_board):
    distance = 0
    for y in range(len(current_board)):
        for x in range(len(current_board)):
            if current_board[y][x] != SOLVED_PUZZLE[y][x]:
                distance += 1
    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="algorithm, order, source_file, solution_file, statistics_file")
    parser.add_argument('algorithm')
    parser.add_argument('order')
    parser.add_argument('source_file')
    parser.add_argument('solution_file')
    parser.add_argument('statistic_file')
    args = parser.parse_args()

    with open(args.source_file) as board:
        is_first_line_flag = True
        for line in board:
            if is_first_line_flag:
                solved_1d = list(range(1, int(line.split()[0]) ** 2))
                solved_1d.append(0)
                for i in range(int(line.split()[0])):
                    SOLVED_PUZZLE.append(solved_1d[i * 4:(i + 1) * 4])
                is_first_line_flag = False
            else:
                START_PUZZLE.append([int(number) for number in line.split()])

    algorithm = globals()[args.algorithm]
    if args.algorithm == 'dfs' or args.algorithm == 'bfs':
        ORDER = [order for order in args.order]
        start_time = time.time()
        moves, processed_nodes, visited_nodes, depth_level = algorithm()
    else:
        heuristic = globals()[args.order]
        start_time = time.time()
        moves, processed_nodes, visited_nodes, depth_level = algorithm(heuristic)

    create_solution_files(start_time, moves, processed_nodes, visited_nodes, depth_level, args.solution_file, args.statistic_file)


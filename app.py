import argparse
import random
import time
import copy


# GLOBAL VARIABLES
SOLVED_BOARD = []
START_BOARD = []
FIELD_0 = {}
ORDER = ['L', 'R', 'U', 'D']
MAX_DEPTH = 20


class Node:
    def __init__(self, current_puzzle, parent, last_move, moves, score=0):
        self.puzzle = current_puzzle
        # children['L'] - dziecko po ruchu w lewo, R prawo itd jakby cos XD
        self.children = {}
        # sekwencja ruch : błąd jaki po tym ruchu bedzie
        self.errors = {}
        self.parent = parent
        self.last = last_move
        # A to jest jakie ruchy do tego doprowadzily
        self.moves = moves.copy()
        self.moves.append(last_move)
        # Kolejka do odwiedzenia
        self.possible_moves = ORDER.copy()
        self.score = score

    def copy_puzzle(self):
        return copy.deepcopy(self.puzzle)

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return len(self.moves) + self.score

    def create_child(self, board_after_move, move):
        child = Node(board_after_move, self, move, self.moves)
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

        swap_puzzle = []
        for row in self.puzzle:
            swap_puzzle.append(row.copy())
        swap_puzzle[y1][x1], swap_puzzle[y2][x2] = swap_puzzle[y2][x2], swap_puzzle[y1][x1]

        child = Node(swap_puzzle, self, move, self.moves)
        self.children[move] = child

    def remove_edge_directions(self):
        y, x = self.find_field_0()
        if x == 0:
            self.possible_moves.remove('L')
        if x == len(SOLVED_BOARD[0])-1:
            self.possible_moves.remove('R')
        if y == 0:
            self.possible_moves.remove('U')
        if y == len(SOLVED_BOARD)-1:
            self.possible_moves.remove('D')

    def remove_last_visited_direction(self):
        if self.last == 'R':
            self.possible_moves.remove('L')
        elif self.last == 'L':
            self.possible_moves.remove('R')
        elif self.last == 'U':
            self.possible_moves.remove('D')
        elif self.last == 'D':
            self.possible_moves.remove('U')

    def find_field_0(self):
        return find_indexes(self.puzzle, 0)

    def is_solved(self):
        if self.puzzle == SOLVED_BOARD:
            return True


def find_indexes(puzzle, value):
    for j in range(len(puzzle)):
        if value in puzzle[j]:
            return j, puzzle[j].index(value)


def prepare_solution(data, solution_file, statistic_file, s_time):
    moves, processed_nodes, visited_nodes, depth_level = data

    file = open(solution_file, 'w+')
    if moves != -1:
        moves.remove(moves[0])
        solution_length = len(moves)
        file.write(str(solution_length))
        file.write('\n')
        file.write(''.join(moves))
    else:
        solution_length = -1
        file.write(str(solution_length))
    file.close()

    file = open(statistic_file, 'w+')
    file.write(str(solution_length))
    file.write('\n')
    file.write(str(visited_nodes))
    file.write('\n')
    file.write(str(processed_nodes))
    file.write('\n')
    file.write(str(depth_level))
    file.write('\n')
    file.write(str(round((time.time() - s_time) * 1000, 3)))
    file.close()


def find_0(board):
    for j in range(len(board)):
        if 0 in board[j]:
            FIELD_0['row'] = j
            FIELD_0['column'] = board[j].index(0)


# Algorithms
def dfs(start_time):
    processed_nodes, visited_nodes, depth = 1, 1, 0
    current_node = Node(START_BOARD, None, None, [])
    moved_back = False
    max_depth_visited = False
    current_node.remove_edge_directions()
    return -1, processed_nodes, visited_nodes, depth

    while True:
        if current_node.is_solved():
            if max_depth_visited:
                depth = MAX_DEPTH
            else:
                depth = len(current_node.moves) - 1
            return current_node.moves, processed_nodes, visited_nodes, depth
        elif len(current_node.moves) == MAX_DEPTH:
            current_node = current_node.parent
            moved_back, max_depth_visited = True, True
        elif len(current_node.possible_moves) != 0:
            if current_node.parent and not moved_back:
                current_node.remove_edge_directions()
                current_node.remove_last_visited_direction()
            if len(current_node.possible_moves) != 0:
                move = current_node.possible_moves[0]
                current_node.make_move(move)
                current_node.possible_moves.remove(move)
                current_node = current_node.children[move]
                moved_back = False
                visited_nodes += 1
                processed_nodes += 1
            else:
                if current_node.last is None or time.time() - start_time > MAX_DEPTH:
                    return -1, processed_nodes, visited_nodes, depth
                else:
                    current_node = current_node.parent
                    moved_back = True
        else:
            if current_node.last is None or time.time() - start_time > MAX_DEPTH:
                return -1, processed_nodes, visited_nodes, depth
            else:
                current_node = current_node.parent
                moved_back = True


def bfs(start_time):
    processed_nodes, visited_nodes = 1, 1
    current_node = Node(START_BOARD, None, None, [])
    current_node.remove_edge_directions()
    queue = []
    counter = 0
    while True:
        counter += 1
        if time.time() - start_time > MAX_DEPTH:
            return -1, processed_nodes, visited_nodes, len(current_node.moves) - 1
        if current_node.is_solved():
            return current_node.moves, processed_nodes, visited_nodes, len(current_node.moves) - 1
        else:
            if current_node.parent:
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


def get_index_of_value(board, value):
    for index_row, row in enumerate(board):
        for index_col, elem in enumerate(row):
            if elem == value:
                return index_row, index_col


def manh(current_board):
    distance = 0
    for y1 in range(len(current_board)):
        for x1 in range(len(current_board)):
            y2, x2 = find_indexes(SOLVED_BOARD, current_board[y1][x1])
            distance += abs(y1 - y2) + abs(x1 - x2)
    return distance


def hamm(current_board):
    distance = 0
    for y in range(len(current_board)):
        for x in range(len(current_board)):
            if current_board[y][x] != SOLVED_BOARD[y][x]:
                distance += 1
    return distance

def astr(heuristic, start_time):
    visited_nodes, processed_nodes = 0, 1
    nodes_pool = []
    nodes_history = []
    start_node = Node(START_BOARD, None, None, [])
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


if __name__ == '__main__':

    # Parsing
    parser = argparse.ArgumentParser(description="Algorithm, order, source file, solution file, statistics file.")
    parser.add_argument('algorithm')
    parser.add_argument('order')
    parser.add_argument('source_file')
    parser.add_argument('solution_file')
    parser.add_argument('statistic_file')
    args = parser.parse_args()

    # Loading start board from file
    with open(args.source_file) as board:
        is_first_line_flag = True
        for line in board:
            if is_first_line_flag:
                # Setting up solved board
                solved_1d = list(range(1, int(line.split()[0]) ** 2))
                solved_1d.append(0)
                for i in range(int(line.split()[0])):
                    SOLVED_BOARD.append(solved_1d[i*4:(i+1)*4])
                is_first_line_flag = False
            else:
                START_BOARD.append([int(number) for number in line.split()])

    algorithm = globals()[args.algorithm]
    if args.algorithm == 'dfs' or args.algorithm == 'bfs':
        ORDER = [order for order in args.order]
        start_time = time.time()
        prepare_solution(algorithm(start_time), args.solution_file, args.statistic_file, start_time)
    else:
        heuristic = globals()[args.order]
        start_time = time.time()
        prepare_solution(algorithm(heuristic, start_time), args.solution_file, args.statistic_file, start_time)


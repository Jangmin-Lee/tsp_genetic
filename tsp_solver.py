import argparse
import csv
import multiprocessing
import pprint
import random
import re
import sys

from error import TSPError
from ds import Node, Result

pp = pprint.PrettyPrinter(indent=4)


def create_argv_parser():
    parser = argparse.ArgumentParser(description='TSP solver using GA')
    parser.add_argument('filename', metavar='fileName', type=str, help='*.tsp file')
    parser.add_argument('-f', action='append',
                        help='limit the total number of fitness eval')
    parser.add_argument('-p', action='append',
                        help='set population size')
    parser.add_argument('-m', action='append',
                        help='set mutation rate')
    parser.add_argument('-e', action='append',
                        help='set elitism size')
    return parser


# 몇백개는 iterator 안써도될듯
# def create_info_iter(f):
#     for info_line in f:
#         if info_line.strip() == "EOF":
#             break
#         m = re.search(r"(?P<index>\d+)\s(?P<x>\d+\.\d+)\s(?P<y>\d+\.\d+)", info_line)
#         assert m
#         node = Node(m.group('index'), m.group('x'), m.group('y'))
#         yield node

def create_tsp_list(f):
    tsp_list = []
    for info_line in f:
        if info_line.strip() == "EOF":
            break
        m = re.search(r"(?P<index>\d+)\s(?P<x>\d+\..*)\s(?P<y>\d+\..*)", info_line)
        assert m
        node = Node(m.group('index'), float(m.group('x')), float(m.group('y')))
        tsp_list.append(node)
    return tsp_list


def create_tsp_parser(f, option=None):
    cond = {}
    for condition_line in f:
        if condition_line.strip() == "NODE_COORD_SECTION":
            break
        key, value = condition_line.split(":")
        cond[key.strip()] = value.strip()
    tsp_list = create_tsp_list(f)
    return cond, tsp_list


def create_route_list(tsp_list):
    route_list = tsp_list[:]
    random.SystemRandom().shuffle(route_list)
    return route_list


def sort_results(random_res):
    return sorted(random_res, key=lambda result: result.fitness, reverse=True)


def selection_roulette(sorted_res):
    selected_results = []

    total_fitness = 0
    fitness_cum = []
    for res in sorted_res:
        total_fitness += int(res.fitness)
        fitness_cum.append(total_fitness)

    while len(selected_results) < 2:
        roulette_num = random.SystemRandom().randint(0, total_fitness)
        for cum in fitness_cum:
            if cum < roulette_num:
                continue
            selected_res = sorted_res[fitness_cum.index(cum)]
            if not selected_res in selected_results:
                selected_results.append(selected_res)
                break

    return selected_results

################################################ dict를 사용한 crossover
# def _make_edge_dict(parent):
#     ero_dict = dict()
#     for i in range(len(parent.tsp_list)):
#         if i == 0:
#             ero_dict[parent.tsp_list[i]] = [parent.tsp_list[-1], parent.tsp_list[i + 1]]
#         elif i == len(parent.tsp_list) - 1:
#             ero_dict[parent.tsp_list[i]] = [parent.tsp_list[i - 1], parent.tsp_list[0]]
#         else:
#             ero_dict[parent.tsp_list[i]] = [parent.tsp_list[i - 1], parent.tsp_list[i + 1]]
#     return ero_dict
#
#
# # 그러니까 elitism 은 그냥 fitness 순으로 n개를 앞에서 미리 할당해놓고
# # 그 다음부터 selection_roulette로 2개씩 뽑아서 breeding_crossover, mutation_exchange 과정을 거치고
# # selection_roulette를 population_num - elitism_size 만큼 실행해서 population을 만들고, 다음 세대해서 반복한다.
# # 그러니까 selection_roulette를 수정해야함.
# def _sort_dict_by_len(dictionary):
#     temp_list = []
#     for k in dictionary.keys():
#         temp_list.append((len(dictionary[k]), k, dictionary[k]))
#     sorted_dict = {}
#     for t in sorted(temp_list, key=lambda x: x[0]):
#         sorted_dict[t[1]] = t[2]
#     return sorted_dict
#
#
# def _get_target_nodes(dictionary):
#     min_val = 0
#     dict_keys = dictionary.keys()
#
#     for k in dict_keys:
#         if len(dictionary[k]) > min_val and min_val is 0:
#             min_val = len(dictionary[k])
#         elif len(dictionary[k]) > min_val and min_val is not 0:
#             break
#     target_index = list(dict_keys).index(k)
#     if target_index == 0:
#         return list(dict_keys)
#     return [key for key in list(dict_keys)[:target_index]]
#
#
# # Edge Recombination Crossover for TSP
# def breeding_crossover(parent_1, parent_2):
#     comb_dict = dict()
#     sys_random = random.SystemRandom()
#
#     breeding_res = []
#     par1_dict, par2_dict = _make_edge_dict(parent_1), _make_edge_dict(parent_2)
#
#     for key in par1_dict.keys():
#         comb_dict[key] = list(set(par1_dict[key] + par2_dict[key]))
#
#     while len(breeding_res) < len(parent_1.tsp_list):
#         comb_dict = _sort_dict_by_len(comb_dict)
#         target_nodes = _get_target_nodes(comb_dict)
#         target_node = sys_random.sample(target_nodes, 1)[0]
#         breeding_res.append(target_node)
#
#         for k in comb_dict.keys():
#             if target_node in comb_dict[k]:
#                 comb_dict[k].remove(target_node)
#         try:
#             del comb_dict[target_node]
#         except KeyError:
#             pass
#     return Result(breeding_res)


def _make_edge_list(parent):
    ero_list = []
    for i in range(len(parent.tsp_list)):
        if i == 0:
            ero_list.append((parent.tsp_list[i], [parent.tsp_list[-1], parent.tsp_list[i + 1]]))
        elif i == len(parent.tsp_list) - 1:
            ero_list.append((parent.tsp_list[i], [parent.tsp_list[i-1], parent.tsp_list[0]]))
        else:
            ero_list.append((parent.tsp_list[i], [parent.tsp_list[i-1], parent.tsp_list[i + 1]]))
    return ero_list


def _get_target_nodes(comb_list):
    target_index = 0
    min_val = 0
    for c in comb_list:
        if len(c[1]) > min_val and min_val is 0:
            min_val = len(c[1])
        elif len(c[1]) > min_val and min_val is not 0:
            target_index = comb_list.index(c)
            break
    if target_index == 0:
        return comb_list
    return comb_list[:target_index]


# Edge Recombination Crossover for TSP
def breeding_crossover(parent_1, parent_2):
    comb_list = []
    sys_random = random.SystemRandom()

    breeding_res = []
    par1_list, par2_list = _make_edge_list(parent_1), _make_edge_list(parent_2)
    par1s_list = sorted(par1_list, key=lambda x: int(str(x[0])))
    par2s_list = sorted(par2_list, key=lambda x: int(str(x[0])))
    for z in zip(par1s_list, par2s_list):
        comb_list.append((z[0][0], list(set(z[0][1] + z[1][1]))))

    while len(breeding_res) < len(parent_1.tsp_list):
        comb_list = sorted(comb_list, key=lambda x: len(x[1]))
        target_nodes = _get_target_nodes(comb_list)
        target_node = sys_random.sample(target_nodes, 1)[0]
        breeding_res.append(target_node[0])

        new_comb_list = []
        for c in comb_list:
            if target_node[0] in c[1]:
                c[1].remove(target_node[0])
                new_comb_list.append(c)
            elif target_node[0] == c[0]:
                continue
            else:
                new_comb_list.append(c)
        comb_list = new_comb_list
    return Result(breeding_res)


def mutation_exchange(selected_res, mutation_rate):
    _tsp_list = selected_res.tsp_list
    for i in range(len(_tsp_list)):
        if mutation_rate < random.uniform(0, 1):
            continue
        j = random.SystemRandom().randint(0, len(_tsp_list) - 1)
        _tsp_list[i], _tsp_list[j] = _tsp_list[j], _tsp_list[i]
    selected_res.tsp_list = _tsp_list
    return selected_res


# def _for_multiprocess_ga(sorted_res, mutation_rate, i):
#     parent_1, parent_2 = selection_roulette(sorted_res)
#     offspring = breeding_crossover(parent_1, parent_2)
#     mutated_offspring = mutation_exchange(offspring, mutation_rate)
#     return mutated_offspring
#
#
# def do_ga(sorted_res, population_size, mutation_rate, elitism):
#     elite_size = len(elitism)
#     # elitism
#     prod = [(sorted_res, mutation_rate, i) for i in range(population_size - elite_size)]
#     with multiprocessing.Pool(4) as pool:
#         elitism.extend(pool.starmap(_for_multiprocess_ga, prod))
#     print(elitism)
#     return elitism


def do_ga(sorted_res, population_size, mutation_rate, elitism, remainder=None):
    # elitism
    ga_eval_num = population_size - len(elitism)
    if remainder:
        ga_eval_num = remainder
    for i in range(ga_eval_num):
        parent_1, parent_2 = selection_roulette(sorted_res)
        # print("A")
        offspring = breeding_crossover(parent_1, parent_2)
        # print("B")
        mutated_offspring = mutation_exchange(offspring, mutation_rate)
        # print("C")
        elitism.append(mutated_offspring)
    return elitism


def main(population_size, mutation_rate, elite_size, fitness_eval_num):
    parser = create_argv_parser()
    parse_res = parser.parse_args(sys.argv[1:])
    if parse_res.p:
        population_size = int(parse_res.p[0])
    if parse_res.f:
        fitness_eval_num = int(parse_res.f[0])
    if parse_res.m:
        mutation_rate = int(parse_res.m[0])
    if parse_res.e:
        elite_size = int(parse_res.e[0])

    if not '.tsp' in parse_res.filename:
        raise TSPError("file extension is not supported.")

    try:
        f = open(parse_res.filename, 'r')
    except FileNotFoundError:
        raise TSPError("File Not Found, Please check file name again.")
    except IOError:
        raise TSPError("IOError, Can not read file")

    try:
        condition, tsp_list = create_tsp_parser(f)
    except Exception as e:
        print(e)
        raise TSPError("Parsing Error Occurred, Please check Tspfile format")

    random_res = [Result(create_route_list(tsp_list)) for i in range(population_size)]

    generation_num = fitness_eval_num//population_size
    generation_remainder = fitness_eval_num%population_size

    count = 0
    while count < generation_num:
        sorted_res = sort_results(random_res)
        # print(">>>>>", sorted_res[:2])
        random_res = do_ga(sorted_res, population_size, mutation_rate, sorted_res[:elite_size])
        count = count + 1

    sorted_res = sort_results(random_res)
    if generation_remainder:
        random_res = do_ga(sorted_res, population_size, mutation_rate, sorted_res[:elite_size], generation_remainder)

    sorted_res = sort_results(random_res)
    with open('solution.csv', 'w', encoding='utf-8', newline='') as sol:
        sol_writer = csv.writer(sol)
        for node in sorted_res[0].tsp_list:
            sol_writer.writerow([str(node)])
    print(sorted_res[0].distance)


if __name__ == "__main__":
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1
    ELITE_SIZE = 2
    FITNESS_EVAL_NUM = 10000

    main(POPULATION_SIZE, MUTATION_RATE, ELITE_SIZE, FITNESS_EVAL_NUM)

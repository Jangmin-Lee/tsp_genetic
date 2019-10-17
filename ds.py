import math

from error import TSPError


class Node(object):
    def __init__(self, order, x, y):
        self.x = x
        self.y = y
        self.order = order

    def __repr__(self):
        return "Node{order}({x}, {y})".format(order=self.order, x=self.x, y=self.y)

    def __str__(self):
        return self.order

    def distance_btw_node(self, node):
        x_diff = self.x - node.x
        y_diff = self.y - node.y
        # print("Node{} to Node{}".format(self.order, node.order))
        return math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))


class Result(object):
    def __init__(self, tsp_list):
        self.tsp_list = tsp_list
        self.distance = self._calculate_distance()
        self.fitness = self._calculate_fitness()

    def __repr__(self):
        return "Result fitness : {}".format(self.fitness)

    def _calculate_distance(self):
        self.distance = 0
        from_node = self.tsp_list[0]
        for to_node in self.tsp_list[1:]:
            self.distance += from_node.distance_btw_node(to_node)
            from_node = to_node
        return self.distance

    def _calculate_fitness(self):
        distance = self._calculate_distance()
        try:
            self.fitness = 1000000000 / distance
        except ZeroDivisionError:
            # print(self.tsp_list)
            raise TSPError("Divide by Zero at calculating fitness")
        return self.fitness



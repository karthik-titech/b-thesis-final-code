"""
SAT solver using CDCL, tensorflow computation graph
"""
import os
import time
from collections import deque
import tensorflow as tf
import numpy as np

from pysat.formula import CNF as CNF_extract_file
from cnf import CNF

TRUE = 1
FALSE = 0
UNASSIGN = -1

# @tf.function()
def out_unit(table, query):

    # Flip -1 -> 0 0 -> -1 1 -> 1
    tmp = tf.abs(table) - 1
    new_table = tmp + tf.cast(table == 1, tf.int32)

    # Absolute query
    abs_query = tf.abs(query)
    sum_each_query = tf.reduce_sum(abs_query, axis=1)

    # -1 in table
    unassign_table = tf.cast(table == -1, tf.int32)

    check_all_false = (tf.squeeze(tf.matmul(query, new_table)) + sum_each_query - 1) == 0 # Just a dot product, to check the number variables assigned to False [No. of false assignments + no. of vars -1]

    check_unassign = (tf.squeeze(tf.matmul(abs_query, unassign_table)) - 1) == 0 # a dot product to check if exactly one variable is unassigned

    ok_idx = tf.where(tf.math.logical_and(check_all_false, check_unassign)) # mark the indices of the clauses with only 1 assigned variable
    que_ok = tf.gather_nd(query, ok_idx) # gather the clauses with only one assigned variable
    test = tf.where(tf.transpose(tf.transpose(tf.abs(que_ok)) * table == -1)) # identify the unit clause variable

    return tf.where(tf.math.logical_and(check_all_false, check_unassign))[:,0] + 1, tf.cast(test[:,1]+1, tf.int32) * (tf.gather_nd(que_ok, test))

# out_unit(table, query)

def dict2tensor(clauses, num_var):
    idx, fill = [],[]
    for i, x in enumerate(clauses):
        for el in x:
            idx.append([i, abs(el) - 1])
            fill.append(int(el > 0)* 2 - 1)

    return tf.scatter_nd(idx, fill, [len(clauses),num_var])




    # mask = query != 0
    # masked = tf.transpose(tf.cast(mask, tf.int32)) * table
    # count_unass = tf.reduce_sum(tf.cast(masked == -1, tf.int32))
    # if count_unass != 1:
    #     return False

    # assign_mask = tf.math.logical_and(table != -1, mask)

    # assign_arr = tf.boolean_mask(query, assign_mask) * tf.boolean_mask(table * 2 - 1, assign_mask)
    # if tf.reduce_sum(assign_arr) == -tf.shape(assign_arr)[0]:
    #     return True
    # else:
    #     return False


class CDCL():
    def run(self, cnf):
        raw_solver = CDCLRawSolver(cnf)
        raw_solver.suggest = self.suggest
        solution = raw_solver.run()
        self.number_of_runs = raw_solver.number_of_runs
        return solution

    def suggest(self, cnf):
        return next(iter(cnf.vars))


class CDCLRawSolver:
    def __init__(self, cnf):
        self.clauses, self.vars = {frozenset(cl) for cl in cnf.clauses}, cnf.vars
        self.num_var = max(self.vars)
        self.learnts = set()
        self.assigns = dict.fromkeys(list(self.vars), UNASSIGN)
        self.assigns_np = tf.Variable(-tf.ones((self.num_var, 1), tf.int32))
        self.level = 0
        self.nodes = dict((k, ImplicationNode(k, UNASSIGN)) for k in list(self.vars))
        self.branching_vars = set()
        self.branching_history = {}  # level -> branched variable
        self.propagate_history = {}  # level -> propagate variables list
        self.branching_count = 0
        # print(self.assigns)
        # print(self.clauses)

    def run(self):
        sat = self.solve()
        if sat is False:
            return None
        else:
            return [(k if v == 1 else (-k)) for k, v in self.assigns.items()]
            # return [(k if v == 1 else (-k)) for k, v in enumerate(self.assigns)]

    def solve(self):
        """
        Returns TRUE if SAT, False if UNSAT
        :return: whether there is a solution
        """
        self.number_of_runs = 0
        while not self.are_all_variables_assigned():
            conf_cls = self.unit_propagate()
            if conf_cls is not None:
                # there is conflict in unit propagation
                lvl, learnt = self.conflict_analyze(conf_cls)
                if lvl < 0:
                    return False
                self.learnts.add(learnt)
                self.backtrack(lvl)
                self.level = lvl
            elif self.are_all_variables_assigned():
                break
            else:
                # branching
                self.level += 1
                self.branching_count += 1
                bt_var, bt_val = self.pick_branching_variable()
                self.assigns[bt_var] = bt_val
                self.assigns_np[bt_var-1,0].assign( bt_val)
                self.branching_vars.add(bt_var)
                self.branching_history[self.level] = bt_var
                self.propagate_history[self.level] = deque()
                self.update_graph(bt_var)

        return True

    def compute_value(self, literal):
        """
        Compute the value of the literal (could be -/ve or +/ve) from
        `assignment`. Returns -1 if unassigned
            :param literal: an int, can't be 0
            :returns: value of the literal
        """
        value = self.assigns[abs(literal)]
        value = value if value == UNASSIGN else value ^ (literal < 0)
        return value

    def compute_clause(self, clause):
        values = list(map(self.compute_value, clause))
        value = UNASSIGN if UNASSIGN in values else max(values)
        return value

    def compute_cnf(self):
        return min(map(self.compute_clause, self.clauses))

    def is_unit_clause(self, clause):
        """
        Checks if clause is a unit clause. If and only if there is
        exactly 1 literal unassigned, and all the other literals having
        value of 0.
            :param clause: set of ints
            :returns: (is_clause_a_unit, the_literal_to_assign, the clause)
        """
        values = []
        unassigned = None

        for literal in clause:
            value = self.compute_value(literal)
            values.append(value)
            unassigned = literal if value == UNASSIGN else unassigned

        check = ((values.count(FALSE) == len(clause) - 1 and
                  values.count(UNASSIGN) == 1) or
                 (len(clause) == 1
                  and values.count(UNASSIGN) == 1))
        return check, unassigned

    def assign(self, literal):
        """ Assign the variable so that literal is TRUE """

    def update_graph(self, var, clause=None):
        node = self.nodes[var]
        node.value = self.assigns[var]
        node.level = self.level

        # update parents
        if clause:  # clause is None, meaning this is branching, no parents to update
            for v in [abs(lit) for lit in clause if abs(lit) != var]:
                node.parents.append(self.nodes[v])
                self.nodes[v].children.append(node)
            node.clause = clause

    def unit_propagate(self):
        """
        A unit clause has all of its literals but 1 assigned to 0. Then, the sole
        unassigned literal must be assigned to value 1. Unit propagation is the
        process of iteratively applying the unit clause rule.
        :return: None if no conflict is detected, else return the literal
        """
        while True:


            # propagate_queue = deque()
            # union_clauses = self.clauses.union(self.learnts)
            list_union_clauses = list(self.clauses.union(self.learnts))

            undetermined_clauses = []
            for clause in list_union_clauses:
                c_val = self.compute_clause(clause)
                if c_val == TRUE:
                    continue
                if c_val == FALSE:
                    return clause
                else:
                    undetermined_clauses.append(clause)


            query_tensor = dict2tensor(undetermined_clauses, self.num_var)
            is_unit, lit_unit = out_unit(self.assigns_np, query_tensor)
            is_unit, lit_unit = is_unit.numpy(), lit_unit.numpy()

            if len(is_unit) == 0:
                return None

            for idx, prop_lit in zip(is_unit, lit_unit):
            # for prop_lit, clause in propagate_queue:
                clause = undetermined_clauses[idx-1]
                prop_var = abs(prop_lit)
                self.assigns[prop_var] = TRUE if prop_lit > 0 else FALSE
                self.assigns_np[prop_var-1,0].assign(TRUE if prop_lit > 0 else FALSE)
                self.update_graph(prop_var, clause=clause)
                try:
                    self.propagate_history[self.level].append(prop_lit)
                except KeyError:
                    pass  # propagated at level 0

    def get_unit_clauses(self):
        return list(filter(lambda x: x[0], map(self.is_unit_clause, self.clauses)))

    def are_all_variables_assigned(self):
        all_assigned = all(var in self.assigns for var in self.vars)
        none_unassigned = not any(var for var in self.vars if self.assigns[var] == UNASSIGN)
        return all_assigned and none_unassigned

    def all_unassigned_vars(self):
        return filter(
            lambda v: v in self.assigns and self.assigns[v] == UNASSIGN,
            self.vars)

    def suggest(self, cnf):
        
        return next(iter(cnf.vars))
        # raise NotImplementedError()

    # def pick_branching_variable(self, bt_var=None, bt_val=None):
    def pick_branching_variable(self):
        """
        Pick a variable to assign a value.
        :return: variable, value assigned
        """
        new_clauses = []
        for clause in self.clauses.union(self.learnts):
            lit_vals = [self.compute_value(lit) for lit in clause]
            if TRUE in lit_vals:
                continue
            new_clause = [lit for lit in clause if self.compute_value(lit) == UNASSIGN]
            new_clauses.append(new_clause)
        if not new_clauses:
            # pick whatever, clause will be solved anyway
            var = next(self.all_unassigned_vars())
            return var, TRUE
        self.number_of_runs += 1
        lit = self.suggest(CNF(new_clauses))
        return abs(lit), (TRUE if lit > 0 else FALSE)

    def conflict_analyze(self, conf_cls):
        """
        Analyze the most recent conflict and learn a new clause from the conflict.
        - Find the cut in the implication graph that led to the conflict
        - Derive a new clause which is the negation of the assignments that led to the conflict

        Returns a decision level to be backtracked to.
        :param conf_cls: (set of int) the clause that introduces the conflict
        :return: ({int} level to backtrack to, {set(int)} clause learnt)
        """
        def next_recent_assigned(clause):
            """
            According to the assign history, separate the latest assigned variable
            with the rest in `clause`
            :param clause: {set of int} the clause to separate
            :return: ({int} variable, [int] other variables in clause)
            """
            for v in reversed(assign_history):
                if v in clause or -v in clause:
                    return v, [x for x in clause if abs(x) != abs(v)]

        if self.level == 0:
            return -1, None


        assign_history = [self.branching_history[self.level]] + list(self.propagate_history[self.level])

        pool_lits = conf_cls
        done_lits = set()
        curr_level_lits = set()
        prev_level_lits = set()

        while True:
            for lit in pool_lits:
                if self.nodes[abs(lit)].level == self.level:
                    curr_level_lits.add(lit)
                else:
                    prev_level_lits.add(lit)

            if len(curr_level_lits) == 1:
                break

            last_assigned, others = next_recent_assigned(curr_level_lits)

            done_lits.add(abs(last_assigned))
            curr_level_lits = set(others)

            pool_clause = self.nodes[abs(last_assigned)].clause
            pool_lits = [
                l for l in pool_clause if abs(l) not in done_lits
            ] if pool_clause is not None else []

        learnt = frozenset([l for l in curr_level_lits.union(prev_level_lits)])
        if prev_level_lits:
            level = max([self.nodes[abs(x)].level for x in prev_level_lits])
        else:
            level = self.level - 1

        return level, learnt

    def backtrack(self, level):
        """
        Non-chronologically backtrack ("back jump") to the appropriate decision level,
        where the first-assigned variable involved in the conflict was assigned
        """
        for var, node in self.nodes.items():
            if node.level <= level:
                node.children[:] = [child for child in node.children if child.level <= level]
            else:
                node.value = UNASSIGN
                node.level = -1
                node.parents = []
                node.children = []
                node.clause = None
                self.assigns[node.variable] = UNASSIGN
                self.assigns_np[node.variable-1, 0].assign(UNASSIGN)

        self.branching_vars = set([
            var for var in self.vars
            if (self.assigns[var] != UNASSIGN
                and len(self.nodes[var].parents) == 0)
        ])

        levels = list(self.propagate_history.keys())
        for k in levels:
            if k <= level:
                continue
            del self.branching_history[k]
            del self.propagate_history[k]


class ImplicationNode:
    """
    Represents a node in an implication graph. Each node contains
    - its value
    - its implication children (list)
    - parent nodes (list)
    """

    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        self.level = -1
        self.parents = []
        self.children = []
        self.clause = None

    def all_parents(self):
        parents = set(self.parents)
        for parent in self.parents:
            for p in parent.all_parents():
                parents.add(p)
        return list(parents)

    def __str__(self):
        sign = '+' if self.value == TRUE else '-' if self.value == FALSE else '?'
        return "[{}{}:L{}, {}p, {}c, {}]".format(
            sign, self.variable, self.level, len(self.parents), len(self.children), self.clause)

    def __repr__(self):
        return str(self)

def timer(f):
    import time
    st = time.time()
    f()
    ed = time.time()

    print("it took", ed - st)

def main():
    s = CNF_extract_file(from_file='uf125-01.cnf')
    #s = CNF(s)
    #print(s.clauses)
    #print(CDCLRawSolver(CNF(s.clauses)).run())


    query = np.array([[ 0, -1,  1,  0,  0,  0, -1],
       [-1, -1,  0,  0,  1,  1,  1],
       [ 1,  0,  1,  0,  0,  0, -1],
       [ 1,  0,  1,  0,  0,  0,  1]], dtype=np.int32)
    table = np.array([[-1],
       [-1],
       [ 0],
       [ 1],
       [ 1],
       [ 0],
       [ 1]], dtype=np.int32)

    # Warm up
    #out_unit(table, query)

    timer(lambda: print(CDCLRawSolver(CNF(s.clauses)).run()))
    # timer(CDCLRawSolver(s).run)

"""
    rcnf = get_random_kcnf(3, 4, 20)
    print(rcnf)
    print(CDCLRawSolver(rcnf).run())
    #print(rcnf.simplified())
"""

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import numpy as np
import math
import utils

# gene（f，c1，c2） f:function type, c:connection (nodeID)
class Individual(object):

    def __init__(self, net_info, init):
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        if init:
            print('init with specific architectures')
            self.init_gene_with_conv() # In the case of starting only convolution
        else:
            self.init_gene()           # generate initial individual randomly

    def init_gene_with_conv(self):
        """"""

    def init_gene(self):
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            #self.node_num = rows * cols
            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)

            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0

            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            if n >= self.net_info.node_num:    # output node
                in_num = self.net_info.out_in_num[t]
            else:    # intermediate node
                in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i+1] >= self.net_info.input_num:
                    self.__check_course_to_out(self.gene[n][i+1] - self.net_info.input_num)

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        while current == mutated_gene:

            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate=0.2):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            # self.node_num = rows * cols
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)
                    if self.is_active[n] and i < in_num:
                        active_check = True


        self.check_active()
        return active_check


    def neutral_mutation(self, mutation_rate=0.2):
        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if not self.is_active[n] and np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= in_num) and np.random.rand() < mutation_rate \
                        and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)

        self.check_active()
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):

        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval
    def _crossover(self, r, source1, source2):
        self.gene[:r] = source1.gene[:r] #parent1 inherits more genes
        self.gene[r:] = source2.gene[r:]
        num1 = r - 1
        num2 = r
        col1 = np.min((int(num1 / self.net_info.rows), self.net_info.cols))
        col2 = np.min((int(num2 / self.net_info.rows), self.net_info.cols))
        while col1 == col2 and num1 > 0:
            num1 = num1 - 1
            col1 = np.min((int(num1 / self.net_info.rows), self.net_info.cols))
        while not source1.is_active[num1] and num1 > 0:
            num1 = num1 - 1
        while not source2.is_active[num2] and num2 < self.net_info.node_num:
            num2 = num2 + 1
        self.gene[num2][1] = num1 + 1
        self.gene[num2][2] = num1 + 1

        num2 = num2 + 1
        col_max = np.min((int(num2 / self.net_info.rows), self.net_info.cols)) + self.net_info.level_back
        col = np.min((int(num2 / self.net_info.rows), self.net_info.cols))

        while num2 < self.net_info.node_num and col <= col_max:
            col = np.min((int(num2 / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            if source2.is_active[num2]:

                if self.gene[num2][0] == 12 or self.gene[num2][0] == 13 or self.gene[num2][0] == 14:
                    input_number = 2
                else:
                    input_number = 1
                for i in range(input_number):
                    input = self.gene[num2][i]
                    if input < r and not source1.is_active[input]:
                        print("source1.is_active[input1]", input)
                    elif input >= r and not source2.is_active[input]:
                        print("source2.is_active[input1]", input)
                    else:  #
                        print("inactive")
                        count_num = 0
                        while True:
                            input = self.__mutate(input, min_connect_id, max_connect_id)
                            if input < r and source1.is_active[input]:
                                self.gene[num2][i] = input
                                break
                            elif input >= r and source2.is_active[input]:
                                self.gene[num2][i] = input
                                break
                            elif count_num == 20:
                                break
                            else:
                                count_num = count_num + 1
            num2 = num2 + 1
    def crossover(self, source1, source2):

        self.net_info = source1.net_info
        r = np.random.randint(self.net_info.node_num)
        if r == 0:
            self.gene = source1.gene
        elif r > self.net_info.node_num/2:
            self._crossover(r, source1, source2)

        else:
            self._crossover(r, source2, source1)
        self.check_active()

    def active_net_list(self):
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:    # intermediate node
                    type_str = self.net_info.func_type[t]
                else:    # output node
                    type_str = self.net_info.out_type[t]
                connections = [active_cnt[self.gene[n][i+1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list


class ECGP(object):
    def __init__(self, net_info, eval_func, lam=4, rowSize=20,colSize=32, init=False):
        self.lam = lam
        self.pop = [Individual(net_info, init) for _ in range(2 + self.lam)]
        self.swap_ind = Individual(net_info, init)
        self.eval_func = eval_func
        self.num_gen = 0
        self.num_eval = 0
        self.init = init
    def evaluation(self, pop):
        # create network list
        net_lists = []

        net_lists.append(pop.active_net_list())

        # evaluation
        fp = self.eval_func(net_lists)

        pop.eval = fp[0]
        evaluations = pop.eval
        self.num_eval += len(net_lists)
        return evaluations
    def _evaluation(self, pop, eval_flag):
        # create network list
        net_lists = []

        active_index = np.where(eval_flag)[0]
        for i in active_index:
            net_lists.append(pop[i].active_net_list())

        # evaluation
        fp = self.eval_func(net_lists)
        for i, j in enumerate(active_index):
            pop[j].eval = fp[i]
        evaluations = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations[i] = pop[i].eval
        self.num_eval += len(net_lists)
        return evaluations


    def _log_data(self, net_info_type='active_only', start_time=0):
        log_list = [self.num_gen, self.num_eval, time.time()-start_time, self.pop[0].eval, self.pop[0].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[0].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[0].gene.flatten().tolist()
        else:
            pass
        return log_list

    def _log_data_children(self, net_info_type='active_only', start_time=0, pop=None):
        log_list = [self.num_gen, self.num_eval, time.time()-start_time, pop.eval, pop.count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        else:
            pass
        return log_list

    def load_log(self, log_data):
        self.num_gen = log_data[0]
        self.num_eval = log_data[1]
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].gene = np.array(log_data[5:]).reshape((net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()

    def modified_evolution(self, max_eval=100, mutation_rate=0.2, log_file='./log.txt', arch_file='./arch.txt'):
        with open('child.txt', 'w') as fw_c :
            writer_c = csv.writer(fw_c, lineterminator='\n')
            start_time = time.time()
            eval_flag = np.empty(self.lam)
            active_num = self.pop[0].count_active_node()
            active_num1 = self.pop[1].count_active_node()
            if self.init:
                pass
            else: # in the case of not using an init indiviudal
                while active_num < self.pop[0].net_info.min_active_num:
                    self.pop[0].mutation(1.0)
                    active_num = self.pop[0].count_active_node()
                while active_num1 < self.pop[1].net_info.min_active_num:
                    self.pop[1].mutation(1.0)
                    active_num1 = self.pop[1].count_active_node()
            self._evaluation(self.pop[:2], np.array([True, True]))

            if self.pop[0].eval < self.pop[1].eval:
                self.swap_ind.copy(self.pop[0])
                self.pop[0].copy(self.pop[1])
                self.pop[1].copy(self.swap_ind)
            update_flag = 0
            live_flag = 0
            ite = np.linspace(1, max_eval, max_eval)
            mutation_rate2 = 1 + 1 * ite / max_eval
            mutation_rate = 1 - 0.5 * ite / max_eval
            while self.num_gen < max_eval:


                if update_flag ==5:
                    live_flag = 3
                    while active_num1 < self.pop[1].net_info.min_active_num:
                        self.pop[1].mutation(1.0)
                        active_num1 = self.pop[1].count_active_node()

                    self.evaluation(self.pop[1])


                    if self.pop[0].eval < self.pop[1].eval:
                        self.swap_ind.copy(self.pop[0])
                        self.pop[0].copy(self.pop[1])
                        self.pop[1].copy(self.swap_ind)


                self.swap_ind.copy(self.pop[0])
                self.pop[2].crossover(self.pop[0],self.pop[1])
                self.pop[0].copy(self.swap_ind)
                eval_flag[0] = True
                # reproduction
                for i in range(2):
                    eval_flag[i+1] = False
                    self.pop[i + 3].copy(self.pop[0])  # copy a parent
                    active_num = self.pop[i + 3].count_active_node()

                    while not eval_flag[i+1] or active_num < self.pop[i + 3].net_info.min_active_num:
                        self.pop[i + 3].copy(self.pop[0])                       # copy a parent
                        eval_flag[i+1] = self.pop[i + 3].mutation(mutation_rate[self.num_gen])  # mutation
                        active_num = self.pop[i + 3].count_active_node()
                self.pop[5].copy(self.pop[1])
                active_num = self.pop[5].count_active_node()
                while not eval_flag[3] or active_num < self.pop[5].net_info.min_active_num:
                    self.pop[5].copy(self.pop[1])  # copy a parent
                    eval_flag[3] = self.pop[5].mutation(mutation_rate2[self.num_gen])  # mutation
                    active_num = self.pop[5].count_active_node()

                evaluations = self._evaluation(self.pop[2:], eval_flag=eval_flag)
                best_arg = evaluations.argmax()
                # save
                f = open('arch_child.txt', 'a')
                writer_f = csv.writer(f, lineterminator='\n')
                for c in range(2 + self.lam):
                    writer_c.writerow(self._log_data_children(net_info_type='full', start_time=start_time, pop=self.pop[c]))
                    writer_f.writerow(self._log_data_children(net_info_type='active_only', start_time=start_time, pop=self.pop[c]))
                f.close()
                # replace the parent by the best individual
                if evaluations[best_arg] > self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg + 2])
                    update_flag = 0
                elif evaluations[best_arg] > self.pop[1].eval :
                    if live_flag<0:
                        self.pop[1].copy(self.pop[best_arg + 2])
                        update_flag = 0
                else:
                    self.pop[0].neutral_mutation(1)  # modify the parent (neutral mutation)
                    self.pop[1].neutral_mutation(1)
                live_flag = live_flag-1
                update_flag = update_flag +1
                self.num_gen += 1
                fw = open(log_file, 'a')
                writer = csv.writer(fw, lineterminator='\n')
                writer.writerow(self._log_data(net_info_type='full', start_time=start_time))
                fa = open('arch.txt', 'a')
                writer_a = csv.writer(fa, lineterminator='\n')
                writer_a.writerow(self._log_data(net_info_type='active_only', start_time=start_time))
                fw.close()
                fa.close()

"""
This is the Hard-sphere model for infectious disease
"""

import os
import time
from datetime import datetime
from collections import OrderedDict
import numpy as np
import click
import math
from matplotlib import pyplot

import utils

NOW = datetime.today().strftime("%s")
INPUT_FILE = "./input.csv"
Z_COORDINATE = 1
SIGMA = 1.0


class _UserParams:
    """ class containing user parameters """

    def __init__(self):
        """ initialize the object """

        self._input_file = INPUT_FILE
        self._params_dict = self._read_input()

        self._cleaned_old_files = False
        self._prng_obj = utils.RandomNumberGenerator()

    def _read_input(self) -> OrderedDict:
        """ read input """
        raw_data = utils.read_csv(self._input_file)
        result = OrderedDict({parameter: utils.my_round(value) for parameter, value in raw_data})

        return result

    @property
    def density(self):
        return self._params_dict["density"]

    @property
    def members(self):
        return self._params_dict["members"]

    @property
    def eq_steps(self):
        return self._params_dict['eq_steps']

    @property
    def mc_steps(self):
        return self._params_dict["mc_steps"]

    def _output_file_dict(self):
        output_file_dict = {"xyz_file": "vmd.xyz", "time_series": "time_series.csv"}
        if not self._cleaned_old_files:
            for _, item in output_file_dict.items():
                if os.path.exists(item):
                    os.remove(item)
            self._cleaned_old_files = True
        else:
            pass

        return output_file_dict

    @property
    def xyz_file(self):
        return self._output_file_dict()['xyz_file']

    @property
    def time_series(self):
        return self._output_file_dict()['time_series']

    @property
    def sample_freq(self):
        return self._params_dict["sampling_freq"]

    @property
    def get_random_number(self):
        return self._prng_obj.get

    @property
    def infection_zone(self):
        return self._params_dict["infection_zone"]


class Person(_UserParams):
    """ class which defined a person """

    def __init__(self, age: str = None, position: tuple = (None, None, None)):
        """ initialize the class """
        super().__init__()  # initialize the child class
        self._age = age
        self._infected = False
        self._position = position
        self._size = utils.my_round(1.0)

    @property
    def immunity(self) -> float:
        """ define immunity """
        result = utils.my_round(0.5)
        return result

    @property
    def is_infected(self) -> bool:
        """ check if the person is infected """
        return self._infected

    def now_infected(self):
        """ function to infect a person """
        self._infected = True

    def now_cured(self):
        """ function to cure a person """
        self._infected = False

    @property
    def position(self) -> tuple:
        """ get the position of a person """
        if self._position != (None, None, None):
            utils.my_round(self._position)

        return self._position

    def set_position(self, position: tuple):
        """ set the position of a person """
        self._position = position

    @property
    def size(self):
        """ get the size of a person """
        return self._size

    @property
    def identity(self):
        """ get the molecular identity of a person """
        result = "C" if self.is_infected else "Si"
        return result


class System(_UserParams):
    """ class to initialize the system """

    def __init__(self):
        """ define the class """
        # self._parameters = parameter_dict
        super().__init__()
        self.members_obj_list = []

    @staticmethod
    def _max_density(lattice_type="sc"):
        """ get max density """
        if lattice_type == "sc":
            result = utils.my_round(0.523)
        elif lattice_type == "fcc":
            result = utils.my_round(0.74)
        elif lattice_type == "bcc":
            result = utils.my_round(0.68)
        else:
            raise NotImplementedError(
                "Lattice type not implemented: {}".format(lattice_type))

        return result

    @property
    def dimension(self):
        """ get the system dimension """
        volume = self.members / self.density
        result = volume ** (1 / 2)  # for 2D system
        return result

    def get_random_location(self) -> tuple:
        """ get a random location in system """
        rand_x = self._prng_obj.get
        rand_y = self._prng_obj.get
        position_x = utils.my_round(rand_x * self.dimension)
        position_y = utils.my_round(rand_y * self.dimension)

        return position_x, position_y, Z_COORDINATE

    def initialize(self):
        """ setup the system """

        print("*** Initializing System ***")

        total_members_added = 1
        # place the first member anywhere in the box
        a_random_position = self.get_random_location()
        member_obj = Person()
        member_obj.set_position(a_random_position)
        self.add_person(member_obj)

        while total_members_added < self.members:
            a_random_position = self.get_random_location()
            attempt = 1
            member_can_be_added = True

            for member_obj in self.members_obj_list:
                mem_dist = self.get_points_separation(a_random_position, member_obj.position)
                if mem_dist < SIGMA:
                    member_can_be_added = False
                    break

            if not member_can_be_added:
                if attempt > 1000:
                    raise ValueError("Total Attempts exceeded. System too dense")
                attempt += 1
                continue
            else:
                member_obj = Person()
                member_obj.set_position(a_random_position)
                self.add_person(member_obj)
                total_members_added += 1

        if total_members_added != self.members:
            raise ValueError(
                "Total placed {} not same as members in input {}".format(total_members_added, self.members))

        for imem_obj in self.members_obj_list:
            for jmem_obj in self.members_obj_list:
                if imem_obj == jmem_obj:
                    continue

                imem_x, imem_y, _ = imem_obj.position
                jmem_x, jmem_y, _ = jmem_obj.position
                mem_dist = math.sqrt(((imem_x - jmem_x) ** 2 + (imem_y - jmem_y) ** 2))
                if mem_dist < SIGMA:
                    raise ValueError("System not initialized properly")

    def get_points_separation(self, point_a: tuple, point_b: tuple) -> float:
        """ get the separation of points in the system consiering the minimum image convention """

        point_a_x, point_a_y, _ = point_a
        point_b_x, point_b_y, _ = point_b

        delta_x = point_a_x - point_b_x
        delta_x = delta_x - self.dimension * round(delta_x / self.dimension)  # min image X

        delta_y = point_a_y - point_b_y
        delta_y = delta_y - self.dimension * round(delta_y / self.dimension)  # min image Y

        # check moving condition
        ab_dist = utils.my_round(math.sqrt((delta_x ** 2 + delta_y ** 2)))

        return ab_dist

    def add_person(self, person: Person):
        """ add a person to the population """
        self.members_obj_list.append(person)

    def remove_person(self, person: Person):
        """ remove a person rom the system """
        self.members_obj_list.remove(person)


class MonteCarlo(_UserParams):
    """ class to run MCs """

    def __init__(self):
        """ initialize the class object """
        super().__init__()
        self._obj_system = System()
        self._system_dimension = self._obj_system.dimension

        # define radial distribution function
        self._gr = {}
        nbins = 20
        self._bin_dstep = utils.my_round(self._system_dimension / nbins)
        # initiate self._gr
        self._gr["bins"] = np.arange(0, self._system_dimension, self._bin_dstep)
        self._gr["values"] = np.zeros(len(self._gr["bins"]))

    def attempt_moving_person(self, obj_person: Person) -> bool:
        """ attempt moving a person """
        # new_position_x, new_position_y, _ = self._get_member_new_possible_position(obj_person)
        new_position_x, new_position_y, _ = self._obj_system.get_random_location()

        # attempt to move the person
        move_person = True
        neigh_list = []
        for neighbor_obj in self._obj_system.members_obj_list:
            if neighbor_obj == obj_person:
                continue

            neigh_x, neigh_y, _ = neighbor_obj.position
            neighbor_dist = self._obj_system.get_points_separation((new_position_x, new_position_y, _),
                                                                   (neigh_x, neigh_y, _))
            if neighbor_dist <= self.infection_zone and (neighbor_obj.is_infected or obj_person.is_infected):
                neigh_list.append(neighbor_obj)

            if neighbor_dist < SIGMA:
                move_person = False
                break

        # move the person
        if move_person:
            # update the coordinate
            obj_person.set_position((new_position_x, new_position_y, Z_COORDINATE))

            # update the infection of person and its neighbors
            if obj_person.is_infected:
                for neigh_obj in neigh_list:
                    if not neigh_obj.is_infected:
                        neigh_obj.now_infected()
            else:
                is_any_infected = False
                for neigh_obj in neigh_list:
                    if neigh_obj.is_infected:
                        is_any_infected = True
                        break
                if is_any_infected:
                    obj_person.now_infected()

        return move_person

    def write_xyz(self):
        """ save xyz coordinates to the file """
        with open(self.xyz_file, mode="a") as file_handle:
            file_handle.write("{}\n".format(int(self.members)))
            # file_handle.write("identity, x, y, z \n")
            file_handle.write("\n")

            for obj_person in self._obj_system.members_obj_list:
                person_identity = obj_person.identity
                x, y, z = obj_person.position
                # file_handle.write("{}\t {}\t {}\t {}\n".format(person_identity, x, y, z))
                file_handle.write("{:4} {:11.6f} {:11.6f} {:11.6f}\n".format(person_identity, x, y, z))

    def _equilibrate(self):
        """ execute the Monte Carlo simulations """
        for step in range(1, round(self.eq_steps) + 1):  # for each MC step
            print("*******\nEquilibration Step: {}\n*******".format(step))
            # write_xyz(self.__obj_system, self.__output_file_dict["vmd"])

            for obj_person in self._obj_system.members_obj_list:  # loop over all the people in the system
                self.attempt_moving_person(obj_person)

            self.write_xyz()

    def _sample(self):
        """ start the MC sampling """

        # impart the first infection
        random_member = self._obj_system.members_obj_list[int(self.get_random_number * self.members)]
        random_member.now_infected()

        for step in range(1, round(self.mc_steps) + 1):  # for each MC step

            # write_xyz(self.__obj_system, self.__output_file_dict["vmd"])
            for obj_person in self._obj_system.members_obj_list:  # loop over all the people in the system
                self.attempt_moving_person(obj_person)
                self.write_xyz()

            if step % self.sample_freq == 0:
                print("*******\nSampling Step: {}\n*******".format(step))
                # self._update_gr()  # update radial distribution function

        # for idx, elem in enumerate(self._gr["values"][1:]):
        #     normalization_factor = self.density * 2 * math.pi * self._gr["bins"][idx] * self._bin_dstep
        #     self._gr["values"][idx] = elem / normalization_factor
        # # self._gr["values"] = [elem / gr_mormalizing_factor for elem in self._gr["values"]]
        # pyplot.plot(self._gr["bins"], self._gr["values"])
        # pyplot.show()

    def _update_gr(self):
        """ claculate radial distribution function """

        for idx in range(len(self._obj_system.members_obj_list)):
            imem_obj = self._obj_system.members_obj_list[idx]
            # print("=== idx ===\t", idx)
            for jdx in range(idx + 1, len(self._obj_system.members_obj_list)):
                # print("jdx\t", jdx)
                jmem_obj = self._obj_system.members_obj_list[jdx]
                i_j_dist = self._obj_system.get_points_separation(imem_obj.position, jmem_obj.position)

                if i_j_dist < SIGMA:
                    raise ValueError(
                        "Members too close: {} -- {}\nSeparation: {}\tsystem size: {}".format(imem_obj.position,
                                                                                              jmem_obj.position,
                                                                                              i_j_dist,
                                                                                              self._system_dimension))

                for kdx, kbin in enumerate(self._gr["bins"]):
                    if kbin - self._bin_dstep / 2.0 < i_j_dist < kbin + self._bin_dstep / 2.0:
                        self._gr["values"][kdx] += 2

        # for key, val in self._gr.items():
        #     print(key, val)

    def execute(self):
        """ execute the Monte Carlo simulations """

        # initialize the system
        self._obj_system.initialize()
        self.write_xyz()

        # 0. equilibrate the system
        self._equilibrate()

        # 1. start sampling
        self._sample()


@click.command()
def main():
    """ the main executable function """

    start_time = time.time()

    # run the Monte Carlo Simulations
    obj_mc = MonteCarlo()
    obj_mc.execute()

    print("--- %s seconds ---" % utils.my_round((time.time() - start_time)))


if __name__ == "__main__":
    main()

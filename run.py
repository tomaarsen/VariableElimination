"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""
from types import FunctionType
from typing import List, Union
from read_bayesnet import BayesNet
from variable_elim import VariableElimination as VE
import sys, os, time

def output_to_file(func):
    def wrapper(*args, **kwargs):
        orig_stdout = sys.stdout
        folder = f"outputs/{args[0]}/"
        if isinstance(args[3], FunctionType):
            folder += args[3].__name__ + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}verbose_{kwargs.get('verbosity', 0)}.txt", "w+") as f:
            sys.stdout = f
            output = func(*args, **kwargs)
        sys.stdout = orig_stdout
        return output
    return wrapper

def output_times_to_file(func):
    def wrapper(*args, **kwargs):
        folder = f"outputs/{args[0]}/"
        if isinstance(args[3], FunctionType):
            folder += args[3].__name__ + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        start_t = time.time()
        output = func(*args, **kwargs)
        delta_t = time.time() - start_t
        with open(f"{folder}exec_times.txt", "a+") as f:
            f.write(f"{delta_t}\n")
        return output
    return wrapper

# @output_to_file
def test(file_name: str, query: str, evidence: dict, elim_order: Union[List[str], FunctionType], verbosity:int = 0):
    net = BayesNet(f'bif/{file_name}.bif')
    test_ve(file_name, query, evidence, elim_order, net, verbosity=verbosity)

# @output_times_to_file
def test_ve(file_name: str, query: str, evidence: dict, elim_order: Union[List[str], FunctionType], net: BayesNet, verbosity:int = 0):
    ve = VE(net, verbosity)
    print(ve.run(query, evidence, elim_order))

def test_earthquake(elim_order: Union[List[str], FunctionType] = VE.least_incoming_arcs, verbosity:int = 0):
    file_name = 'earthquake'
    query = 'Alarm'
    evidence = {'Burglary': 'True'}
    test(file_name, query, evidence, elim_order, verbosity=verbosity)

def test_sachs(elim_order: Union[List[str], FunctionType] = VE.most_incoming_arcs, verbosity:int = 0):
    file_name = 'sachs'
    query = 'Akt'
    evidence = {'Mek': 'LOW', 'Plcg': 'AVG', 'Jnk': 'HIGH'}
    test(file_name, query, evidence, elim_order, verbosity=verbosity)

def test_alarm(elim_order: Union[List[str], FunctionType] = VE.least_outgoing_arcs, verbosity:int = 0):
    file_name = 'alarm'
    query = 'MINVOLSET'
    evidence = {'VENTMACH': "LOW"}
    test(file_name, query, evidence, elim_order, verbosity=verbosity)

def main():
    """
    Should only be used with the `output_to_file` decorator applied on `test`, 
    and the `output_times_to_file` decorator not applied on `test_ve`.

    If set up in that way, generates all output files for all verbosity levels, 
    for all BN's, for all heuristics.
    """
    for func in [test_earthquake, test_sachs, test_alarm]:
        for verbosity in range(3):
            for elim_order in [VE.least_incoming_arcs, VE.most_incoming_arcs, VE.least_outgoing_arcs, VE.most_outgoing_arcs]:
                func(elim_order, verbosity=verbosity)

def test_time():
    """
    Should only be used with the `output_to_file` decorator not applied on `test`, 
    and the `output_times_to_file` decorator applied on `test_ve`.

    If set up in that way, appends to a `exec_times.txt` file with execution
    times for all BN's, for all heuristics, for verbosity level 0.
    """
    verbosity = 0
    for func in [test_earthquake, test_sachs, test_alarm]:
        for elim_order in [VE.least_incoming_arcs, VE.most_incoming_arcs, VE.least_outgoing_arcs, VE.most_outgoing_arcs]:
            for _ in range(5):
                func(elim_order, verbosity=verbosity)


if __name__ == '__main__':
    # Uncomment these only after reading the doctexts for these functions
    # main()
    # test_time()

    """
    Verbose levels:
    0 => No debug information
    1 => Some debug information
    2 => All debug information
    """
    test_earthquake(verbosity=2)
    print("-" * 50)
    test_sachs(verbosity=1)
    print("-" * 50)
    test_alarm(verbosity=0)

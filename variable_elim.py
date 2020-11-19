"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Class for the implementation of the variable elimination algorithm.

"""

from types import FunctionType
from read_bayesnet import BayesNet
from typing import Callable, List, Optional, Set, Union
from functools import reduce
import pandas as pd


class Factor(object):
    def __init__(self, ve: "VariableElimination", nodes: Set[str], df: pd.DataFrame) -> None:
        self.ve = ve
        self.nodes = nodes
        self.df = df
        self.i = self.ve.increment_i()

    def marginalize(self, node: Union[str, Set[str]]) -> None:
        """
        Sum over nodes in `node`.

        >>> factor
        f_2(Alarm, Burglary, Earthquake) =      
           Alarm Burglary Earthquake   prob     
        0   True     True       True  0.950     
        1  False     True       True  0.050     
        2   True    False       True  0.290     
        3  False    False       True  0.710     
        4   True     True      False  0.940     
        5  False     True      False  0.060     
        6   True    False      False  0.001     
        7  False    False      False  0.999

        Sum over "Alarm":
        >>> factor.marginalize("Alarm")
        f_2(Burglary, Earthquake) =    
          Burglary Earthquake  prob
        0 False    False        1.0
        1 False    True         1.0
        2 True     False        1.0
        3 True     True         1.0

        Alternatively:
        >>> factor.marginalize({"Alarm", "Burglary"})
        f_2(Earthquake) = 
          Earthquake  prob
        0 False        2.0
        1 True         2.0
        """
        if isinstance(node, str):
            node = {node}
        self.nodes -= node
        if self.nodes:
            self.df = self.df.groupby(list(self.nodes)).sum().reset_index()

    def reduce(self, node_dict: dict) -> None:
        """
        Reduce factor by a dict of known values.

        >>> factor
        f_2(Alarm, Burglary, Earthquake) =      
           Alarm Burglary Earthquake   prob     
        0   True     True       True  0.950     
        1  False     True       True  0.050     
        2   True    False       True  0.290     
        3  False    False       True  0.710     
        4   True     True      False  0.940     
        5  False     True      False  0.060     
        6   True    False      False  0.001     
        7  False    False      False  0.999

        >>> factor.reduce({'Burglary': 'True'})
          Earthquake Alarm  prob
        0 False      False  0.06
        1 False      True   0.94
        2 True       False  0.05
        3 True       True   0.95

        Alternatively:
        >>> factor.reduce({'Burglary': 'True', 'Earthquake': 'False'})
        f_2(Alarm) = 
          Alarm  prob
        0 False  0.06  
        1 True   0.94 
        """
        # Remove cases where the value for a specific column does not
        # match the observed value from `node_dict`
        relevant_nodes = set(node_dict.keys()).intersection(self.nodes)
        if relevant_nodes:
            self.df = self.df[reduce(
                lambda x, y: x & y,
                (self.df[key] == val for key, val in node_dict.items() if key in relevant_nodes))
            ]
            # Remove columns that have been fixed.
            self.df = self.df.drop(columns=relevant_nodes)
            self.nodes -= relevant_nodes

    @classmethod
    def product(cls, factor_x: "Factor", factor_y: "Factor") -> "Factor":
        """
        Create new factor from the factor product between two factors.

        >>> f2
        f_2(Burglary, Alarm, Earthquake) = 
           Alarm Burglary Earthquake   prob
        0   True     True       True  0.950
        1  False     True       True  0.050
        2   True    False       True  0.290
        3  False    False       True  0.710
        4   True     True      False  0.940
        5  False     True      False  0.060
        6   True    False      False  0.001
        7  False    False      False  0.999

        >>> f3
        f_3(JohnCalls, Alarm) = 
          JohnCalls  Alarm  prob
        0      True   True  0.90
        1     False   True  0.10
        2      True  False  0.05
        3     False  False  0.95

        >>> Factor.product(f2, f3)
        f_5(Burglary, JohnCalls, Alarm, Earthquake) =   
            Alarm Burglary Earthquake JohnCalls     prob
        0    True     True       True      True  0.85500
        1    True     True       True     False  0.09500
        2    True    False       True      True  0.26100
        3    True    False       True     False  0.02900
        4    True     True      False      True  0.84600
        5    True     True      False     False  0.09400
        6    True    False      False      True  0.00090
        7    True    False      False     False  0.00010
        8   False     True       True      True  0.00250
        9   False     True       True     False  0.04750
        10  False    False       True      True  0.03550
        11  False    False       True     False  0.67450
        12  False     True      False      True  0.00300
        13  False     True      False     False  0.05700
        14  False    False      False      True  0.04995
        15  False    False      False     False  0.94905
        """
        new_df = factor_x.df.merge(factor_y.df, on=list(
            factor_x.nodes.intersection(factor_y.nodes)))
        new_df["prob"] = new_df["prob_x"] * new_df["prob_y"]
        new_df = new_df.drop(columns=["prob_x", "prob_y"])
        return cls(factor_x.ve, factor_x.nodes.union(factor_y.nodes), new_df)

    def normalize(self):
        """
        Normalize the reduced factor so it sums to 1.

        >>> factor
        f_18(Akt) = 
            Akt      prob
        0   AVG  0.009226
        1  HIGH  0.000229
        2   LOW  0.015517

        >>> factor.normalize()
        f_18(Akt) = 
            Akt      prob
        0   AVG  0.369440
        1  HIGH  0.009173
        2   LOW  0.621387
        """
        self.df["prob"] /= self.df["prob"].sum()

    def __str__(self) -> str:
        return f"f_{self.i}({', '.join(self.nodes)})"

    def __repr__(self) -> str:
        return f"f_{self.i}({', '.join(self.nodes)}) = \n{self.df}\n"


class VariableElimination():

    def __init__(self, network, verbose: int = 0):
        """
        Initialize the variable elimination algorithm with the specified network.
        """
        self.network = network
        """
        Verbose levels:
        0 => No debug information
        1 => Some debug information
        2 => All debug information
        """
        self.verbose = verbose

        self.i = -1

    def increment_i(self):
        self.i += 1
        return self.i

    def run(self, query: str, observed: dict, elim_order: Union[List[str], Callable[["BayesNet"], List[str]]]) -> dict:
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable
        """
        self.log(f"Verbosity level: {self.verbose}", 1)
        self.log(f"Query Variable: {query!r}", 1)
        self.log(f"Observed Variables: {observed!r}\n", 1)

        # Convert function that determines ordering into a list of nodes
        if isinstance(elim_order, FunctionType):
            self.log(f"Elimination ordering function: {elim_order.__name__}.", 1)
            elim_order = elim_order(self.network)
            self.log(f"Elimination variable ordering: {elim_order}.\n", 1)

        # Create a list of Factor objects *before* reducing observed variables
        factors = [
            Factor(
                self, {node, *set(self.network.parents[node])}, self.network.probabilities[node])
            for node in self.network.nodes
        ]
        self.log("Identified Factors, before reducing observed variables", 1)
        self.log('\n'.join(str(factor) for factor in factors), 1, only=True)
        self.log('\n'.join(repr(factor) for factor in factors), 2)

        # Reduce factors by observed variables.
        for factor in factors:
            factor.reduce(observed)

        # Remove (now) empty factors.
        factors = [factor for factor in factors if factor.nodes]

        self.log("\nIdentified Factors, after reducing observed variables", 1)
        self.log('\n'.join(str(factor) for factor in factors), 1, only=True)
        self.log('\n'.join(repr(factor) for factor in factors), 2)

        # Remove query and observed variables from elimination ordering
        elim_order = [node for node in elim_order
                      if node != query and node not in observed]
        self.log(f"\nElimination order:\n{elim_order}", 1)
        for i, node in enumerate(elim_order):
            self.log(f"{self.get_p(query, observed)} = {'sum_' + str(elim_order[i:]) if elim_order[i:] else ''}{''.join(str(factor) for factor in factors)}\n", 1)
            self.log(f"Variable {node!r} is being eliminated.\n", 1)
            # Get factors that use `node`
            filtered_factors = [factor for factor in factors
                                if node in factor.nodes]

            if filtered_factors:
                # Multiply factors containing `node`
                product_factor = reduce(Factor.product, filtered_factors)
                self.log((("".join(str(factor) for factor in filtered_factors) + " = ") if len(filtered_factors) > 1 else "") +
                         f"{product_factor!r}", 2)

                # Sum out `node`
                product_factor.marginalize(node)
                self.log(f"After summing out {node!r}: {product_factor!r}", 2)

                # Remove the multiplied factors, and add the new one,
                # unless the new one is empty
                factors = [factor for factor in factors
                           if factor not in filtered_factors]
                if product_factor.nodes:
                    factors.append(product_factor)
            self.log(f"Variable {node!r} has been eliminated.\n", 1)

        self.log(f"{self.get_p(query, observed)} = {''.join(str(factor) for factor in factors)}\n", 1)
        
        # Multiply remaining factors, i.e. factors with `query`
        final_factor = reduce(Factor.product, factors)
        self.log((("".join(str(factor) for factor in factors) + " = ") if len(factors) > 1 else "") +
                  f"{final_factor!r}", 2)

        # Normalize this factor
        final_factor.normalize()
        self.log(f"Now normalize the final vector.", 1)

        return {query: final_factor.df}

    def get_p(self, query: str, observed: Optional[dict]):
        out = f"P({query}"
        if observed:
            out += "|"
            out += ', '.join(key + ' = ' + val for key, val in observed.items())
        return out + ")"

    @staticmethod
    def incoming_arcs(network: "BayesNet", reverse: bool):
        return sorted(network.parents, key=lambda x: len(network.parents[x]), reverse=reverse)

    @staticmethod
    def least_incoming_arcs(network: "BayesNet"):
        """
        Ordering prioritising nodes with the least parents.
        """
        return VariableElimination.incoming_arcs(network, reverse=False)

    @staticmethod
    def most_incoming_arcs(network: "BayesNet"):
        """
        Ordering prioritising nodes with the most parents.
        """
        return VariableElimination.incoming_arcs(network, reverse=True)

    @staticmethod
    def outgoing_arcs(network: "BayesNet", reverse: bool):
        flattened_values = [entry for value in network.parents.values() for entry in value]
        return sorted(network.parents, key=lambda x: flattened_values.count(x), reverse=reverse)

    @staticmethod
    def least_outgoing_arcs(network: "BayesNet"):
        """
        Ordering prioritising nodes with the least children.
        i.e. prioritises leaf nodes.
        """
        return VariableElimination.outgoing_arcs(network, reverse=False)

    @staticmethod
    def most_outgoing_arcs(network: "BayesNet"):
        """
        Ordering prioritising nodes with the most children.
        """
        return VariableElimination.outgoing_arcs(network, reverse=True)

    def log(self, out: str, level: int, only: bool = False):
        """
        Print `out` if verbosity is higher or equal to `level`,
        unless `only` is true, in which case verbosity must equal `level`.
        """
        if self.verbose == level and only or self.verbose >= level and not only:
            print(out)

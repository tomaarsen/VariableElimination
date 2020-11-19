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
        """
        if isinstance(node, str):
            node = {node}
        self.nodes -= node
        if self.nodes:
            self.df = self.df.groupby(list(self.nodes)).sum().reset_index()

    def reduce(self, node_dict: dict) -> None:
        """
        Reduce factor by a dict of known values.
        """
        relevant_nodes = set(node_dict.keys()).intersection(self.nodes)
        if relevant_nodes:
            self.df = self.df[reduce(
                lambda x, y: x & y,
                (self.df[key] == val for key, val in node_dict.items() if key in relevant_nodes))
            ]
            self.df = self.df.drop(columns=relevant_nodes)
            self.nodes -= relevant_nodes

    @classmethod
    def product(cls, factor_x: "Factor", factor_y: "Factor") -> "Factor":
        """
        Create new factor from the factor product between two factors.
        """
        new_df = factor_x.df.merge(factor_y.df, on=list(
            factor_x.nodes.intersection(factor_y.nodes)))
        new_df["prob"] = new_df["prob_x"] * new_df["prob_y"]
        new_df = new_df.drop(columns=["prob_x", "prob_y"])
        return cls(factor_x.ve, factor_x.nodes.union(factor_y.nodes), new_df)

    def normalize(self):
        """
        Normalize the reduced factor so it sums to 1.
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

        `verbose` keyword exists for compatibility.
        """
        self.network = network

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
        # Convert function that determines ordering into a list of nodes
        if isinstance(elim_order, FunctionType):
            elim_order = elim_order(self.network)

        # Create a list of Factor objects *before* reducing observed variables
        factors = [
            Factor(
                self, {node, *set(self.network.parents[node])}, self.network.probabilities[node])
            for node in self.network.nodes
        ]

        # Reduce factors by observed variables.
        for factor in factors:
            factor.reduce(observed)

        # Remove (now) empty factors.
        factors = [factor for factor in factors if factor.nodes]

        # Remove query and observed variables from elimination ordering
        elim_order = [node for node in elim_order
                      if node != query and node not in observed]
        for i, node in enumerate(elim_order):
            # Get factors that use `node`
            filtered_factors = [factor for factor in factors
                                if node in factor.nodes]

            if filtered_factors:
                # Multiply factors containing `node`
                product_factor = reduce(Factor.product, filtered_factors)

                # Sum out `node`
                product_factor.marginalize(node)

                # Remove the multiplied factors, and add the new one,
                # unless the new one is empty
                factors = [factor for factor in factors
                           if factor not in filtered_factors]
                if product_factor.nodes:
                    factors.append(product_factor)
        
        # Multiply remaining factors, i.e. factors with `query`
        final_factor = reduce(Factor.product, factors)

        # Normalize this factor
        final_factor.normalize()

        return {query: final_factor.df}

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

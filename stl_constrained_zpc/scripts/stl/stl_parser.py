import re


class STLParser:
    def __init__(self, formula: str):
        """
        Decomposes an STL formula into atomic predicates.
        The format of the STL formula is assumed to be as follows:
            Operator_(Condition1; Condition2; ...; ConditionN)
        where each condition is of the form:
            Operator [Time1, Time2] (Variable Operator Value)
        
        Args:
            formula (str): STL formula to be decomposed.

        Example:
            formula = "G [0,5] (vx<=0.1)"

        Returns:
            list: Decomposed atomic predicates.
        """
        self.formula = formula
        self.decomposed = self._decompose()
    
    def _decompose(self, formula=None):
        """
        Extracts atomic predicates from STL formula. 

        Args:
            formula (str): STL formula to be decomposed. Format is assumed to be as follows:
                Operator_(Condition1; Condition2; ...; ConditionN)

        
        Returns:
            list: List of atomic predicates. Each atomic predicate is of the form:
                Operator [Time1, Time2] (Variable Operator Value)
        """
        if formula is None:
            formula = self.formula
        
        # Extract operator and conditions
        atomic_predicates = []
        match = re.search(r'([A-Z]+)_\((.*)\)', formula)
        if match:
            operator, conditions = match.groups()
            conditions = conditions.split(";")
            for condition in conditions:
                atomic_predicates.append(f"{operator} {condition.strip()}")
        else:
            atomic_predicates.append(formula)

        return atomic_predicates
    
    def get_atomic_conditions(self):
        """
        Returns decomposed atomic predicates.

        Returns:
            list: List of atomic predicates. Each atomic predicate is of the form:
                Operator [Time1, Time2] (Variable Operator Value)
        """
        return self.decomposed
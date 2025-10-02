#!/usr/bin/env python3
import argparse, csv, sys
from typing import Dict, List, Optional, Tuple
from math import isqrt

Assignment = Dict[str, int]

DOMAIN = range(1, 121)

# helper functions

def in_dom(x: int) -> bool:
    return 1 <= x <= 120

def is_int_div(num: int, den: int) -> bool:
    return den != 0 and num % den == 0

def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def feasible_partial(assign: Assignment, problem: str) -> bool:
    """Check if partial assignments might still lead to a full solution.
    Does not assign new variables; just checks if current values make sense.
    """
    # get current values
    B = assign.get("B")
    C = assign.get("C")
    E = assign.get("E")

    # check if A and D are still in range
    if B is not None and C is not None:
        A = derive_A(B, C)
        D = derive_D(B, C)
        if not (in_dom(A) and in_dom(D)):
            return False

    # if B,C,E are known, F must work out evenly
    if B is not None and C is not None and E is not None:
        F = derive_F(B, C, E)
        if F is None:
            return False

    # check C+E > B
    if B is not None and C is not None and E is not None:
        if not (C + E > B):
            return False

    # divisibility check with C and I
    I = assign.get("I")
    if C is not None and I is not None:
        num = (C + I) * (C + I)
        den = I + 3
        if den == 0 or num % den != 0:
            return False
        # if B and E known, check exact equality
        if B is not None and E is not None:
            if num != B * E * den:
                return False

    # check perfect square condition with G and I
    G = assign.get("G")
    if G is not None and I is not None:
        s = (G + I) ** 3 - 4
        if s < 0 or not is_perfect_square(s):
            return False

    # check G + I < E + 3
    if G is not None and I is not None and E is not None:
        if G + I >= E + 3:
            return False
    elif G is not None and I is not None and E is None:
        # check if some E in range can satisfy inequality
        if (G + I - 2) > 120:
            return False
    elif E is not None:
        # if one of G or I known, check minimum possible sum
        if G is not None and (G + 1) >= E + 3:
            return False
        if I is not None and (I + 1) >= E + 3:
            return False

    # check D + H > 180
    H = assign.get("H")
    if H is not None and B is not None and C is not None:
        D = derive_D(B, C)
        if not (D + H > 180):
            return False
    elif H is not None and (B is None or C is None):
        # check if some D can satisfy inequality
        if H <= 60:
            return False
    elif H is None and B is not None and C is not None:
        # check if some H can satisfy inequality
        D = derive_D(B, C)
        if D <= 60:
            return False

    # for problem B/C, check bounds on J if possible
    if problem in ("B", "C"):
        have_BE = (B is not None and E is not None and C is not None)
        have_GHI = (G is not None and assign.get("H") is not None and I is not None)
        if have_BE and have_GHI:
            F = derive_F(B, C, E)
            if F is None:
                return False
            D = derive_D(B, C)
            J = assign.get("J")
            lower = assign["H"] + E + F + G + I + 1  # from C12 (strict >)
            upper = D + E + F - 1                    # from C11 (strict <)
            # if J present, check bounds; if not, check if interval is possible
            if J is not None:
                if not (J > (lower - 1) and J < (upper + 1)):
                    return False
            else:
                if lower > upper or lower > 120 or upper < 1:
                    return False

    # extra checks for problem C
    if problem == "C":
        K = assign.get("K")
        L = assign.get("L")
        M = assign.get("M")
        Jv = assign.get("J")
        Hv = assign.get("H")
        Gv = assign.get("G")
        # if K,B,L known and M unknown, check if M can be integer in range
        if K is not None and B is not None and L is not None and M is None:
            rhs = B * (K + 5)
            den = K * L
            if den == 0 or rhs % den != 0:
                return False
            m = rhs // den
            if not in_dom(m):
                return False
        # if K,B,L,M known, check equality
        if K is not None and B is not None and L is not None and M is not None:
            if K * L * M != B * (K + 5):
                return False
        # if K,L,B,C,E known, check F condition
        if K is not None and L is not None and B is not None and C is not None and E is not None:
            F = derive_F(B, C, E)
            if F is None:
                return False
            if F ** 3 != K * K * (L - 29) + 25:
                return False
        # if H,L,G,M known, check relation
        if Hv is not None and L is not None and Gv is not None and M is not None:
            if Hv * M * M != L * Gv - 3:
                return False
        # if J,M,L,E,G known, check relation
        if Jv is not None and M is not None and L is not None and E is not None and Gv is not None:
            if Jv + M != (L - 15) * (E + Gv):
                return False
        # if K,J,L known, check relation
        if K is not None and Jv is not None and L is not None:
            if K ** 3 != (Jv - 4) * (L - 20):
                return False

    return True

# calculate A from B and C
def derive_A(B: int, C: int) -> int:
    # A = B^2 - C^2
    return B*B - C*C

# calculate D from B and C
def derive_D(B: int, C: int) -> int:
    # D = 4*C^2 - 3*B^2
    return 4*C*C - 3*B*B

# calculate F from B, C, E if possible
def derive_F(B: int, C: int, E: int) -> Optional[int]:
    # F = ((B-C)^2 + 396) / (E*B) if divisible and in range
    num = (B - C)*(B - C) + 396
    den = E * B
    if not is_int_div(num, den):
        return None
    f = num // den
    if not in_dom(f):
        return None
    return f

# check all constraints for problem A
def check_problem_A_full(assign: Assignment) -> bool:
    """
    Given B,C,E and any extras, compute A,D,F and check all rules for A.
    """
    B = assign["B"]; C = assign["C"]; E = assign["E"]

    A = derive_A(B, C)
    D = derive_D(B, C)
    F = derive_F(B, C, E)
    if F is None: return False

    # check A and D in range
    if not (in_dom(A) and in_dom(D)):
        return False

    # check constraints C1..C5
    # C1: A = B^2 - C^2 (by construction)
    # C2: C + E > B
    if not (C + E > B): return False
    # C3: D = B^2 - 4A (check equality)
    if D != B*B - 4*A: return False
    # C4: (B - C)^2 = E*F*B - 396
    if (B - C)*(B - C) != E*F*B - 396: return False
    # C5: sum less than 125
    if not (C + D + E + F < 125): return False

    # save computed values
    assign["A"] = A; assign["D"] = D; assign["F"] = F
    return True

# check all constraints for problem B
def check_problem_B_full(assign: Assignment) -> bool:
    if not check_problem_A_full(assign):
        return False

    A = assign["A"]; B = assign["B"]; C = assign["C"]
    D = assign["D"]; E = assign["E"]; F = assign["F"]
    G = assign["G"]; H = assign["H"]; I = assign["I"]; J = assign["J"]

    # check constraints C6..C12
    if (G + I)**3 - 4 != (H - A)**2: return False
    if C*E*F + 40 != (H - F - I) * (I + G): return False
    if (C + I)**2 != B * E * (I + 3): return False
    if not (G + I < E + 3): return False
    if not (D + H > 180): return False
    if not (J < D + E + F): return False
    if not (J > H + E + F + G + I): return False
    return True

# check all constraints for problem C
def check_problem_C_full(assign: Assignment) -> bool:
    if not check_problem_B_full(assign):
        return False

    B = assign["B"]; E = assign["E"]; F = assign["F"]
    G = assign["G"]; H = assign["H"]; I = assign["I"]; J = assign["J"]
    K = assign["K"]; L = assign["L"]; M = assign["M"]

    # check constraints C13..C17
    if K * L * M != B * (K + 5): return False
    if F**3 != K*K*(L - 29) + 25: return False
    if H * M * M != L * G - 3: return False
    if J + M != (L - 15) * (E + G): return False
    if K**3 != (J - 4) * (L - 20): return False
    return True

# backtracking solver with elimination and early checks
class Solver:
    def __init__(self, base_vars: List[str], csv_path: str, problem: str):
        self.base_vars = base_vars[:]              # variables to assign
        self.csv_path = csv_path
        self.problem = problem                     # "A" | "B" | "C"
        self.nva = 0
        self.solution: Optional[Assignment] = None

    def _order_vars(self) -> List[str]:
        # fixed order of variables
        return self.base_vars

    def _try(self, idx: int, assign: Assignment) -> Optional[Assignment]:
        if idx == len(self.base_vars):
            # all assigned, check full solution
            full = dict(assign)  # copy
            ok = {
                "A": check_problem_A_full,
                "B": check_problem_B_full,
                "C": check_problem_C_full,
            }[self.problem](full)
            if ok:
                return full
            return None

        var = self.base_vars[idx]
        for val in DOMAIN:
            self.nva += 1
            assign[var] = val
            # quit early if current numbers can't work
            if not feasible_partial(assign, self.problem):
                del assign[var]
                continue
            res = self._try(idx + 1, assign)
            if res is not None:
                return res
            del assign[var]
        return None

    def solve(self) -> Tuple[Optional[Assignment], int]:
        self.solution = self._try(0, {})
        self._write_csv()
        return self.solution, self.nva

    def _write_csv(self):
        # prepare all variables for output
        if self.problem == "A":
            all_vars = ["A","B","C","D","E","F"]
        elif self.problem == "B":
            all_vars = ["A","B","C","D","E","F","G","H","I","J"]
        else:
            all_vars = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]

        row = {v: None for v in all_vars}
        if self.solution:
            row.update({k:v for k,v in self.solution.items() if k in row})
            # fill in eliminated vars if missing
            if "B" in self.solution and "C" in self.solution and "E" in self.solution:
                row["A"] = row["A"] if row["A"] is not None else derive_A(self.solution["B"], self.solution["C"])
                dtmp = derive_D(self.solution["B"], self.solution["C"])
                row["D"] = row["D"] if row["D"] is not None else dtmp
                ftmp = derive_F(self.solution["B"], self.solution["C"], self.solution["E"])
                if ftmp is not None and row["F"] is None:
                    row["F"] = ftmp
        row["nva"] = self.nva

        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

# create solver with variables for each problem
def build_solver(problem: str, csv_path: str) -> Solver:
    if problem == "A":
        # assign B, C, E only
        base_vars = ["B", "C", "E"]
    elif problem == "B":
        # assign B, C, E, G, H, I, J
        base_vars = ["B", "C", "E", "G", "H", "I", "J"]
    else:
        # assign B, C, E, G, H, I, J, K, L, M
        base_vars = ["B", "C", "E", "G", "H", "I", "J", "K", "L", "M"]
    return Solver(base_vars, csv_path, problem)

# command line interface
def main():
    p = argparse.ArgumentParser(description="CSP solver with elimination + early propagation.")
    p.add_argument("problem", choices=["A","B","C"], help="Which problem to solve.")
    p.add_argument("--csv", default=None, help="CSV output path (default: Task 2 <problem>.csv)")
    args = p.parse_args()

    csv_path = args.csv or f"Task 2 {args.problem}.csv"
    solver = build_solver(args.problem, csv_path)
    sol, nva = solver.solve()

    if sol is None:
        print(f"[{args.problem}] no solution exists")
        print(f"nva = {nva}")
        print(f"CSV written to: {csv_path}")
        return

    print(f"[{args.problem}] solution found with nva={nva}")
    # print variables in order
    if args.problem == "A":
        order = ["A","B","C","D","E","F"]
    elif args.problem == "B":
        order = ["A","B","C","D","E","F","G","H","I","J"]
    else:
        order = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
    ordered = {v: sol.get(v) for v in order}
    print(ordered)
    print(f"CSV written to: {csv_path}")

if __name__ == "__main__":
    sys.setrecursionlimit(10_000_000)
    main()
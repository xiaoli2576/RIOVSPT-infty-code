"""Small local check; does not replace or overwrite the recorded study."""
import warnings
from scipy.optimize import OptimizeWarning
from study_instances import named_cases
from study_runner import solve_lp, validate_solution, TOL
from algorithms import solve_cases, solve_uniform

warnings.filterwarnings('ignore', category=OptimizeWarning)
cases = named_cases()
for ins in cases:
    lp = solve_lp(ins)
    assert lp['status'] in ('optimal', 'infeasible'), (ins.label, lp)
    if lp['status'] == 'optimal':
        assert validate_solution(ins, lp['x'], lp['objective'])['valid'], ins.label
    for solver in (solve_cases, solve_uniform):
        result = solver(ins)
        assert result['status'] == lp['status'], (ins.label, solver.__name__, result, lp)
        if result['status'] == 'optimal':
            assert validate_solution(ins, result['x'], result['objective'])['valid'], ins.label
            assert abs(result['objective'] - lp['objective']) <= TOL, ins.label
print(f'PASS: {len(cases)} named cases; both algorithms agree with LP and pass feasibility checks.')

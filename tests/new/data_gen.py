import random


from uncertainties.new.umath import float_funcs_dict


no_other_list = [
    "__abs__",
    "__pos__",
    "__neg__",
    "__trunc__",
]

for func in float_funcs_dict:
    vals = []
    first = random.uniform(-2, +2)
    vals.append(first)
    if func not in no_other_list:
        second = random.uniform(-2, +2)
        vals.append(second)
    vals = tuple(vals)
    unc = random.uniform(-1, 1)

    print(f"(\"{func}\", {vals}, {unc}),")

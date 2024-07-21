import numpy as np

from uncertainties.new import UArray, UFloat

UFloat.__repr__ = lambda self: f'{self.n:.3f} +/- {self.s:.3f}'

x = UFloat(1, 0.1)
y = UFloat(2, 0.2)
z = UFloat(3, 0.3)

print('')
print("## Numpy operations (ufuncs) work on scalars (UFloat) ##")
print(f'{x=}')
print(f'{np.sin(x)=}')
print(f'{np.exp(x)=}')
print('')

print("## Constructing a UArray from an \"array\" of UFloat and looking at its properties ##")
uarr = UArray([x, y, z])
print(f'{uarr=}')
print(f'{uarr.nominal_value=}')
print(f'{uarr.std_dev=}')
print(f'{uarr.uncertainty=}')
print('')

print("## Constructing a UArray from 2 \"arrays\" of floats and looking at its properties ##")
uarr = UArray.from_val_arr_std_dev_arr([1, 2, 3], [0.1, 0.2, 0.3])
print(f'{uarr=}')
print(f'{uarr.nominal_value=}')
print(f'{uarr.std_dev=}')
print(f'{uarr.uncertainty=}')
print('')

print("## Binary operations with varying types")
narr = np.array([10, 20, 30])

print("# UArray : UArray")
print(f'{(uarr + uarr)=}')
print(f'{(uarr - uarr)=}')

print("# UArray : ndarray")
print(f'{(uarr + narr)=}')
print("# ndarray : UArray")
print(f'{(narr - uarr)=}')

print("# UFloat: UArray #")
print(f"{x * uarr}")
print("# UArray: UFloat #")
print(f"{uarr * x}")

print("# float : UArray #")
print(f"{42 * uarr}")
print("# UArray: float #")
print(f"{uarr * 42}")
print('')

print('## Numpy broadcasting works ##')
uarr1 = UArray.from_val_arr_std_dev_arr([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.ones((3, 3)))
uarr2 = UArray.from_val_arr_std_dev_arr([100, 1000, 1000], [10, 10, 10])
print(f'{uarr1=}')
print(f'{uarr2=}')
print(f'{(uarr1 + uarr2)=}')
print(f'{(uarr1 + uarr2).shape=}')
print('')

print('## More ufuncs work ##')

print('# np.mean #')
print(f'{np.mean(uarr1)=}')
print(f'{np.mean(uarr1, axis=0)=}')
print(f'{np.mean(uarr2)=}')

print('# other ufuncs #')
print(f'{np.exp(uarr1)=}')
print(f'{np.sin(uarr1)=}')
print(f'{np.sqrt(uarr1)=}')
print(f'{np.hypot(uarr1, uarr2)=}')
print(f'{np.hypot(uarr1, uarr2).shape=}')

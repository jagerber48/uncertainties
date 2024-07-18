import numpy as np

# TODO: We need this import to execute the code that actually adds the ufuncs to UFloat.
#   This can and should be refactored to be cleaner. I recommend adding the ufuncs to
#   UFloat in uncertainties/__init__.py. That way UFloat will always have these methods
#   defined whether or not numpy is present
from uncertainties import umath_new
from uncertainties.core_new import UFloat


class UArray(np.ndarray):
    def __new__(cls, input_array) -> "UArray":
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        # Finally, we must return the newly created object:
        return obj

    @property
    def nominal_value(self):
        return np.array(np.vectorize(lambda uval: uval.value)(self), dtype=float)

    @property
    def std_dev(self):
        return np.array(np.vectorize(lambda uval: uval.std_dev)(self), dtype=float)

    @property
    def uncertainty(self):
        return np.array(np.vectorize(lambda uval: uval.uncertainty)(self), dtype=object)

    @classmethod
    def from_val_arr_std_dev_arr(cls, val_arr, std_dev_arr):
        return cls(np.vectorize(UFloat)(val_arr, std_dev_arr))

    def __str__(self):
        return f"{self.__class__.__name__}({super().__str__()})"

import numpy as np

# TODO: We need this import to execute the code that actually adds the ufuncs to UFloat.
#   This can and should be refactored to be cleaner. I recommend adding the ufuncs to
#   UFloat in uncertainties/__init__.py. That way UFloat will always have these methods
#   defined whether or not numpy is present
from uncertainties import umath_new
from uncertainties.core_new import UFloat


class UArray(np.ndarray):
    def __new__(cls, input_array) -> "UArray":
        obj = np.asarray(input_array).view(cls)
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

    def mean(self, axis=None, dtype=None, out=None, keepdims=None, *, where=None):
        """
        Include to fix an issue where np.mean is returning a singleton 0d array rather
        than scalar.
        """
        args = [axis, dtype, out]
        if keepdims is not None:
            args.append(keepdims)
        kwargs = {}
        if where is not None:
            kwargs['where'] = where
        result = super().mean(*args, **kwargs)

        if result.ndim == 0:
            return result.item()
        return result

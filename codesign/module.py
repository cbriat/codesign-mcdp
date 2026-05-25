"""
Class-based declarative design problems.

A :class:`Module` is a thin convenience class that lets you define a design
problem by subclassing and providing class-level ``F`` and ``R`` dicts plus
an ``h`` method. The constructor wires everything into the underlying
:class:`~codesign.dp.DesignProblem` machinery.

Example
-------

.. code-block:: python

    from codesign import Module, Reals

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass":     Reals(unit="kg")}

        def h(self, f):
            return {"mass": f["capacity"] / 1.8e6}

    battery = Battery()
    # battery is a DesignProblem ready to pass to System.add(...)

Parameterised modules
---------------------

To define a parameterised module, override ``__init__`` and call
``super().__init__()`` at the end:

.. code-block:: python

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass":     Reals(unit="kg")}

        def __init__(self, specific_energy=1.8e6):
            self.specific_energy = specific_energy
            super().__init__()

        def h(self, f):
            return {"mass": f["capacity"] / self.specific_energy}

    bat_old = Battery(specific_energy=1.6e6)
    bat_new = Battery(specific_energy=2.0e6)
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

from .antichains import Antichain
from .dp import DesignProblem
from .posets import Ports, Poset


# Module sentinel: the class attributes F and R are dicts (not Portss).
# The instance attributes (set in __init__) shadow them with Portss.


class Module(DesignProblem):
    """Base class for declarative design problems.

    Subclasses must define:

    - class attribute ``F``: ``dict[str, Poset]`` declaring functionality ports,
    - class attribute ``R``: ``dict[str, Poset]`` declaring resource ports,
    - instance method ``h(self, f)``: takes a functionality dict, returns
      either a dict (treated as singleton antichain), a list of dicts
      (multi-valued antichain), or an :class:`Antichain` directly.

    Subclasses may optionally set a class attribute ``module_name`` overriding
    the default name (derived from the class name).
    """

    # Class-level declarations, overridden by subclasses. These get
    # shadowed by Ports instance attributes after __init__ runs.
    F: Dict[str, Poset] = {}
    R: Dict[str, Poset] = {}
    module_name: str = None  # type: ignore

    def __init__(self):
        F_decl = type(self).F
        R_decl = type(self).R
        if not F_decl and not R_decl:
            raise ValueError(
                f"{type(self).__name__} must declare class-level F and R "
                f"dicts before calling super().__init__()."
            )

        F_poset: Ports = (
            F_decl if isinstance(F_decl, Ports)
            else Ports(dict(F_decl))
        )
        R_poset: Ports = (
            R_decl if isinstance(R_decl, Ports)
            else Ports(dict(R_decl))
        )

        cls_name = type(self).module_name or type(self).__name__.lower()

        # Shadow the class-level dicts with Ports instance attributes.
        self.F = F_poset
        self.R = R_poset
        self.name = cls_name

    # The framework calls dp.h(f). We provide a default that delegates to
    # the user's h after wrapping the result. Subclasses override h
    # directly with the user-facing signature h(self, f).
    def h(self, f: Dict[str, Any]) -> Antichain:
        raise NotImplementedError(
            f"{type(self).__name__} must override h(self, f)."
        )

    # When the framework calls h, the Module subclass's h shadows this
    # implementation. We provide a small adapter that wraps non-Antichain
    # return values for convenience.
    @staticmethod
    def _wrap_result(R_poset: Ports, result: Any) -> Antichain:
        if isinstance(result, Antichain):
            return result
        if isinstance(result, dict):
            return Antichain.singleton(R_poset, result)
        if isinstance(result, (list, tuple)):
            return Antichain.from_set(R_poset, list(result))
        raise TypeError(
            f"Module.h must return a dict, list of dicts, or Antichain; "
            f"got {type(result).__name__}"
        )

    def __init_subclass__(cls, **kwargs):
        """Wrap a user-defined h so its return value is normalised to an
        Antichain, while keeping the user-facing signature ``h(self, f)``."""
        super().__init_subclass__(**kwargs)
        user_h = cls.__dict__.get("h", None)
        if user_h is None or getattr(user_h, "_module_wrapped", False):
            return

        def wrapped_h(self, f, _user_h=user_h):
            result = _user_h(self, f)
            return Module._wrap_result(self.R, result)

        wrapped_h._module_wrapped = True
        wrapped_h.__name__ = "h"
        wrapped_h.__qualname__ = f"{cls.__qualname__}.h"
        wrapped_h.__doc__ = user_h.__doc__
        cls.h = wrapped_h

LAVA-R
------
Nonlinear recursive system identification using latent variables. Returns a predictor on the form

:math:`\hat{y} = \Theta \varphi(t) + Z \gamma (t)`

where an ARX formulation is implemented as the nominal model :math:`\Theta \varphi(t)`, and the
residual prediction error :math:`\varepsilon (t)` is modeled by the latent variable model
:math:`Z\gamma(t)`.

The theoretical background can be found in https://arxiv.org/abs/1606.04366 by
Per Mattsson, Dave Zachariah and Petre Stoica.

Installation
============
The simplest way is to install with ``pip``. Just make sure to call it like ``pip install /path/to/package/``.
Dont forget you can point to a git location! :D
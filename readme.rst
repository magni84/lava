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
++++++++++++
The simplest way to install LAVA-R is via ``pip install git+https://github.com/el-hult/lava.git``.
This way ``pip`` will handle all the details regarding installation path and so on.

Then one simply uses the library like

.. code-block:: python

    import lava.core as lava
    model = lava.LavaLaplace(...)

Check the files in the ``./examples/`` folder to get an idea about how to use the model.

Development
+++++++++++
If you want to contribute to the code, please do so! One pointer that might be of use:

To build the documentation, make sure that sphinx and its theme is properly installed.
Then clean and build via the sphinx make files.

If you are using conda for management of virtual environment, running on windows, one convenient way to do it is as below.

.. code-block::

    conda install --file requirements.txt
    cd docs
    make.bat clean && make.bat html

If you don't use conda, you might have to deal with path variables to make sure ``sphinx-build`` resolves nicely.

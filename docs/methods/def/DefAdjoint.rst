.. _DefAdjoint:

Definition of Term: Adjoint
===========================

An adjoint is an extension to a :ref:`simulator<DefSimulator>` which
produces derivatives of the simulator output with respect to its inputs.

Technically, an adjoint is the transpose of a tangent linear
approximation; written either by hand or with the application of
automatic differentiation, it produces partial derivatives in addition
to the standard output of the simulator. An adjoint is computationally
more expensive to run than the standard simulator, but efficiency is
achieved in comparison to the finite differences method to generating
derivatives. This is because where computing partial derivatives by
finite differences requires at least two runs of the simulator for each
input parameter, the corresponding derivatives can all be generated by
one single run of an appropriate adjoint.

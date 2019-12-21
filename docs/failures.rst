=====================
Dealing with failures
=====================

In general, when running fireworks workflows you can encounter two different kind
of failures. In the following we will distinguish the possible failures in
**soft and hard failures**. We will call a failure *soft* if a firework
is in the ``FIZZLED`` state. Conversely we will identify a failure as *hard* if
the job was killed by some external action leaving the firework in a ``RUNNING``
state as a *lost run*. Here we will provide a guide about how to deal with
both kinds of failure.

Before proceeding with the detailed exaplanation about how to handle specific cases in Abiflows
you might want to review the basic
`tutorial on failures <https://materialsproject.github.io/fireworks/failures_tutorial.html>`_
in the original fireworks documentation.
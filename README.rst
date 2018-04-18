*************************
Bandit task for Nengo 1.4
*************************

``bandit_task_*.py`` files can be run from the
`Nengo 1.4 GUI <https://www.nengo.ai/nengo-1.4/>`_.

Each ``.py`` contains a local variable, ``directory``,
which determines if the file will be used to run an experiment,
or visualized through the GUI.
If ``directory == None``, then the network
will be added to the GUI and ready for investigation.
If a ``directory`` is set (to a string path),
then experiments will be run,
and the results output to that directory.
The directory is relative to the main Nengo directory.

The ``.py`` files in the plots directories
can be run on any machine
with Python, numpy, and Matplotlib installed.
From the command line, enter, for example,
``python single-trial-3Arm.py``.
Each plot file has a local variable ``save``
that determine if the plot will be saved
as a ``pdf`` file (if ``save == True``)
or displayed in a Matplotlib window (if ``save == False``).

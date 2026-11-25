"""How HighLevel reports cases that fail to load.

``HighLevel.run_on_directory`` and ``processed_run_on_directory`` load every case
folder in parallel. A case can fail (unreadable file, missing segmentation, a
worker that crashes or runs past ``thread_deadline_time``); it is then left out
of the returned list. The ``on_error`` argument decides how you hear about it:

* ``"warn"`` (default): issue a :class:`CaseFailureWarning` and return the
  cases that did load.
* ``"raise"``: raise :class:`CaseFailureError`.
* ``"ignore"``: say nothing (the previous behavior).

Both the warning and the error carry ``.failures``, the full list of
:class:`CaseFailure` records; the message itself only shows the first few.
"""

from typing import NamedTuple

__all__ = ["CaseFailure", "CaseFailureWarning", "CaseFailureError"]


class CaseFailure(NamedTuple):
    """One case folder that didn't make it into the results, and why."""

    folder: str
    reason: str


class CaseFailureWarning(RuntimeWarning):
    """Issued by the HighLevel run methods (``on_error="warn"``) when cases fail.

    ``.failures`` is every failed case, as a list of :class:`CaseFailure`.
    Turn it into an error for a whole script with
    ``warnings.simplefilter("error", CaseFailureWarning)``.
    """

    def __init__(self, message, failures=()):
        super().__init__(message)
        self.failures = list(failures)


class CaseFailureError(RuntimeError):
    """Raised by the HighLevel run methods (``on_error="raise"``) when cases fail.

    ``.failures`` is every failed case, as a list of :class:`CaseFailure`.
    ``.results`` is what the call would have returned -- the cases that did
    load, in the same format -- so a long run's good cases aren't lost.
    """

    # failures/results default so the exception survives pickling (e.g. raised
    # inside a ProcessPoolExecutor): unpickling calls cls(*args) with only the
    # message, then restores .failures/.results from __dict__.
    def __init__(self, message, failures=(), results=None):
        super().__init__(message)
        self.failures = list(failures)
        self.results = [] if results is None else results

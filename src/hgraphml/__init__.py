"""Public package surface for HGraphML."""

from hgraphml._version import __version__
from hgraphml.collapse import HGraphMLResult, collapse_messages

__all__ = ["HGraphMLResult", "__version__", "collapse_messages"]

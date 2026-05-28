from a import OldName as _A


class OldName:
    """b.OldName references a.OldName via _A — cross-file caller dependency."""

    def parent(self):
        return _A().hello()

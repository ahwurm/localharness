from b import OldName as _B


class OldName:
    """c.OldName references b.OldName via _B — second cross-file caller."""

    def use(self):
        return _B().parent()

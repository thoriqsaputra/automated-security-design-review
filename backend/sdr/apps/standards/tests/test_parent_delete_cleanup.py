from types import SimpleNamespace

from sdr.apps.standards.models import CategoryParameterParent
from sdr.apps.standards.routers.categories import delete_parameter_parent


class _FakeSession:
    def __init__(self, parent):
        self.parent = parent
        self.executed = []
        self.deleted = []
        self.committed = False

    def get(self, model, object_id):
        if model is CategoryParameterParent and object_id == getattr(self.parent, "id", None):
            return self.parent
        return None

    def execute(self, statement):
        self.executed.append(statement)
        return None

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed = True


def test_delete_parameter_parent_deletes_parent_and_commits():
    parent = SimpleNamespace(id=478)
    db = _FakeSession(parent)

    delete_parameter_parent(478, db=db)

    assert db.executed == []
    assert db.deleted == [parent]
    assert db.committed is True

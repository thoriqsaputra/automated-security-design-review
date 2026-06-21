from types import SimpleNamespace
from pathlib import Path

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


def test_delete_parameter_parent_removes_cfsrs_before_parent_delete():
    parent = SimpleNamespace(id=478)
    db = _FakeSession(parent)

    delete_parameter_parent(478, db=db)

    assert len(db.executed) == 1
    compiled = str(db.executed[0])
    assert "DELETE FROM standards_categorycontrolsummaryrequirement" in compiled
    assert "parent_id" in compiled
    assert db.deleted == [parent]
    assert db.committed is True


def test_cfsr_parent_relationship_declares_delete_orphan_and_passive_deletes():
    model_file = Path(__file__).resolve().parents[1] / "models" / "control_summary_requirement.py"
    content = model_file.read_text()

    assert 'cascade="all, delete-orphan"' in content
    assert "passive_deletes=True" in content

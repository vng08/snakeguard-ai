from types import SimpleNamespace

from backend.app.services.media import storage_service
from backend.app.services.prediction import prediction_log_service


class FakeQuery:
    """Giả lập query PredictionLog."""
    def __init__(self, log=None, count=0):
        self.log = log
        self._count = count

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.log

    def count(self):
        return self._count


class FakeDB:
    """Giả lập database session cho prediction log service."""

    def __init__(self, log=None, count=0):
        self.log = log
        self._count = count
        self.deleted = None
        self.executed = None

    def query(self, *args, **kwargs):
        return FakeQuery(self.log, self._count)

    def delete(self, log):
        self.deleted = log

    def execute(self, statement):
        self.executed = statement

    def commit(self):
        pass


def test_delete_prediction_log_removes_image(tmp_path, monkeypatch):
    """Kiểm tra xoá một log sẽ xoá luôn ảnh tương ứng."""
    storage_dir = tmp_path / "data/predictions"
    storage_dir.mkdir(parents=True)

    image_path = storage_dir / "test.jpg"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setattr(storage_service, "ROOT_DIR", tmp_path)

    log = SimpleNamespace(id=1, image_url="data/predictions/test.jpg")
    db = FakeDB(log=log)

    deleted = prediction_log_service.delete_prediction_log(db, 1)

    assert deleted is True
    assert not image_path.exists()
    assert db.deleted == log


def test_delete_all_prediction_logs_removes_all_images(tmp_path, monkeypatch):
    """Kiểm tra xoá toàn bộ logs sẽ xoá toàn bộ ảnh prediction."""
    storage_dir = tmp_path / "data/predictions"
    storage_dir.mkdir(parents=True)

    (storage_dir / "image1.jpg").write_bytes(b"image-1")
    (storage_dir / "image2.jpg").write_bytes(b"image-2")

    monkeypatch.setattr(storage_service.settings, "PREDICTION_STORAGE_DIR", storage_dir)

    db = FakeDB(count=2)

    deleted_count = prediction_log_service.delete_all_prediction_logs(db)

    assert deleted_count == 2
    assert list(storage_dir.iterdir()) == []
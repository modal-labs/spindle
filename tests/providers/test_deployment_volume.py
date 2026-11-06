import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from spindle.providers.modal.kv import ModalVolumeStore


class FakeVolume:
    """The slice of modal.Volume the registry uses, over a dict of file paths."""

    def __init__(self):
        self.files = {}
        self.commits = 0
        self.listdir = SimpleNamespace(aio=self._listdir)
        self.remove_file = SimpleNamespace(aio=self._remove)
        self.batch_upload = SimpleNamespace(aio=self._batch)

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    async def _listdir(self, path):
        folder = path.rstrip("/") + "/"
        names = [name for name in self.files if name.startswith(folder)]
        if not names:
            raise FileNotFoundError(path)
        return [SimpleNamespace(path=name.lstrip("/")) for name in names]

    async def _remove(self, path, recursive=False):
        if recursive:
            folder = path.rstrip("/") + "/"
            removed = [name for name in self.files if name.startswith(folder)]
            if not removed:
                raise FileNotFoundError(path)
            for name in removed:
                del self.files[name]
            return
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    @asynccontextmanager
    async def _batch(self, force=False):
        volume = self
        volume.commits += 1

        class Batch:
            def put_file(self, fileobj, remote_path):
                volume.files[remote_path] = fileobj.read()

        yield Batch()


def test_volume_store_keeps_one_json_file_per_recipe_in_its_folder():
    async def run():
        volume = FakeVolume()
        store = ModalVolumeStore(volume, "ap-one")
        assert await store.list_keys("") == ()
        assert await store.get("missing") is None
        await store.put("qwen-lora", {"config": {"recipe": {"name": "qwen-lora"}}})
        await store.put("qwen-fft", {"config": {"recipe": {"name": "qwen-fft"}}})
        assert set(volume.files) == {"/ap-one/qwen-lora.json", "/ap-one/qwen-fft.json"}
        assert volume.files["/ap-one/qwen-lora.json"].startswith(b"{")
        assert store.read("qwen-lora") == {"config": {"recipe": {"name": "qwen-lora"}}}
        assert await store.list_keys("qwen") == ("qwen-fft", "qwen-lora")
        assert [key for key, _ in await store.list_items("")] == [
            "qwen-fft",
            "qwen-lora",
        ]
        await store.delete("qwen-fft")
        await store.delete("qwen-fft")
        assert await store.list_keys("") == ("qwen-lora",)

    asyncio.run(run())


def test_frontends_share_one_volume_and_clear_only_their_folder():
    async def run():
        volume = FakeVolume()
        one, two = (
            ModalVolumeStore(volume, "ap-one"),
            ModalVolumeStore(volume, "ap-two"),
        )
        await one.put("qwen", {"n": 1})
        await two.put("qwen", {"n": 2})
        assert await one.get("qwen") == {"n": 1}
        assert await two.get("qwen") == {"n": 2}
        await one.clear()
        await one.clear()
        assert await one.list_keys("") == ()
        assert await two.list_items("") == (("qwen", {"n": 2}),)

    asyncio.run(run())


def test_put_many_writes_every_recipe_in_one_commit():
    async def run():
        volume = FakeVolume()
        store = ModalVolumeStore(volume, "ap-one")
        await store.put_many({"a": {"n": 1}, "b": {"n": 2}})
        await store.put_many({})
        assert volume.commits == 1
        assert dict(await store.list_items("")) == {"a": {"n": 1}, "b": {"n": 2}}

    asyncio.run(run())

import json

from scripts.multigpu_eval_utils import merge_json_shards, parse_device_list, shard_indices


def test_parse_device_list_uses_explicit_devices():
    assert parse_device_list("cuda:0", ["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]
    assert parse_device_list("cuda:0", None) == ["cuda:0"]


def test_shard_indices_partition_by_modulo():
    assert shard_indices(7, 2, 0) == [0, 2, 4, 6]
    assert shard_indices(7, 2, 1) == [1, 3, 5]


def test_merge_json_shards_restores_eval_index_order(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps([{"eval_index": 2}, {"eval_index": 0}]))
    b.write_text(json.dumps([{"eval_index": 3}, {"eval_index": 1}]))

    assert [row["eval_index"] for row in merge_json_shards([a, b])] == [0, 1, 2, 3]


def test_worker_logging_suppresses_non_main_stdout(capsys):
    from argparse import Namespace
    from scripts.multigpu_eval_utils import worker_logging

    with worker_logging(Namespace(shard_rank=1)):
        print("worker progress")
    with worker_logging(Namespace(shard_rank=0)):
        print("main progress")
    assert capsys.readouterr().out == "main progress\n"

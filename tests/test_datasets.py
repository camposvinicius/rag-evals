"""Tests for dataset loading and validation."""

import pytest

from ragevals.datasets import DatasetError, load_corpus, load_qa


@pytest.fixture()
def corpus_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "d1", "title": "Alpha", "text": "About alpha."}\n'
        '{"id": "d2", "title": "Beta", "text": "About beta."}\n'
    )
    return path


def test_load_corpus_ok(corpus_file):
    docs = load_corpus(corpus_file)
    assert [d.id for d in docs] == ["d1", "d2"]


def test_duplicate_doc_id_fails(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "d1", "title": "A", "text": "a"}\n{"id": "d1", "title": "B", "text": "b"}\n'
    )
    with pytest.raises(DatasetError, match="duplicate document id"):
        load_corpus(path)


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "d1", "title": "A", "text": "a"}\nnot json\n')
    with pytest.raises(DatasetError, match=":2"):
        load_corpus(path)


def test_qa_referencing_unknown_doc_fails(tmp_path, corpus_file):
    qa = tmp_path / "qa.jsonl"
    qa.write_text('{"id": "q1", "question": "?", "relevant_doc_ids": ["d999"]}\n')
    corpus_ids = {d.id for d in load_corpus(corpus_file)}
    with pytest.raises(DatasetError, match="unknown doc ids"):
        load_qa(qa, corpus_ids)


def test_qa_without_annotation_fails(tmp_path, corpus_file):
    qa = tmp_path / "qa.jsonl"
    qa.write_text('{"id": "q1", "question": "?", "relevant_doc_ids": []}\n')
    corpus_ids = {d.id for d in load_corpus(corpus_file)}
    with pytest.raises(DatasetError, match="no relevant_doc_ids"):
        load_qa(qa, corpus_ids)

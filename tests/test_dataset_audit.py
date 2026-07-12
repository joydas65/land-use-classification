from pathlib import Path

from PIL import Image

from terraclass.dataset_audit import difference_hash, find_near_duplicates


def test_difference_hash_is_stable_and_sensitive() -> None:
    dark = Image.new("RGB", (32, 32), "black")
    gradient = Image.new("L", (9, 8))
    gradient.putdata([(8 - column) * 20 for _row in range(8) for column in range(9)])
    assert difference_hash(dark) == difference_hash(dark.copy())
    assert difference_hash(dark) != difference_hash(gradient)


def test_near_duplicate_search_excludes_exact_content_hashes() -> None:
    entries = [
        ("a.tif", "same", 0b0000),
        ("b.tif", "same", 0b0000),
        ("c.tif", "different", 0b0001),
        ("d.tif", "far", 0b1111),
    ]
    count, examples = find_near_duplicates(entries, threshold=1)
    assert count == 2
    assert {(Path(item["left"]).name, Path(item["right"]).name) for item in examples} == {
        ("a.tif", "c.tif"),
        ("b.tif", "c.tif"),
    }

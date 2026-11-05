import json

from src.data_quality import ANN_NAME, check_coco, find_cross_split_duplicates


def edit(split_dir, fn):
    p = split_dir / ANN_NAME
    coco = json.loads(p.read_text())
    fn(coco)
    p.write_text(json.dumps(coco))


def test_clean_split_has_no_issues(split_dir):
    r = check_coco(split_dir, split_dir / ANN_NAME)
    assert r.n_issues == 0
    assert r.n_images == 3 and r.n_annotations == 3 and r.n_empty_images == 1
    assert r.class_counts == {"player": 2, "football": 1}


def test_invalid_unknown_oob_duplicate(split_dir):
    def add(coco):
        a = coco["annotations"]
        a.append({"id": 10, "image_id": 0, "category_id": 2, "bbox": [5, 5, 0, 10]})
        a.append({"id": 11, "image_id": 0, "category_id": 99, "bbox": [5, 5, 10, 10]})
        a.append({"id": 12, "image_id": 1, "category_id": 2, "bbox": [180, 0, 50, 50]})
        a.append({"id": 13, "image_id": 0, "category_id": 2, "bbox": [10, 20, 30, 40]})
    edit(split_dir, add)
    r = check_coco(split_dir, split_dir / ANN_NAME)
    for kind in ("invalid_bbox", "unknown_category_id",
                 "out_of_bounds_bbox", "duplicate_annotation"):
        assert kind in r.issues, kind


def test_missing_files_reported(split_dir, tmp_path):
    (split_dir / "train_img0.png").unlink()
    r = check_coco(split_dir, split_dir / ANN_NAME)
    assert "missing_image_file" in r.issues
    r2 = check_coco(split_dir, tmp_path / "absent.json")
    assert "missing_annotation_file" in r2.issues


def test_cross_split_duplicates(make_split):
    a, b = make_split("train"), make_split("valid")  # same seed => identical images
    groups = find_cross_split_duplicates({"train": a, "valid": b})
    assert len(groups) == 3

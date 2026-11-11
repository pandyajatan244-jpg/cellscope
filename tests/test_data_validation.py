from cellscope.config import CLASS_NAMES, NUM_CLASSES


def test_class_count():
    assert NUM_CLASSES == 8


def test_class_names_unique():
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))

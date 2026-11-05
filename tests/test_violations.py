"""Unit tests for the F2 violation rules, using hand-made boxes (no model needed)."""

from src.detect import Detection
from src.violations import check_people, count_verdicts, overlap_fraction, summary_text


def box(name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(class_id=0, class_name=name, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


# A person standing at x 100–200, y 0–300.
# A helmet's centre must sit in y 0–100 (top third); a vest's centre in y 45–210 (15%–70%).
PERSON = box("Person", 100, 0, 200, 300)
HELMET = box("helmet", 130, 5, 170, 40)
VEST = box("vest", 110, 110, 190, 190)

# The person is 300 px tall in a 1000 px image (30%), so they are judged.
IMAGE_HEIGHT = 1000


def test_compliant_when_helmet_and_vest_in_zones():
    [v] = check_people([PERSON, HELMET, VEST], IMAGE_HEIGHT)
    assert v.status == "compliant"
    assert not v.is_violation


def test_missing_helmet():
    [v] = check_people([PERSON, VEST], IMAGE_HEIGHT)
    assert v.status == "missing helmet"


def test_missing_vest():
    [v] = check_people([PERSON, HELMET], IMAGE_HEIGHT)
    assert v.status == "missing vest"


def test_missing_both_when_only_person_detected():
    [v] = check_people([PERSON], IMAGE_HEIGHT)
    assert v.status == "missing helmet and vest"
    assert v.missing == ["helmet", "vest"]


def test_helmet_in_wrong_zone_does_not_count():
    helmet_at_feet = box("helmet", 130, 250, 170, 290)
    [v] = check_people([PERSON, helmet_at_feet, VEST], IMAGE_HEIGHT)
    assert not v.has_helmet


def test_vest_at_head_height_does_not_count():
    vest_too_high = box("vest", 110, 0, 190, 30)  # centre at 5% of height
    [v] = check_people([PERSON, HELMET, vest_too_high], IMAGE_HEIGHT)
    assert not v.has_vest


def test_vest_high_on_body_still_counts():
    # Regression: in close-ups cut off at the waist, the vest sits in the upper part of the
    # person box. The first version of the rule (middle third only) wrongly rejected this.
    high_vest = box("vest", 110, 40, 190, 110)  # centre at 25% of height
    [v] = check_people([PERSON, HELMET, high_vest], IMAGE_HEIGHT)
    assert v.has_vest


def test_helmet_mostly_outside_person_does_not_count():
    # Only a sliver of this helmet (a neighbour's) overlaps the head zone.
    neighbours_helmet = box("helmet", 190, 10, 240, 50)
    [v] = check_people([PERSON, neighbours_helmet, VEST], IMAGE_HEIGHT)
    assert not v.has_helmet


def test_no_helmet_beats_less_confident_helmet():
    helmet = box("helmet", 130, 5, 170, 40, conf=0.5)
    no_helmet = box("no_helmet", 130, 5, 170, 40, conf=0.8)
    [v] = check_people([PERSON, helmet, no_helmet, VEST], IMAGE_HEIGHT)
    assert not v.has_helmet


def test_helmet_beats_less_confident_no_helmet():
    helmet = box("helmet", 130, 5, 170, 40, conf=0.8)
    no_helmet = box("no_helmet", 130, 5, 170, 40, conf=0.5)
    [v] = check_people([PERSON, helmet, no_helmet, VEST], IMAGE_HEIGHT)
    assert v.has_helmet


def test_each_person_judged_separately():
    other = box("Person", 300, 0, 400, 300)
    verdicts = check_people([PERSON, HELMET, VEST, other], IMAGE_HEIGHT)
    assert [v.status for v in verdicts] == ["compliant", "missing helmet and vest"]
    assert summary_text(verdicts) == "2 people: 1 compliant, 1 in violation, 0 too far to judge"


def test_no_people_gives_no_verdicts():
    assert check_people([HELMET, VEST], IMAGE_HEIGHT) == []


def test_overlap_fraction():
    zone = (0, 0, 100, 100)
    assert overlap_fraction(box("helmet", 10, 10, 50, 50), *zone) == 1.0
    assert overlap_fraction(box("helmet", 50, 0, 150, 100), *zone) == 0.5
    assert overlap_fraction(box("helmet", 200, 200, 250, 250), *zone) == 0.0


# --- Too far to judge: people under 15% of the image height ---

def test_small_person_is_too_far_to_judge():
    # 300 px tall in a 3000 px image = 10%.
    [v] = check_people([PERSON, HELMET, VEST], image_height=3000)
    assert v.status == "too far to judge"
    assert v.too_far


def test_too_far_is_unknown_not_missing():
    # Even with a confident no_helmet box, a too-far person is not a violation:
    # their PPE is unknown (None), which is different from missing (False).
    no_helmet = box("no_helmet", 130, 5, 170, 40, conf=0.95)
    [v] = check_people([PERSON, no_helmet], image_height=3000)
    assert v.has_helmet is None and v.has_vest is None
    assert v.missing == []
    assert not v.is_violation


def test_person_exactly_at_cut_off_is_judged():
    # 300 px in a 2000 px image = exactly 15%: the rule is "under 15%", so this person is judged.
    [v] = check_people([PERSON, HELMET, VEST], image_height=2000)
    assert v.status == "compliant"


def test_summary_counts_too_far_people():
    far = box("Person", 300, 0, 320, 100)  # 100 px = 10% of the 1000 px image
    verdicts = check_people([PERSON, VEST, far], IMAGE_HEIGHT)
    assert count_verdicts(verdicts) == {"people": 2, "compliant": 0, "violations": 1, "too_far": 1}
    assert summary_text(verdicts) == "2 people: 0 compliant, 1 in violation, 1 too far to judge"


def test_summary_singular():
    verdicts = check_people([PERSON, HELMET, VEST], IMAGE_HEIGHT)
    assert summary_text(verdicts) == "1 person: 1 compliant, 0 in violation, 0 too far to judge"

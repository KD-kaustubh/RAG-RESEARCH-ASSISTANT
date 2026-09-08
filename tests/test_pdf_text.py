from pdf_text import (
    build_vocabulary,
    continues_sentence,
    join_hyphenated,
    strip_page_number,
    take_leading_captions,
)


def test_page_number_at_the_foot_is_removed():
    assert strip_page_number("some body text\n2") == "some body text"


def test_page_number_at_the_head_is_removed():
    assert strip_page_number("7\nsome body text") == "some body text"


def test_numbers_inside_the_page_are_kept():
    text = "results table\n512\nmore text"

    assert strip_page_number(text) == text


def test_body_text_is_never_mistaken_for_a_page_number():
    text = "The encoder has N = 6 layers"

    assert strip_page_number(text) == text


def test_leading_caption_is_separated_from_the_body():
    captions, body = take_leading_captions("Figure 1: The Transformer - model architecture.\nwise fully connected")

    assert captions == ["Figure 1: The Transformer - model architecture."]
    assert body == "wise fully connected"


def test_table_caption_is_also_recognised():
    captions, _ = take_leading_captions("Table 2: BLEU scores.\nbody text")

    assert captions == ["Table 2: BLEU scores."]


def test_text_mentioning_a_figure_is_not_treated_as_a_caption():
    text = "shown in the left and right halves of Figure 1, respectively."

    captions, body = take_leading_captions(text)

    assert captions == []
    assert body == text


def test_line_break_hyphen_is_healed_when_the_document_uses_the_merged_word():
    vocabulary = build_vocabulary(["convolutional layers are used here"])

    assert join_hyphenated("convolu-\ntional layers", vocabulary) == "convolutional layers"


def test_real_compound_keeps_its_hyphen():
    vocabulary = build_vocabulary(["the position-wise feed-forward network"])

    assert join_hyphenated("position-\nwise fully connected", vocabulary) == "position-wise fully connected"


def test_unknown_hyphenated_break_keeps_the_hyphen():
    vocabulary = build_vocabulary(["unrelated words only"])

    assert join_hyphenated("Salakhutdi-\nnov", vocabulary) == "Salakhutdi-nov"


def test_continues_sentence_detects_an_unfinished_page():
    assert continues_sentence("the second is a simple, position-") is True


def test_continues_sentence_is_false_after_a_full_stop():
    assert continues_sentence("This sentence is complete.") is False

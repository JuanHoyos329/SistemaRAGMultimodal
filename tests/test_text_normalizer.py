from app.infrastructure.text_normalizer import normalize_extracted_text


def test_normalizes_spacing_acute_and_html_space_entity():
    extracted = "Son los puntos m\u00b4as juntos de lo que ser\u00b4ian&#x20;"

    assert normalize_extracted_text(extracted) == "Son los puntos m\u00e1s juntos de lo que ser\u00edan "

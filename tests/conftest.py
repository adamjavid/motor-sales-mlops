import pytest

from motor_sales.synthetic import generate


@pytest.fixture(scope="session")
def quotes():
    return generate(n_rows=3000, seed=0)

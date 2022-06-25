from common.lint.sqlfluff_wrapper import apply_sqlfluff

my_bad_query = "SeLEct  *, 1, blah as  fOO \n" "  from myTable"
my_good_query = """
SELECT
    blah AS foo,
    1 AS one
FROM myTable
"""


def test_sqlfluff_has_output_for_unformatted_sql() -> None:
    bad_reformat = apply_sqlfluff(my_bad_query, "postgres")
    assert bad_reformat != "", "an output should exist"


def test_sqlfluff_is_running_and_has_no_output_on_correct_sqls() -> None:
    good_reformat = apply_sqlfluff(my_good_query, "postgres", do_print=True)
    assert good_reformat == "", "no output"

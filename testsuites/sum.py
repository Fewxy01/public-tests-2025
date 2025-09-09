import testsuites.suite as suite

from typing import Iterable, Tuple, List, Dict, Optional, Union

SUITE_NAME = "sum"

class __Comparator(suite.Comparator):
	def __init__(self):
		super().__init__()

	def test(self, user_process: suite.UserProcess, test: suite.Test) -> suite.Result:
		# Sum's input.
		assert isinstance(test.input, list), "Input is not list"
		l = list(test.input)
		a, b = int(l[0]), int(l[1])

		# Sum's expected.
		assert isinstance(test.expected, int), "Expected is not integer"
		expected = int(test.expected)

		try:
			# Sum's actual value.
			actual = int(user_process.stdout)

			# Assert.
			if actual != expected:
				return suite.Result(suite.Errno.ERROR_ASSERTION, f"wrong sum, {a} + {b} = {expected}, not {actual}")

			return suite.err_ok()
		except Exception as _:
			return suite.Result(suite.Errno.ERROR_TYPE_ERROR, f"excepted integer, but found '{suite.escape(user_process.stdout)}'")


def __test_naming(a: int, b: int) -> str:
	return f"{a} + {b}"

def __generate_tests() -> Iterable[Tuple[str, List[int], int]]:
	generated: Iterable[Tuple[str, List[int], int]]= []

	for a in range(1, 10):
		for b in range(10, 20):
			name = __test_naming(a, b)
			input = [a, b]
			expected = a + b
			test_data = (name, input, expected)
			generated.append(test_data)

	return generated

def get_instance() -> Tuple[suite.Tester, Optional[Dict[str, float]]]:
	__COEFF_TO_ENVNAME = { "a + b": "A_PLUS_B" }

	cmp = __Comparator()
	tester = suite.Tester(
		comparator = cmp,
		is_stdin_input = True,
		is_raw_input = True,
		is_raw_output = True
	)
	coefficients = suite.get_coefficients(SUITE_NAME, __COEFF_TO_ENVNAME)

	tests = __generate_tests()
	for test_data in tests:
		test_name, test_input, test_expected = test_data
		tester.add_success(test_name, test_input, test_expected, output_stream = None, categories = ["a + b"])

	return tester, coefficients

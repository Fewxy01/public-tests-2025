import testsuites.suite as suite
import os
import numpy as np

from typing import Any, Iterable, Tuple, List, Dict, Optional, Union

SUITE_NAME = "math"

def __print_mtx_to_file(file, mtx):
	rc = mtx.shape
	file.write(f"{rc[0]} {rc[1]}\n")
	for i in range(0, rc[0]):
		for k in range(0, rc[1]):
			file.write("{:g} ".format(mtx[i][k]))
		file.write("\n")

def __generate_rnd_mtx(rng, sizes, multiply):
	return rng.random(size=sizes, dtype=np.float32) * multiply

class __Comparator(suite.Comparator):
	def __init__(self):
		super().__init__()
		self.DELTA = 1e-4

	def test(self, user_process: suite.UserProcess, test: suite.Test) -> suite.Result:
		def __generate_mtx_from_lists(nested_list: List[List[str]]) -> Union[np.ndarray, suite.Result]:
			matrix = np.zeros(shape=(len(nested_list), len(nested_list[0])), dtype=np.float32)
			for y in range(matrix.shape[0]):
				for x in range(matrix.shape[1]):
					try:
						matrix[y][x] = np.float32(nested_list[y][x])
					except Exception:
						if (nested_list[y][x] == "-nan(ind)"):
							matrix[y][x] = np.float32(np.nan)
						else:
							return suite.Result(suite.Errno.ERROR_ASSERTION, f"expected float value at position ({y}, {x}), but found '{suite.escape(nested_list[y][x])}'")
			return matrix

		# Extract output filename.
		assert isinstance(test.input, list), "Input is not list"
		input_list = list(test.input)
		output_filename = str(input_list[1])

		# Extract reference filename.
		assert isinstance(test.expected, str), "Expected is not path to reference filename"
		reference_filename = str(test.expected)

		# Check output file existing.
		if not os.path.exists(output_filename):
			return suite.err_file_not_found(output_filename)

		# Read files.
		actual = [s.strip() for s in open(output_filename, "r").readlines()]
		expected = [s.strip() for s in open(reference_filename, "r").readlines()]

		# Check number of lines.
		if len(actual) != len(expected):
			return suite.Result(suite.Errno.ERROR_ASSERTION, f"wrong number of lines: {len(actual)} (in '{output_filename}') vs. {len(expected)} (in '{reference_filename}')")

		# Check on 'no solution'.
		if expected[0] == "no solution":
			if actual[0] != "no solution":
				return suite.Result(suite.Errno.ERROR_ASSERTION, f"expected 'no solution' as answer, but found something else")
			return suite.err_ok()

		# Check on 'det' operation.
		if (len(expected) == 1):
			det_ref = np.abs(np.float32(expected[0]))
			try:
				det_out = np.abs(np.float32(actual[0]))
				if np.isnan(det_ref) and not np.isnan(det_out) or not np.isnan(det_ref) and np.isnan(det_out) or np.abs(det_ref - det_out) / np.fmax(det_ref, det_out) > self.DELTA:
					return suite.Result(suite.Errno.ERROR_ASSERTION, f"wrong det-operation answer, found {det_out}, expected {det_ref} with relative delta {self.DELTA}")
				return suite.err_ok()
			except Exception:
				return suite.Result(suite.Errno.ERROR_ASSERTION, f"expected float value as det-operation answer, but found '{suite.escape(actual[0])}'")

		# Expect sizes.
		sizes_ref = list(map(int, expected[0].split(" ")))
		sizes_out: List[int] = []
		try:
			sizes_out = list(map(int, actual[0].split(" ")))
		except Exception:
			return suite.Result(suite.Errno.ERROR_ASSERTION, f"expected matrix sizes, but found '{suite.escape(actual[0])}'")

		if sizes_ref[0] != sizes_out[0] or sizes_ref[1] != sizes_out[1]:
			return suite.Result(suite.Errno.ERROR_ASSERTION, f"wrong matrix sizes, expected {sizes_ref[0]}x{sizes_ref[1]}, but found {sizes_out[0]}x{sizes_out[1]}")

		# Comparison matrices.
		nested_actual_matrix = [s.strip().split(' ') for s in actual[1:]]
		nested_expected_matrix = [s.strip().split(' ') for s in expected[1:]]
		
		if len(nested_actual_matrix) != len(nested_expected_matrix) or len(nested_actual_matrix[0]) != len(nested_expected_matrix[0]):
			return suite.Result(suite.Errno.ERROR_ASSERTION, f"Matrix size mismatch: expected {expected_matrix.shape}, but got {actual_matrix.shape}")

		generate_mtx_result = __generate_mtx_from_lists(nested_actual_matrix)
		if isinstance(generate_mtx_result, suite.Result):
			return generate_mtx_result
		actual_matrix = generate_mtx_result

		generate_mtx_result = __generate_mtx_from_lists(nested_expected_matrix)
		if isinstance(generate_mtx_result, suite.Result):
			return generate_mtx_result
		expected_matrix = generate_mtx_result

		max_abs = np.fmax(np.abs(expected_matrix), np.abs(actual_matrix))
		max_by_row = np.amax(max_abs, axis = 1)
		max_by_col = np.amax(max_abs, axis = 0)

		for y in range(expected_matrix.shape[0]):
			for x in range(expected_matrix.shape[1]):
				a = actual_matrix[y][x]
				e = expected_matrix[y][x]

				if np.isnan(a) and np.isnan(e):
					continue
				if np.isnan(a) and not np.isnan(e) or not np.isnan(a) and np.isnan(e):
					return suite.err_assertion_pos(y, x, str(a), str(e))
				
				diff = abs(a - e)
				m = min(max_by_row[y], max_by_col[x])
				d = diff / m
				if d > self.DELTA:
					return suite.err_assertion_pos(y, x, str(a), str(e))

		return suite.err_ok()

__BASE_DIR = os.path.join(suite.TESTDATA_DIR, SUITE_NAME)
__INPUT_DIR = os.path.join(__BASE_DIR, "input")
__OUTPUT_DIR = os.path.join(__BASE_DIR, "output")
__REFERENCE_DIR = os.path.join(__BASE_DIR, "reference")

def __generate_test(test_data: dict) -> Union[np.ndarray, np.float32, str]:
	op = test_data["op"]
	amtx = test_data["amtx"]
	
	if (op in {"+", "-", "*"}):
		bmtx = test_data["bmtx"]
	elif (op == "^"):
		p = test_data["p"]

	try:
		if (op == "+"):
			if (amtx.shape != bmtx.shape):
				raise np.linalg.LinAlgError
			return amtx + bmtx
		elif (op == "-"):
			if (amtx.shape != bmtx.shape):
				raise np.linalg.LinAlgError
			return amtx - bmtx
		elif (op == "*"):
			return np.linalg.matmul(amtx, bmtx)
		elif (op == "^"):
			return np.linalg.matrix_power(amtx, p)
		else:
			return np.float32(np.linalg.det(amtx))
	except:
		return "no solution"

def __get_tests(category_name: str) -> Iterable[Tuple[str, str, str, str, float]]:
	tests: Iterable[Tuple[str, str, str, str, float]] = []

	f = lambda x : str(x).startswith(category_name)
	ts = list(filter(f, os.listdir(__INPUT_DIR)))

	for i, t in enumerate(ts):
		test_input_filename = os.path.join(__INPUT_DIR, t)
		test_name = f"{category_name.capitalize()} #{i + 1} (see: {os.path.relpath(test_input_filename)})"
		test_output_filename = os.path.join(__OUTPUT_DIR, t)
		test_reference_filename = os.path.join(__REFERENCE_DIR, t)
		test_data = (test_name, test_input_filename, test_output_filename, test_reference_filename, 1.0)
		tests.append(test_data)

	return tests

def __get_big_tests(category_name: str, rng: np.random.Generator, test_params: Dict[str, Any]) -> List[Tuple[str, str, str, str, float]]:
	tests: List[Tuple[str, str, str, str, float]] = []

	SUFFIX = "big"
	POW_CONSTANT = 8
	TIMEOUT_CONSTANT = 4.0 if (test_params["op"] == "*" or test_params["op"] == "|" or test_params["op"] == "^") else 1.0

	for i, param in enumerate(test_params["params"]):
		__make_filename = f"{SUFFIX}_{category_name}_{param['suffix']}.txt"
		make_test_input_filename = os.path.join(__INPUT_DIR, __make_filename)

		test_name = f"Big {category_name} #{i + 1} (see: {os.path.relpath(make_test_input_filename)})"
		test_input_filename = make_test_input_filename
		test_output_filename = os.path.join(__OUTPUT_DIR, __make_filename)
		test_reference_filename = os.path.join(__REFERENCE_DIR, __make_filename)

		size = param['size']
		amtx = (np.zeros(shape=size, dtype=np.float32) if (param['type'] == "zeros") else np.eye(N=size[0], dtype=np.float32) if(param['type'] == "eye") else __generate_rnd_mtx(rng, sizes=size, multiply=1.0))
		bmtx = (np.zeros(shape=size, dtype=np.float32) if (param['type'] == "zeros") else np.eye(N=size[0], dtype=np.float32) if(param['type'] == "eye") else __generate_rnd_mtx(rng, sizes=size, multiply=1.0))
		op = test_params["op"]


		stream_input = open(test_input_filename, "w")
		stream_input.write(f"{op}\n")
		__print_mtx_to_file(stream_input, amtx)
		if category_name == "pow":
			stream_input.write(f"{POW_CONSTANT}\n")
		elif category_name != "det":
			__print_mtx_to_file(stream_input, bmtx)
		stream_input.close()

		result = None
		if category_name == "det":
			result = __generate_test({ "op": op, "amtx": amtx })
		elif category_name == "pow":
			result = __generate_test({ "op": op, "amtx": amtx, "p": POW_CONSTANT })
		else:
			result = __generate_test({ "op": op, "amtx": amtx, "bmtx": bmtx })

		stream_reference = open(test_reference_filename, "w")
		if isinstance(result, str) and result == "no solution":
			stream_reference.write("no solution\n")
		elif isinstance(result, np.float32):
			stream_reference.write(f"{result:g}\n")
		else:
			__print_mtx_to_file(stream_reference, result)

		stream_reference.close()

		test_data = (test_name, test_input_filename, test_output_filename, test_reference_filename, TIMEOUT_CONSTANT)
		tests.append(test_data)

	return tests

def __get_negative_tests(category_name: str) -> List[Tuple[str, List[str], int, List[str]]]:
	tests: List[Tuple[str, List[str], int, List[str]]] = []

	test_input_filename = os.path.join(__INPUT_DIR, f"neg_{0}.txt")
	test_data = (
		f"NEG #{0} empty input file (see: {os.path.relpath(test_input_filename)})",
		[test_input_filename, "non_existing.out"],
		1, [category_name]
	)
	tests.append(test_data)

	test_input_filename = os.path.join(__INPUT_DIR, f"neg_{1}.txt")
	test_data = (
		f"NEG #{1} incorrect operator file (see: {os.path.relpath(test_input_filename)})",
		[test_input_filename, "non_existing.out"],
		1, [category_name]
	)
	tests.append(test_data)

	test_input_filename = os.path.join(__INPUT_DIR, f"neg_{2}.txt")
	test_data = (
		f"NEG #{2} non-existing input file (see: {os.path.relpath(test_input_filename)})",
		[test_input_filename, "non_existing.out"],
		1, [category_name]
	)
	tests.append(test_data)

	test_input_filename = os.path.join(__INPUT_DIR, f"neg_{3}.txt")
	test_output_filename = os.path.join(__OUTPUT_DIR, os.path.join("non_existing", f"neg_{3}.txt"))
	test_data = (
		f"NEG #{3} non-existing output file (see: {os.path.relpath(test_input_filename)})",
		[test_input_filename, test_output_filename],
		1, [category_name]
	)
	tests.append(test_data)

	for i in range(2):
		test_input_filename = os.path.join(__INPUT_DIR, f"oom_{i + 1}.txt")
		test_output_filename = os.path.join(__OUTPUT_DIR, f"should_not_be_existing_{i + 1}.txt")
		test_data = (
			f"NEG #{3 + 1 + i} out-of-memory, trying to allocate 512 Gb. (see: {os.path.relpath(test_input_filename)})",
			[test_input_filename, test_output_filename],
			1, [category_name]
		)
		tests.append(test_data)
	return tests


def get_instance() -> Tuple[suite.Tester, Optional[Dict[str, float]]]:
	__COEFF_TO_ENVNAME = { "det": "DET", "mul": "MUL", "pow": "POW", "sub": "SUB", "sum": "SUM", "neg": "NEG" }

	cmp = __Comparator()
	tester = suite.Tester(
		comparator = cmp,
		is_stdin_input = False,
		is_raw_input = True,
		is_raw_output = True
	)
	coefficients = suite.get_coefficients(SUITE_NAME, __COEFF_TO_ENVNAME)

	suite.ensure_existence_directory(__OUTPUT_DIR)
	
	rng = np.random.default_rng(seed=225526)
	big_test_data = {
		"det": {"op": "|", "params": [ {"type": "rnd", "suffix": "1_9", "size": (1000, 1000)}, {"type": "zeros", "suffix": "8", "size": (1000, 1000)}, {"type": "eye", "suffix": "9", "size": (1000, 1000)}]},
		"mul": {"op": "*", "params": [ {"type": "rnd", "suffix": "6", "size": (1000, 1000)} ]},
		"pow": {"op": "^", "params": [ {"type": "rnd", "suffix": "16", "size": (500, 500)} ]},
		"sub": {"op": "-", "params": [ {"type": "rnd", "suffix": "5", "size": (1000, 1000)} ]},
		"sum": {"op": "+", "params": [ {"type": "rnd", "suffix": "6", "size": (1000, 1000)} ]}
	}

	np.seterr(all='ignore')

	for category_name in __COEFF_TO_ENVNAME.keys():
		if category_name not in big_test_data:
			continue
		tests = [__get_tests(category_name), __get_big_tests(category_name, rng, big_test_data[category_name])]
		for test_datas in tests:
			for test_data in test_datas:
				test_name, test_input_filename, test_output_filename, test_reference_filename, test_timeout = test_data
				tester.add_success(
					name = test_name,
					input = [test_input_filename, test_output_filename],
					expected = test_reference_filename,
					timeout = test_timeout,
					categories = [category_name]
				)

	tests = __get_negative_tests("neg")
	for test_data in tests:
		test_name, test_inputs, exitcode, categories = test_data
		tester.add_failed(
			name = test_name,
			input = test_inputs,
			exitcode = exitcode,
			categories = categories
		)

	return tester, coefficients

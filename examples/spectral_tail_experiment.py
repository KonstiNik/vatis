"""Spectral tail overlap experiment: do two stylistically different
implementations of the same algorithm produce a chi_pos bump?

Hypothesis
----------
Two Python texts that implement the same computation (matrix
multiplication) in maximally different surface styles should have
orthogonal gradients early in training (different token statistics)
but develop shared structure later, when the model resolves deeper
code semantics. This would show up as a **transient bump in
chi_pos(style_a, style_b)** at the checkpoint where training
"arrives at" the spectral band where the two styles overlap.

Contrast with the canonical prose-vs-code example, where chi_pos
starts high (shared surface statistics of natural language) and
declines monotonically as the model specializes.

Style A: verbose, imperative, triple-nested loops, explicit indexing,
         long variable names, comments, type hints, assertions.
Style B: terse, functional, NumPy/list-comprehension idioms,
         one-liners, operator overloading, minimal naming.

Both implement matrix multiplication multiple times in their
respective style to fill a full B=8, S=128 batch without cyclic
repetition.

Run with::

    HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/spectral_tail_experiment.py
    HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/spectral_tail_experiment.py --model EleutherAI/pythia-70m

Run both models and then produce a cross-scale comparison::

    HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/spectral_tail_experiment.py --compare

Outputs are written to ``examples/spectral_tail/``.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/data/knikolaou/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/knikolaou/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/knikolaou/huggingface")

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from _helpers import tokenize_into_lm_batch
from vatis import analyze

# ---------------------------------------------------------------- config

DEFAULT_MODEL = "EleutherAI/pythia-14m"

MODELS_FOR_COMPARISON = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
]

REVISIONS: list[str] = [
    "step1",
    "step8",
    "step64",
    "step512",
    "step1000",
    "step2000",
    "step4000",
    "step8000",
    "step16000",
    "step32000",
    "step64000",
    "step128000",
    "step143000",
]

BATCH_SIZE = 1
SEQ_LEN = 1024
N_HUTCHINSON = 32
SEED = 0

STYLE_A_NAME = "python"
STYLE_B_NAME = "cpp"
CROSS_PAIRS: list[tuple[str, str]] = [(STYLE_A_NAME, STYLE_B_NAME)]

OUT_DIR = Path(__file__).parent / "spectral_tail"


def model_tag(model_name: str) -> str:
    """Extract a short tag from a HF model name, e.g. 'pythia-14m'."""
    return model_name.split("/")[-1]


def parquet_path(model_name: str) -> Path:
    return OUT_DIR / f"results_{model_tag(model_name)}.parquet"



# ---------------------------------------------------------------- texts
#
# Both texts implement the same operations in the same order:
#   1. Naive triple-loop matrix multiply
#   2. Accumulator-reordered matrix multiply (ikj loop order)
#   3. Transpose-and-dot matrix multiply
#   4. Matrix transpose
#   5. Vector dot product
#   6. Matrix trace
#   7. Squared Frobenius norm
#   8. Verification / self-test
#
# The Python text uses verbose names, docstrings, type hints, assertions.
# The C++ text uses raw pointers, manual memory, C-style indexing.
# Token-level overlap between the two is near zero.

STYLE_A_TEXT = '''
def matrix_multiply_naive(
    matrix_a: list[list[float]],
    matrix_b: list[list[float]],
) -> list[list[float]]:
    """Multiply two matrices using three nested for-loops.

    Parameters
    ----------
    matrix_a : list[list[float]]
        The left-hand operand, with shape (num_rows_a, num_cols_a).
    matrix_b : list[list[float]]
        The right-hand operand, with shape (num_rows_b, num_cols_b).
        Requires num_cols_a == num_rows_b.

    Returns
    -------
    list[list[float]]
        The product matrix with shape (num_rows_a, num_cols_b).
    """
    num_rows_a = len(matrix_a)
    num_cols_a = len(matrix_a[0])
    num_rows_b = len(matrix_b)
    num_cols_b = len(matrix_b[0])

    assert num_cols_a == num_rows_b, (
        f"Incompatible dimensions for multiplication: "
        f"({num_rows_a}, {num_cols_a}) x ({num_rows_b}, {num_cols_b})"
    )

    result_matrix: list[list[float]] = []
    for row_index in range(num_rows_a):
        current_row: list[float] = []
        for col_index in range(num_cols_b):
            accumulated_sum: float = 0.0
            for inner_index in range(num_cols_a):
                element_from_a = matrix_a[row_index][inner_index]
                element_from_b = matrix_b[inner_index][col_index]
                accumulated_sum += element_from_a * element_from_b
            current_row.append(accumulated_sum)
        result_matrix.append(current_row)

    return result_matrix


def matrix_multiply_with_accumulator(
    left_operand: list[list[float]],
    right_operand: list[list[float]],
) -> list[list[float]]:
    """Matrix multiplication using ikj loop order with pre-allocated output.

    Reorders the inner loops so the innermost loop streams across the
    columns of B, which is cache-friendly for row-major storage.

    Parameters
    ----------
    left_operand : list[list[float]]
        The left matrix with shape (m, k).
    right_operand : list[list[float]]
        The right matrix with shape (k, n).

    Returns
    -------
    list[list[float]]
        The product matrix with shape (m, n).
    """
    number_of_output_rows = len(left_operand)
    shared_dimension = len(left_operand[0])
    number_of_output_cols = len(right_operand[0])

    assert shared_dimension == len(right_operand), (
        f"Inner dimensions do not match: {shared_dimension} vs {len(right_operand)}"
    )

    output_matrix: list[list[float]] = [
        [0.0 for _col in range(number_of_output_cols)]
        for _row in range(number_of_output_rows)
    ]

    for row_idx in range(number_of_output_rows):
        for inner_idx in range(shared_dimension):
            scaling_factor = left_operand[row_idx][inner_idx]
            for col_idx in range(number_of_output_cols):
                output_matrix[row_idx][col_idx] += (
                    scaling_factor * right_operand[inner_idx][col_idx]
                )

    return output_matrix


def compute_matrix_transpose(
    input_matrix: list[list[float]],
) -> list[list[float]]:
    """Compute the transpose of a matrix by swapping rows and columns.

    Parameters
    ----------
    input_matrix : list[list[float]]
        The matrix to transpose, with shape (num_rows, num_cols).

    Returns
    -------
    list[list[float]]
        The transposed matrix with shape (num_cols, num_rows).
    """
    num_rows = len(input_matrix)
    num_cols = len(input_matrix[0])

    transposed_matrix: list[list[float]] = []
    for col_index in range(num_cols):
        new_row: list[float] = []
        for row_index in range(num_rows):
            new_row.append(input_matrix[row_index][col_index])
        transposed_matrix.append(new_row)

    return transposed_matrix


def compute_dot_product_of_vectors(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Compute the dot product of two vectors element by element.

    Parameters
    ----------
    vector_a : list[float]
        The first vector.
    vector_b : list[float]
        The second vector, must have the same length as vector_a.

    Returns
    -------
    float
        The scalar dot product.
    """
    assert len(vector_a) == len(vector_b), (
        f"Vector lengths do not match: {len(vector_a)} vs {len(vector_b)}"
    )

    dot_product_result: float = 0.0
    for element_index in range(len(vector_a)):
        dot_product_result += vector_a[element_index] * vector_b[element_index]

    return dot_product_result


def matrix_multiply_via_transpose_and_dot(
    first_matrix: list[list[float]],
    second_matrix: list[list[float]],
) -> list[list[float]]:
    """Matrix multiplication by transposing B and taking row dot products.

    C[i][j] = dot(A[i], B^T[j]), which is mathematically equivalent to
    the standard triple-loop but expressed in terms of the transpose
    and dot product primitives defined above.

    Parameters
    ----------
    first_matrix : list[list[float]]
        The left matrix with shape (m, k).
    second_matrix : list[list[float]]
        The right matrix with shape (k, n).

    Returns
    -------
    list[list[float]]
        The product matrix with shape (m, n).
    """
    transposed_second = compute_matrix_transpose(second_matrix)

    num_output_rows = len(first_matrix)
    num_output_cols = len(transposed_second)

    product_matrix: list[list[float]] = []
    for row_index in range(num_output_rows):
        output_row: list[float] = []
        for col_index in range(num_output_cols):
            dot_value = compute_dot_product_of_vectors(
                first_matrix[row_index],
                transposed_second[col_index],
            )
            output_row.append(dot_value)
        product_matrix.append(output_row)

    return product_matrix


def compute_matrix_trace(
    square_matrix: list[list[float]],
) -> float:
    """Compute the trace of a square matrix (sum of diagonal elements).

    Parameters
    ----------
    square_matrix : list[list[float]]
        A square matrix with shape (n, n).

    Returns
    -------
    float
        The trace value.
    """
    num_rows = len(square_matrix)
    num_cols = len(square_matrix[0])
    assert num_rows == num_cols, (
        f"Matrix is not square: ({num_rows}, {num_cols})"
    )

    trace_value: float = 0.0
    for diagonal_index in range(num_rows):
        trace_value += square_matrix[diagonal_index][diagonal_index]

    return trace_value


def compute_frobenius_norm_squared(
    input_matrix: list[list[float]],
) -> float:
    """Compute the squared Frobenius norm of a matrix.

    The squared Frobenius norm is the sum of the squares of all
    elements in the matrix, which equals the trace of M^T M.

    Parameters
    ----------
    input_matrix : list[list[float]]
        The input matrix of arbitrary shape.

    Returns
    -------
    float
        The squared Frobenius norm.
    """
    frobenius_norm_squared: float = 0.0
    for row_index in range(len(input_matrix)):
        for col_index in range(len(input_matrix[row_index])):
            element_value = input_matrix[row_index][col_index]
            frobenius_norm_squared += element_value * element_value

    return frobenius_norm_squared


def verify_matrix_operations() -> None:
    """Run a self-test to verify all implementations agree."""
    test_matrix_a: list[list[float]] = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    test_matrix_b: list[list[float]] = [
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]

    result_naive = matrix_multiply_naive(test_matrix_a, test_matrix_b)
    result_accum = matrix_multiply_with_accumulator(test_matrix_a, test_matrix_b)
    result_trans = matrix_multiply_via_transpose_and_dot(test_matrix_a, test_matrix_b)

    expected_result: list[list[float]] = [
        [58.0, 64.0],
        [139.0, 154.0],
    ]

    for row_index in range(len(expected_result)):
        for col_index in range(len(expected_result[0])):
            assert abs(result_naive[row_index][col_index] - expected_result[row_index][col_index]) < 1e-9
            assert abs(result_accum[row_index][col_index] - expected_result[row_index][col_index]) < 1e-9
            assert abs(result_trans[row_index][col_index] - expected_result[row_index][col_index]) < 1e-9

    trace_of_product = compute_matrix_trace(
        matrix_multiply_naive(
            compute_matrix_transpose(test_matrix_a), test_matrix_a
        )
    )
    frobenius_squared = compute_frobenius_norm_squared(test_matrix_a)
    assert abs(trace_of_product - frobenius_squared) < 1e-9

    dot_result = compute_dot_product_of_vectors(
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    )
    assert abs(dot_result - 32.0) < 1e-9

    identity: list[list[float]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert abs(compute_matrix_trace(identity) - 3.0) < 1e-9
    assert abs(compute_frobenius_norm_squared(identity) - 3.0) < 1e-9


if __name__ == "__main__":
    verify_matrix_operations()
    print("All matrix operation tests passed.")
'''

STYLE_B_TEXT = '''
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>

void matrix_multiply_naive(const double* A, const double* B, double* C,
                           int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols_a; ++k) {
                sum += A[i * cols_a + k] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
}

void matrix_multiply_accumulator(const double* A, const double* B, double* C,
                                  int m, int k, int n) {
    memset(C, 0, m * n * sizeof(double));
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            double scale = A[i * k + p];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += scale * B[p * n + j];
            }
        }
    }
}

void matrix_transpose(const double* M, double* T, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T[j * rows + i] = M[i * cols + j];
        }
    }
}

double dot_product(const double* u, const double* v, int n) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += u[i] * v[i];
    }
    return result;
}

void matrix_multiply_via_transpose(const double* A, const double* B, double* C,
                                    int m, int k, int n) {
    double* BT = new double[n * k];
    matrix_transpose(B, BT, k, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = dot_product(&A[i * k], &BT[j * k], k);
        }
    }
    delete[] BT;
}

double matrix_trace(const double* M, int n) {
    double trace = 0.0;
    for (int i = 0; i < n; ++i) {
        trace += M[i * n + i];
    }
    return trace;
}

double frobenius_norm_squared(const double* M, int rows, int cols) {
    double norm_sq = 0.0;
    for (int i = 0; i < rows * cols; ++i) {
        norm_sq += M[i] * M[i];
    }
    return norm_sq;
}

void verify_matrix_operations() {
    double A[6] = {1, 2, 3, 4, 5, 6};
    double B[6] = {7, 8, 9, 10, 11, 12};
    double C[4], C2[4], C3[4];

    matrix_multiply_naive(A, B, C, 2, 3, 2);
    assert(fabs(C[0] - 58.0) < 1e-9 && fabs(C[1] - 64.0) < 1e-9);
    assert(fabs(C[2] - 139.0) < 1e-9 && fabs(C[3] - 154.0) < 1e-9);

    matrix_multiply_accumulator(A, B, C2, 2, 3, 2);
    assert(fabs(C2[0] - 58.0) < 1e-9 && fabs(C2[1] - 64.0) < 1e-9);
    assert(fabs(C2[2] - 139.0) < 1e-9 && fabs(C2[3] - 154.0) < 1e-9);

    matrix_multiply_via_transpose(A, B, C3, 2, 3, 2);
    assert(fabs(C3[0] - 58.0) < 1e-9 && fabs(C3[1] - 64.0) < 1e-9);
    assert(fabs(C3[2] - 139.0) < 1e-9 && fabs(C3[3] - 154.0) < 1e-9);

    double AT[6];
    matrix_transpose(A, AT, 2, 3);
    double ATA[9];
    matrix_multiply_naive(AT, A, ATA, 3, 2, 3);
    assert(fabs(matrix_trace(ATA, 3) - frobenius_norm_squared(A, 2, 3)) < 1e-9);

    double d = dot_product(A, B, 3);
    assert(fabs(d - 58.0) < 1e-9);

    double I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    assert(fabs(matrix_trace(I, 3) - 3.0) < 1e-9);
    assert(fabs(frobenius_norm_squared(I, 3, 3) - 3.0) < 1e-9);
}

int main() {
    verify_matrix_operations();
    printf("All matrix operation tests passed.\\n");
    return 0;
}
'''

# ---------------------------------------------------------------- main


def pick_chi_net_method(model_name: str) -> str:
    """Choose chi_net method. With B=1 the per-seq-CV grad cache is just
    one gradient vector (~n_params * 4 bytes), so per_sequence_cv fits
    for all models up to ~1B on a 24 GB GPU."""
    return "per_sequence_cv"


def run_single_model(model_name: str) -> None:
    """Run the spectral tail experiment for one model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pq_path = parquet_path(model_name)
    chi_net_method = pick_chi_net_method(model_name)

    print(f"\nspectral tail experiment -- model={model_name}")
    print(f"  device={device}, chi_net_method={chi_net_method}")
    print(f"  revisions={REVISIONS}")
    print(f"  B={BATCH_SIZE}, S={SEQ_LEN}, n_hutchinson={N_HUTCHINSON}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if pq_path.exists():
        pq_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_batches = {
        STYLE_A_NAME: tokenize_into_lm_batch(
            STYLE_A_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
        STYLE_B_NAME: tokenize_into_lm_batch(
            STYLE_B_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
    }
    for name, batch in eval_batches.items():
        n_tokens = int(batch["input_ids"].numel())
        n_valid = int((batch["labels"] != -100).sum().item())
        print(f"  {name}: {n_tokens} tokens, {n_valid} valid ({n_valid / n_tokens:.2%})")

    t_total = time.perf_counter()
    analyze(
        model=model_name,
        revisions=REVISIONS,
        eval_batches=eval_batches,
        cross_pairs=CROSS_PAIRS,
        chi_net_method=chi_net_method,
        n_hutchinson=N_HUTCHINSON,
        micro_batch_size=BATCH_SIZE,
        seed=SEED,
        device=device,
        dtype="fp32",
        sink=str(pq_path),
    )
    total_s = time.perf_counter() - t_total

    n_calls = len(REVISIONS) * (len([STYLE_A_NAME, STYLE_B_NAME]) + len(CROSS_PAIRS))
    print(
        f"\nDone in {total_s:.1f}s. "
        f"{len(REVISIONS)} checkpoints x "
        f"(2 self pairs + {len(CROSS_PAIRS)} cross pair) = "
        f"{n_calls} measurements. Wrote {pq_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral tail overlap experiment")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Run all models ({', '.join(MODELS_FOR_COMPARISON)}) sequentially.",
    )
    args = parser.parse_args()

    if args.all:
        for model_name in MODELS_FOR_COMPARISON:
            pq_path = parquet_path(model_name)
            if pq_path.exists():
                print(f"results for {model_name} already exist at {pq_path}, skipping compute")
            else:
                run_single_model(model_name)
    else:
        run_single_model(args.model)


if __name__ == "__main__":
    main()

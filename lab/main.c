#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

float **matrix_memory_allocation(FILE *f, int line, int column)
{
  float **A = (float **)(malloc(line * sizeof(float *)));
  for (size_t i = 0; i < line; i++)
  {
    A[i] = (float *)(malloc(column * sizeof(float)));
  }
  return A;
}

void read_matrix(FILE *f, float **A, int line, int column)
{
  for (size_t i = 0; i < line; i++)
  {
    for (size_t j = 0; j < column; j++)
    {
      if (fscanf(f, "%f", &A[i][j]) != 1)
      {
        printf("Error reading matrix\n");
      }
    }
  }
}

void free_matrix(float **A, int rows)
{
  for (size_t i = 0; i < rows; i++)
  {
    free(A[i]);
  }
  free(A);
}

void sum_matrices(float **A, float **B, float **C, int line, int column)
{
  for (size_t i = 0; i < line; i++)
  {
    for (size_t j = 0; j < column; j++)
    {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

void minus_matrices(float **A, float **B, float **C, int line, int column)
{
  for (size_t i = 0; i < line; i++)
  {
    for (size_t j = 0; j < column; j++)
    {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
}

void multiply_matrices(float **A, float **B, float **C, int row_1, int column_1,
                       int row_2, int column_2)
{
  for (size_t k = 0; k < row_1; k++)
  {
    for (size_t m = 0; m < column_2; m++)
    {
      C[k][m] = 0;
      for (size_t n = 0; n < column_1; n++)
      {
        C[k][m] += A[k][n] * B[n][m];
      }
    }
  }
}

void matrix_to_a_power(float **A, float **C, int row_1, int column_1, int row_2,
                       int column_2, int power)
{
  if (row_1 != column_1)
  {
    fprintf(stderr, "No solution\n");
    return;
  }
  if (power == 2)
  {
    multiply_matrices(A, A, C, row_1, column_1, row_1, column_1);
    return;
  }

  if (power == 1)
  {
    for (size_t i = 0; i < row_1; i++)
    {
      for (size_t j = 0; j < column_1; j++)
      {
        C[i][j] = A[i][j];
      }
    }
    return;
  }

  float **E = matrix_memory_allocation(NULL, row_1, column_1);
  if (row_1 == column_1 && power > 2)
  {
    for (size_t i = 0; i < row_1; i++)
    {
      for (size_t j = 0; j < column_1; j++)
      {
        C[i][j] = A[i][j];
      }
    }

    for (int power_1 = 1; power_1 < power; power_1++)
    {
      multiply_matrices(C, A, E, row_1, column_1, row_1, column_1);

      for (size_t i = 0; i < row_1; i++)
      {
        for (size_t j = 0; j < column_1; j++)
        {
          C[i][j] = E[i][j];
        }
      }
    }
  }
  free_matrix(E, row_1);
}

void matrix_determinant(float **A, int size, float *det)
{
  *det = 1;
  int swaps = 0;

  for (int i = 0; i < size; i++)
  {
    if (A[i][i] == 0)
    {
      int p = -1;
      for (int j = i + 1; j < size; j++)
      {
        if (A[j][i] != 0)
        {
          p = j;
          break;
        }
      }
      if (p == -1)
      {
        *det = 0;
        return;
      }
      float *ptr = A[i];
      A[i] = A[p];
      A[p] = ptr;
      swaps++;
    }
    for (int j = i + 1; j < size; j++)
    {
      float f = A[j][i] / A[i][i];
      for (int k = i; k < size; k++)
      {
        A[j][k] -= f * A[i][k];
      }
    }

    *det = *det * A[i][i];
  }
  if ((swaps % 2) == 1)
  {
    *det = -(*det);
  }
}

void matrix_print(FILE *output_file, float **C, int rows, int columns)
{
  fprintf(output_file, "%d %d\n", rows, columns);
  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < columns; j++)
    {
      fprintf(output_file, "%g ", C[i][j]);
    }
    fprintf(output_file, "\n");
  }
}

int main()
{
  char operation;
  int row_1;
  int column_1;

  int row_2;
  int column_2;

  int row_3;
  int column_3;

  int power;
  float det;
  char result;

  FILE *f = fopen("input.txt", "r");

  if (f == NULL)
  {
    fprintf(stderr, "Error: file opening\n");
    return EXIT_FAILURE;
  }

  FILE *output = fopen("output.txt", "w");

  if (output == NULL)
  {
    fprintf(stderr, "Error: file opening\n");
    return EXIT_FAILURE;
  }

  if (fscanf(f, " %c", &operation) != 1)
  {
    fprintf(stderr, "Error reading symbol\n");
  }

  if (fscanf(f, "%d %d", &column_1, &row_1) != 2)
  {
    fprintf(stderr, "Error reading parameters\n");
  }

  float **A = matrix_memory_allocation(f, row_1, column_1);

  read_matrix(f, A, row_1, column_1);

  if (operation == '+')
  {
    if (fscanf(f, "%d %d", &column_2, &row_2) == 2)
    {
      printf("%d %d\n", column_2, row_2);
    }
    else
    {
      fprintf(stderr, "Error reading parameters\n");
    }

    if (column_1 == column_2 && row_1 == row_2)
    {
      float **B = matrix_memory_allocation(f, row_2, column_2);
      read_matrix(f, B, row_2, column_2);

      float **C = matrix_memory_allocation(f, row_1, column_1);
      sum_matrices(A, B, C, row_1, column_1);

      matrix_print(output, C, row_1, column_1);
      free_matrix(B, row_2);
      free_matrix(C, row_1);
    }

    else
    {
      fprintf(output, "No solution\n");
    }
  }

  if (operation == '-')
  {
    if (fscanf(f, "%d %d", &column_2, &row_2) != 2)
    {
      fprintf(stderr, "Error reading parameters\n");
    }

    if (column_1 == column_2 && row_1 == row_2)
    {
      float **B = matrix_memory_allocation(f, row_2, column_2);
      read_matrix(f, B, row_2, column_2);
      float **C = matrix_memory_allocation(f, row_1, column_1);
      minus_matrices(A, B, C, row_1, column_1);
      matrix_print(output, C, row_1, column_1);

      free_matrix(B, row_2);
      free_matrix(C, row_1);
    }

    else
    {
      fprintf(output, "No solution\n");
    }
  }

  if (operation == '*')
  {
    if (fscanf(f, "%d %d", &column_2, &row_2) == 2)
    {
      printf("%d %d\n", column_2, row_2);
    }
    else
    {
      fprintf(stderr, "Error reading parameters\n");
    }

    if (column_1 == row_2)
    {
      float **B = matrix_memory_allocation(f, row_2, column_2);

      read_matrix(f, B, row_2, column_2);

      float **C = matrix_memory_allocation(f, row_1, column_2);

      multiply_matrices(A, B, C, row_1, column_1, row_1, column_2);
      matrix_print(output, C, row_1, column_2);

      free_matrix(B, row_2);
      free_matrix(C, row_1);
    }
    else
    {
      fprintf(output, "No solution\n");
    }
  }

  if (operation == '^')
  {
    if (row_1 == column_1)
    {
      if (fscanf(f, "%d", &power) != 1)
      {
        fprintf(stderr, "Error reading power\n");
      }

      float **C = matrix_memory_allocation(f, row_1, column_1);
      matrix_to_a_power(A, C, row_1, column_1, row_1, column_1, power);
      matrix_print(output, C, row_1, column_1);

      free_matrix(C, row_1);
    }
    else
    {
      fprintf(output, "No solution\n");
    }
  }

  if (operation == '|')
  {
    if (row_1 == column_1)
    {
      matrix_determinant(A, row_1, &det);

      fprintf(output, "%g\n", det);
    }
    else
    {
      fprintf(output, "No solution\n");
    }
  }

  free_matrix(A, row_1);
  fclose(f);
  fclose(output);
  return 0;
}

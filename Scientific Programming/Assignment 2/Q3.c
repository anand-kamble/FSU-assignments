#include <stdio.h>

#define columns 10
#define rows 10
#define FILE_NAME "matrices.dat"

/**
 * @brief This calculates the average of the numbers.
 *
 * @param numbers array of the numbers whose average is to be calculated.
 * @param size Size of the array.
 * @return double
 */
double calculate_average(double numbers[], int size)
{
    int i;
    double sum = 0;
    double average = 0.0;
    for (i = 0; i < size; i++)
    {
        sum = sum + numbers[i];
    }
    average = sum / size;
    return average;
}

/**
 * @brief Function which read the matrices from the file and multiplies them. After multiplying it calculates the average of the elements of resulting array.
 *
 * @return int
 */
int Question3()
{
    struct matriceStructure readMatrices;

    /* READING THE MATRICES FROM THE FILE.*/
    FILE *fp;
    fp = fopen(FILE_NAME, "rb");
    fread(&readMatrices, sizeof(readMatrices), 1, fp);
    fclose(fp);

    double C[columns][rows];
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            C[i][j] = 0.0;
        }
    }

    /* MATRIX MULTIPLICATION*/
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            for (int k = 0; k < columns; ++k)
            {
                C[i][j] += readMatrices.A[i][k] * readMatrices.B[k][j];
            }
        }
    }

    double rowsAvg[rows];
    for (size_t i = 0; i < rows; i++)
    {
        rowsAvg[i] = calculate_average(C[i], rows);
    }

    double average = calculate_average(rowsAvg, sizeof(rowsAvg) / sizeof(rowsAvg[0]));

    printf("\nAverage of all elements of C : %lf\n", average);

    return 0;
}
#include <stdio.h>

#define columns 10
#define rows 10
#define FILE_NAME "matrices.dat"

/**
 * @brief Structure defining how the arrays are stored.
 *
 */
struct matriceStructure
{
    double A[columns][rows];
    double B[columns][rows];
};

/**
 * @brief Function which creates two 10x10 matrices and writes it into a binary file.
 *
 * @return int
 */

int Question2()
{
    struct matriceStructure matrices;

    for (size_t i = 0; i < columns; i++)
    {
        for (size_t j = 0; j < rows; j++)
        {
            double x = i + j + 1.0;
            matrices.A[i][j] = x;
            matrices.B[i][j] = 1 / x;
        }
    }

    FILE *file = fopen(FILE_NAME, "wb");
    if (file != NULL)
    {
        fwrite(&matrices, sizeof(struct matriceStructure), 1, file);
        fclose(file);
        printf("File %s successfully created.\n", FILE_NAME);
    }

    return 0;
}

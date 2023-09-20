#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
/**
 * @brief This function reads a file into the memory and then counts the times the char 'a' and the word 'the' appears in the file.
 * @returns integer.
 */
int Question1()
{
    FILE *filePointer = fopen("random_text.txt", "r");
    fseek(filePointer, 0, SEEK_END);
    long fsize = ftell(filePointer);
    fseek(filePointer, 0, SEEK_SET);

    char *string = malloc(fsize + 1);
    fread(string, fsize, 1, filePointer);
    fclose(filePointer);

    string[fsize] = 0;

    int charCount = 0;
    int theCount = 0;
    int wordCount = 0;

    char *word = strtok(string, " ");

    while (word != NULL)
    {
        for (int i = 0; word[i] != '\0'; i++)
        {
            if (tolower(word[i]) == 'a')
            {
                charCount++;
            }
        }
        if (strcasecmp(word, "the") == 0)
        {
            theCount++;
        }

        wordCount++;
        word = strtok(NULL, " ");
    }
    printf("\n%d 'a's and %d 'the's are found inside the text.\nTotal number of words in this file is %d.\n", charCount, theCount, wordCount);

    return 0;
}

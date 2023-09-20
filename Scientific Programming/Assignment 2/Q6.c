#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Structure for a trinary tree node
 *
 */
struct TreeNode
{
    int data;
    struct TreeNode *left;
    struct TreeNode *middle;
    struct TreeNode *right;
};

/**
 * @brief Function to create a new trinary tree node
 *
 * @param data
 * @return struct TreeNode*
 */
struct TreeNode *createNode(int data)
{
    struct TreeNode *newNode = (struct TreeNode *)malloc(sizeof(struct TreeNode));
    newNode->data = data;
    newNode->left = NULL;
    newNode->middle = NULL;
    newNode->right = NULL;
    return newNode;
}

/**
 * @brief Function to insert a value into the trinary tree
 *
 * @param root Node of the tree.
 * @param data Integer to be added to the tree.
 * @return struct TreeNode*
 */
struct TreeNode *insert(struct TreeNode *root, int data)
{
    if (root == NULL)
    {
        return createNode(data);
    }

    if (data < root->data)
    {
        root->left = insert(root->left, data);
    }
    else if (data == root->data)
    {
        root->middle = insert(root->middle, data);
    }
    else
    {
        root->right = insert(root->right, data);
    }

    return root;
}

/**
 * @brief Function to find number of nodes in the tree.
 *
 * @param root Node of the tree.
 * @return int
 */
int TotalNumberOfNodes(struct TreeNode *root)
{
    if (root != NULL)
    {
        int count = 1;

        count += TotalNumberOfNodes(root->left);
        count += TotalNumberOfNodes(root->middle);
        count += TotalNumberOfNodes(root->right);

        return count;
    }
    else
    {
        return 0;
    }
}
int Question6()
{
    /*Initializing the tree.*/
    struct TreeNode *root = NULL;

    /* Appending values in the tree.*/
    for (size_t i = 0; i < 50; i++)
    {
        int r = rand();
        root = insert(root, r);
    }

    printf("Number of nodes in the trinary tree = %d\n", TotalNumberOfNodes(root));

    return 0;
}

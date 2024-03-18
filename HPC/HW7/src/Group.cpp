/**
 * @file Group.h
 * @brief Header file for the Group class used in image segmentation with k-means algorithm.
 *
 * This file defines the Group class, which represents a group of pixels with a generator color.
 * The class provides methods for setting and accessing the generator color, adding and accessing pixels,
 * and managing the count of pixels in the group.
 *
 * @name Author: Student Name: Anand Kamble
 * @date Date: 12th Feb 2024
 *
 * @note Make sure to include the necessary dependencies:
 *   - cstdlib
 *
 * @note Class:
 *   - Group: Represents a group of pixels with a generator color.
 *
 * @note Functions:
 *   - `Group()`: Default constructor for the Group class.
 *   - `Group(int r, int g, int b, int MaxPixelCount)`: Parameterized constructor for the Group class.
 *   - `void setGenerator(int r, int g, int b)`: Sets the generator color of the group.
 *   - `int* getGenerator()`: Gets the generator color of the group.
 *   - `void setpixels(int* pixels)`: Sets the pixels of the group.
 *   - `int* getpixels()`: Gets the pixels of the group.
 *   - `void addPixel(int pixel)`: Adds a pixel to the group.
 *   - `void clearPixels()`: Clears all the pixels in the group.
 *   - `void setPixelCount(int pixelCount)`: Sets the count of pixels in the group.
 *   - `int getPixelCount()`: Gets the count of pixels in the group.
 *   - `~Group()`: Destructor for the Group class.
 */

#include <stdlib.h>

using namespace std;

typedef long long int AVG;

/**
 * @brief The Group class represents a group of pixels with a generator color.
 *
 * This class stores information about the generator color, the pixels in the group,
 * and the count of pixels in the group. It also provides methods to manipulate and
 * access these properties.
 */
class Group
{
public:
    int *pixels;
    int pixelCount;
    int MaxPixelCount;
    int *generator;
    AVG average[3];

    /**
     * @brief Default constructor for the Group class.
     */
    Group();

    /**
     * @brief Parameterized constructor for the Group class.
     *
     * @param r The red component of the generator color.
     * @param g The green component of the generator color.
     * @param b The blue component of the generator color.
     * @param MaxPixelCount The maximum number of pixels that can be stored in the group.
     */
    Group(int r, int g, int b, int MaxPixelCount);

    /**
     * @brief Sets the generator color of the group.
     *
     * @param r The red component of the generator color.
     * @param g The green component of the generator color.
     * @param b The blue component of the generator color.
     */
    void setGenerator(int r, int g, int b);

    /**
     * @brief Gets the generator color of the group.
     *
     * @return int* A pointer to the array representing the generator color.
     */
    int *getGenerator();

    /**
     * @brief Sets the pixels of the group.
     *
     * @param pixels An array of pixels to be set.
     */
    void setpixels(int *pixels);

    /**
     * @brief Gets the pixels of the group.
     *
     * @return int* A pointer to the array representing the pixels.
     */
    int *getpixels();

    /**
     * @brief Adds a pixel to the group.
     *
     * @param pixel The pixel to be added.
     */
    void addPixel(int pixel);

    /**
     * @brief Clears all the pixels in the group.
     */
    void clearPixels();

    /**
     * @brief Sets the count of pixels in the group.
     *
     * @param pixelCount The count of pixels to be set.
     */
    void setPixelCount(int pixelCount);

    /**
     * @brief Gets the count of pixels in the group.
     *
     * @return int The count of pixels in the group.
     */
    int getPixelCount();

    /**
     * @brief Destructor for the Group class.
     */
    ~Group();
};

Group::Group()
{
    this->generator = new int[3];
    this->pixelCount = 0;
    this->pixels = nullptr;
    this->average[0] = 0;
    this->average[1] = 0;
    this->average[2] = 0;
}

Group::Group(int r, int g, int b, int MaxPixelCount)
{
    this->generator[0] = r;
    this->generator[1] = g;
    this->generator[2] = b;
    this->pixelCount = 0;
    this->pixels = (int *)calloc(MaxPixelCount, sizeof(int));
    this->average[0] = 0;
    this->average[1] = 0;
    this->average[2] = 0;
}

void Group::setGenerator(int r, int g, int b)
{
    this->generator[0] = r;
    this->generator[1] = g;
    this->generator[2] = b;
}

int *Group::getGenerator()
{
    return this->generator;
}

void Group::setpixels(int *pixels)
{
    this->pixels = pixels;
}

int *Group::getpixels()
{
    return this->pixels;
}

void Group::addPixel(int pixel)
{
    pixels[pixelCount] = pixel;
    pixelCount++;
}

void Group::clearPixels()
{
    /**
     * Commented since we can use the same array for the pixels and just reset the pixel count.
     * This will save us from allocating and deallocating memory for each iteration.
     * Which means we are just overwriting the pixels array with new values.
     */

    // this->pixels = (int *)calloc(this->MaxPixelCount, sizeof(int));

    this->pixelCount = 0;
}

void Group::setPixelCount(int pixelCount)
{
    this->pixelCount = pixelCount;
}

int Group::getPixelCount()
{
    return this->pixelCount;
}

Group::~Group()
{
    delete[] generator;
    delete[] pixels;
}

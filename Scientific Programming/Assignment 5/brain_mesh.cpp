#ifndef __AMK__BRAIN_MESH__

#include <iostream>
#include <cstring>
#include <string>

using namespace std;

/**
 * @brief The BrainMesh class represents a mesh object with an ID.
 *
 * This class provides functionality to manage a BrainMesh object,
 * including creating, copying, and displaying information about the object.
 */
class BrainMesh
{
private:
    string id;               ///< ID of the BrainMesh object.
    bool debug_mode = false; ///< Indicates whether debug messages should be printed.

    /**
     * @brief Private utility function for debugging.
     *
     * This function prints the given message to the standard output
     * if the debug mode is enabled.
     *
     * @param msg The message to be printed.
     */
    void debug(string msg)
    {
        if (this->debug_mode)
            cout << msg << endl;
    }

    /**
     * @brief Utility function for cloning debug mode setting from another BrainMesh object.
     *
     * This function clones the debug mode setting from another BrainMesh object.
     *
     * @param other The source BrainMesh object to clone the debug mode from.
     */
    void clone(const BrainMesh &other)
    {
        this->debug_mode = other.debug_mode;
    }

public:
    /**
     * @brief Constructor for initializing the BrainMesh object with the given ID and optional debug mode.
     *
     * This constructor initializes the BrainMesh object with the provided ID and optional debug mode.
     * It also displays a debug message indicating object creation.
     *
     * @param Id The ID to be assigned to the BrainMesh object.
     * @param debug_mode Whether debug messages should be printed (default is false).
     */
    BrainMesh(const string Id, bool debug_mode = false)
    {
        this->id = Id;
        this->debug_mode = debug_mode;
        this->debug("Constructor: " + this->id);
    }

    /**
     * @brief Destructor for cleaning up resources.
     *
     * This destructor clears the ID string when the BrainMesh object is destroyed.
     */
    ~BrainMesh()
    {
        this->id.clear();
    }

    /**
     * @brief Copy constructor for creating a new BrainMesh object by copying another object's ID and debug mode setting.
     *
     * This constructor creates a new BrainMesh object by copying the ID and debug mode setting
     * from another BrainMesh object. It also displays a debug message indicating object creation.
     *
     * @param other The BrainMesh object to be copied.
     */
    BrainMesh(const BrainMesh &other)
    {
        this->id = "C_C_" + other.id;
        clone(other);
        this->debug("Copy Constructor: " + this->id);
    }

    /**
     * @brief Copy assignment operator for assigning the ID and debug mode from another BrainMesh object to this object.
     *
     * This operator assigns the ID and debug mode from another BrainMesh object to this object.
     * It also displays a debug message indicating object assignment.
     *
     * @param other The BrainMesh object whose ID and debug mode are to be assigned.
     * @return Reference to this BrainMesh object after assignment.
     */
    BrainMesh &operator=(const BrainMesh &other)
    {
        if (this != &other)
        {
            this->id = "C_A_O_" + other.id;
            clone(other);
            this->debug("Copy Assignment Operator: " + this->id);
        }
        return *this;
    }

    /**
     * @brief Displays the ID of the BrainMesh object.
     *
     * This function prints the ID of the BrainMesh object to the standard output.
     */
    void info()
    {
        cout << "ID : " << this->id << endl;
    }
};

#endif

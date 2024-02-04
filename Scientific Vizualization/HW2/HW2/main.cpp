/**
 * @file main.cpp
 * @brief This file contains the main function for an OpenGL assignment to render a square.
 *
 * In this OpenGL assignment, the goal is to generate a dynamic square using the glDrawElements function.
 * Each vertex of the square is assigned a distinct color that evolves over time, creating a lively appearance.
 * To enhance visual interest, the glm::rotate function is utilized to make the square rotate within the x-y plane.
 * The implementation is facilitated by the Shader class from "shader.h".
 *
 * @name Student Name: Anand Kamble
 * @date Date: 2nd Feb 2024
 *
 * @note Make sure to include the necessary dependencies:
 *   - glad/glad.h
 *   - GLFW/glfw3.h
 *   - glm/glm.hpp
 *   - glm/gtc/matrix_transform.hpp
 *   - "shader.h"
 *
 * @note Create and use a shader program:
 *   - The vertex and fragment shader sources are specified in "shader.vs" and "shader.fs" respectively.
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shader.h"
#include <iostream>

// Function declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height); // GLFW: Whenever the window size changes, this callback function executes.
void processInput(GLFWwindow *window); // Process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly.

// Constants
const unsigned int SCR_WIDTH = 800; // Screen width
const unsigned int SCR_HEIGHT = 800; // Screen height

using namespace std;
using namespace glm;

int main()
{
    glfwInit(); // Initialize GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // Set the OpenGL version to 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // Set the OpenGL version to 3.3
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Set the OpenGL profile to core

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "First OpenGL Homework", NULL, NULL); // Create a window
    if (window == NULL) // Check if the window was created successfully
    {
        std::cout << "Failed to create GLFW window" << std::endl; 
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // Make the window the current context
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // Set the callback function for window resizing

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) // Load GLAD to manage OpenGL function pointers and Check if GLAD was initialized successfully
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions         // colors
        0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom right
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
        -0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f,  // top left
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f};  // top right

    GLuint indices[] = {
        0, 1, 2,
        3, 2, 0};

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // Bind the Vertex Array Object, then bind and set vertex buffer(s), and configure vertex attributes(s)
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO
    glBindVertexArray(0);

    // Create and use shader program
    Shader ourShader("shader.vs", "shader.fs"); // Build and compile our shader program
    ourShader.use(); // Use our shader program when we want to render an object

    // Set initial shader parameters
    float time = 0.0f;
    ourShader.setFloat("time", time); // Set the time parameter in the shader

    float now,lastTime,delta;
    

    time = glfwGetTime();

    // Set initial transformation matrix
    float angleOfRotation = 30.0f; // This defines that the sqaure will rotate 30 degrees per second
    vec3 axis(0.0f, 0.0f, 1.0f); // The axis of rotation is the z-axis
    mat4 trans = mat4(1.0f); // Identity matrix
    trans = rotate(trans, radians(0.0f), axis); // Initial rotation angle is 0
    ourShader.setMat4("trans", trans); // Set the transformation matrix in the shader

    // Render loop
    /**
     * @note The render loop continuously updates parameters, renders the square, and handles user input.
     */

    while (!glfwWindowShouldClose(window))
    {
        // Handle input
        processInput(window);

        // Update shader parameters
        now = glfwGetTime(); // Get the current time
        delta = now - lastTime; // Calculate the time difference
        lastTime = now; // Update the last time
        time += delta; // Update the time parameter
        ourShader.setFloat("time", time); // Update the time parameter in the shader

        // Update transformation matrix
        trans = rotate(trans, (delta)/radians(angleOfRotation), axis); // Update the rotation angle
        ourShader.setMat4("trans", trans); // Update the transformation matrix in the shader

        // Render
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render the square
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(indices[0]), GL_UNSIGNED_INT, indices);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // De-allocate resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // Terminate GLFW, clearing all previously allocated resources
    glfwTerminate();
    return 0;
}

// Process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// GLFW: Whenever the window size changes, this callback function executes
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // Make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

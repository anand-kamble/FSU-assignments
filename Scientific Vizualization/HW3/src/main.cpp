
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include "shader.h"

#include "Cube.cpp"

#define FRAGMENT_SHADER_PATH "src/Shader/shader.frag"
#define VERTEX_SHADER_PATH "src/Shader/shader.vert"

// Function declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height); // GLFW: Whenever the window size changes, this callback function executes.
void processInput(GLFWwindow *window);                                     // Process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly.

// Constants
const unsigned int SCR_WIDTH = 800;  // Screen width
const unsigned int SCR_HEIGHT = 800; // Screen height

using namespace std;
using namespace glm;

int main()
{
    glfwInit();                                                    // Initialize GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);                 // Set the OpenGL version to 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);                 // Set the OpenGL version to 3.3
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Set the OpenGL profile to core

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "First OpenGL Homework", NULL, NULL); // Create a window
    if (window == NULL)                                                                                // Check if the window was created successfully
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);                                    // Make the window the current context
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // Set the callback function for window resizing

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) // Load GLAD to manage OpenGL function pointers and Check if GLAD was initialized successfully
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // Use our shader program when we want to render an object

    // Set initial shader parameters
    // float time = 0.0f;
    // ourShader.setFloat("time", time); // Set the time parameter in the shader

    float now, lastTime, delta;

    // time = glfwGetTime();

    // Set initial transformation matrix
    // float angleOfRotation = 30.0f;              // This defines that the sqaure will rotate 30 degrees per second
    // vec3 axis(0.0f, 0.0f, 1.0f);                // The axis of rotation is the z-axis
    // mat4 trans = mat4(1.0f);                    // Identity matrix
    // trans = rotate(trans, radians(0.0f), axis); // Initial rotation angle is 0
    // ourShader.setMat4("trans", trans);          // Set the transformation matrix in the shader

    Cube UpperArm = Cube("UpperArm");
    UpperArm.updateScreenSize(SCR_WIDTH, SCR_HEIGHT);
    UpperArm.applyShader(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH);
    UpperArm.applyTexture("src/Texture/aluminium.jpg");

    while (!glfwWindowShouldClose(window))
    {
        // Handle input
        processInput(window);

        // Update shader parameters
        now = glfwGetTime();    // Get the current time
        delta = now - lastTime; // Calculate the time difference
        lastTime = now;         // Update the last time
        // time += delta;                    // Update the time parameter
        // ourShader.setFloat("time", time); // Update the time parameter in the shader

        // Update transformation matrix
        // trans = rotate(trans, (delta) / radians(angleOfRotation), axis); // Update the rotation angle
        // ourShader.setMat4("trans", trans);                               // Update the transformation matrix in the shader

        // Render
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

        UpperArm.render();
        // Render the square
        // glBindVertexArray(VAO);
        // glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(indices[0]), GL_UNSIGNED_INT, indices);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // De-allocate resources
    // glDeleteVertexArrays(1, &VAO);
    // glDeleteBuffers(1, &VBO);

    UpperArm.destroy();

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

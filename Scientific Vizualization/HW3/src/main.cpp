/**
 * @file main.cpp
 * @brief OpenGL code to render a robotic arm with a camera that can be rotated using the keyboard.
 *
 * @name Author: Student Name: Anand Kamble
 * @date Date: 19th Feb 2024
 *
 * @note Usage:
 *   Press 'S' to rotate the lower arm.
 *   Press 'E' to rotate the upper arm.
 *   Press 'O' to open the finger.
 *   Press 'C' to close the finger.
 *   Press 'R' to rotate the camera.
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_m.h>

#include <iostream>

using namespace glm;

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

/**
 * Angles for the lower arm, upper arm, finger and camera with the initial values.
 * I haven't put these inside the main function scope because I want to use them in the processInput function
 * Which is outside the main function scope.
 */
float LowerArmAngle = 20.0f;
float UpperArmAngle = 40.0f;
float FingerAngle = 20.0f;
float cameraAngle = 0.0f;

int main()
{
    // This is the LearnOpenGL code for creating a window and setting up the OpenGL context.
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL – Robot", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);
    // End of LearnOpenGL code

    // Loading the shader files.
    Shader ourShader("src/Shader/shader.vert", "src/Shader/shader.frag");

    // Vertices for the cube
    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,

        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,

        -0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 1.0f, 0.0f,

        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f};

    // Creating the VAO and VBO for the cube and setting up the vertex attributes.
    // I have used the same code from LearnOpenGL for this.
    // https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/7.1.camera_circle/camera_circle.cpp
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    unsigned int texture1;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ===================== Load and create a texture =====================
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load("src/texture/aluminium.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    ourShader.use();
    ourShader.setInt("texture1", 0);

    mat4 projection = perspective(radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    ourShader.setMat4("projection", projection);

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);

        ourShader.use();

        // Setting the camera position.
        mat4 view = mat4(1.0f);
        float radius = 10.0f;
        float camX = static_cast<float>(sin(cameraAngle) * radius);
        float camZ = static_cast<float>(cos(cameraAngle) * radius);
        view = lookAt(vec3(camX, 0.0f, camZ), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
        ourShader.setMat4("view", view);

        glBindVertexArray(VAO);

        auto ArmScale = vec3(2.0f, 0.4f, 0.4f);

        // ===================== Lower Arm =====================
        auto model = mat4(1.0f);

        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = scale(model, ArmScale);
        model = translate(model, vec3(0.5f, 0.0f, 0.0f));

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // ===================== Upper Arm =====================
        model = mat4(1.0f); // reset it to identity matrix
        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = rotate(model, radians(UpperArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = scale(model, ArmScale);

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // ===================== Finger 1 =====================
        model = mat4(1.0f);
        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = rotate(model, radians(UpperArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = rotate(model, radians(FingerAngle / 2.f), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(0.5f, 0.0f, 0.0f));
        model = scale(model, vec3(1.f, 0.2f, 0.2f));

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // ===================== Finger 2 =====================
        model = mat4(1.0f);
        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = rotate(model, radians(UpperArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = rotate(model, radians(-FingerAngle / 2.f), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(0.5f, 0.0f, 0.0f));
        model = scale(model, vec3(1.f, 0.2f, 0.2f));

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // If 'S' is pressed, rotate the lower arm.
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        LowerArmAngle += 0.03f;

    // If 'E' is pressed, rotate the upper arm.
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        UpperArmAngle += 0.03f;

    // If 'O' is pressed, open the finger. While opening the finger, the angle should not go above 45.
    // i.e. the angle between the two fingers should not be more than 90 degrees.
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && FingerAngle < 45.0f)
        FingerAngle += 0.03f;

    // If 'C' is pressed, close the finger. While closing the finger, the angle should not go below 0.
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && FingerAngle > 0.0f)
        FingerAngle -= 0.03f;

    // If 'R' is pressed, rotate the camera.
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        cameraAngle += 0.001f;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{

    glViewport(0, 0, width, height);
}
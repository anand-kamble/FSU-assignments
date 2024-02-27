/**
 * @file main.cpp
 * @brief OpenGL code  to add illumination to the robot.
 * @author Student Name: Anand Kamble
 * @date Date: 25th Feb 2024
 * @package Scientific Vizualization/HW4/
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <iostream>

#include "utils/utils.cpp"

#define SHADER_SRC = "src/Shader/";

using namespace glm;

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

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

/**
 * Using the Camera class from LearnOpenGL to create a camera object.
 */
Camera camera(vec3(0.0f, 0.0f, 5.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Setting the material properties.
unsigned int materialIndex = 0;
Material *Selected_Material = getNewMaterial(materialIndex);

/**
 * I am using this function to update the material for the objects every second.
 * @param time The current time.
 * @param next A boolean sent when the material needs to be updated for a different part of the robot, and not every second.
 */
void UpdateMaterial(float time, bool next = false)
{
    if (!next)
    {
        if ((int)time % NUM_MATERIALS != materialIndex)
        {
            materialIndex = (int)time % NUM_MATERIALS;
            Selected_Material = getNewMaterial(materialIndex);
        }
    }
    else
    {
        materialIndex < NUM_MATERIALS ? materialIndex++ : materialIndex = 0;
        Selected_Material = getNewMaterial(materialIndex);
    }
}

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
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Robot Illumination", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback); ///< This is the callback for the mouse movement.
    glfwSetScrollCallback(window, scroll_callback);   ///< This is the callback for the mouse scroll.
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);
    // End of LearnOpenGL code

    // Loading the shader files.
    Shader ourShader("src/Shader/material.vert", "src/Shader/material.frag");
    Shader lightCubeShader("src/Shader/light.vert", "src/Shader/light.frag");

    /**
     * This array represents the vertices of a 3D cube in a 3D space.
     * Each vertex is represented by 6 float values:
     * - The first 3 values represent the x, y, and z coordinates of the vertex.
     * - The next 3 values represent the normal vector of the vertex.
     *
     * The cube is defined with 36 vertices (6 faces with 2 triangles per face, and 3 vertices per triangle).
     */
    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f,

        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,

        -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f,

        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f};

    // Creating the VAO and VBO for the cube and setting up the vertex attributes.
    // I have used the same code from LearnOpenGL for this.
    // https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/3.1.materials/materials.cpp
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // ===================== Light Cube VAO =====================
    unsigned int lightCubeVAO;
    glGenVertexArrays(1, &lightCubeVAO);
    glBindVertexArray(lightCubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    vec3 lightPos(5.f, 5.f, 5.f); // Setting the light position.

    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        UpdateMaterial(currentFrame); ///< Updating the material every second.

        processInput(window);

        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ourShader.use();

        // ===================== Light Cube =====================
        ourShader.use();
        ourShader.setVec3("light.position", camera.Position);
        ourShader.setVec3("viewPos", camera.Position);

        vec3 lightColor(1.0f, 1.0f, 1.0f);
        vec3 diffuseColor = lightColor * vec3(0.5f);   ///< decrease the influence
        vec3 ambientColor = diffuseColor * vec3(0.2f); ///< low influence
        ourShader.setVec3("light.ambient", ambientColor);
        ourShader.setVec3("light.diffuse", diffuseColor);
        ourShader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);

        mat4 projection = perspective(radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        auto view = camera.GetViewMatrix();
        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);

        glBindVertexArray(VAO);

        auto ArmScale = vec3(2.0f, 0.4f, 0.4f);

        // ===================== Lower Arm =====================
        auto model = mat4(1.0f);

        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = scale(model, ArmScale);
        model = translate(model, vec3(0.5f, 0.0f, 0.0f));

        // Updating the material for the lower arm.
        UpdateMaterial(currentFrame, true);                                     ///< Updating the material for every part.
        ourShader.setVec3("material.ambient", Selected_Material->ambient);      ///< Setting the ambient material property.
        ourShader.setVec3("material.diffuse", Selected_Material->diffuse);      ///< Setting the diffuse material property.
        ourShader.setVec3("material.specular", Selected_Material->specular);    ///< Setting the specular material property.
        ourShader.setFloat("material.shininess", Selected_Material->shininess); ///< Setting the shininess material property.

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

        // Updating the material for the Upper arm.
        UpdateMaterial(currentFrame, true);
        ourShader.setVec3("material.ambient", Selected_Material->ambient);
        ourShader.setVec3("material.diffuse", Selected_Material->diffuse);
        ourShader.setVec3("material.specular", Selected_Material->specular);
        ourShader.setFloat("material.shininess", Selected_Material->shininess);

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

        // Updating the material for the finger.
        UpdateMaterial(currentFrame, true);
        ourShader.setVec3("material.ambient", Selected_Material->ambient);
        ourShader.setVec3("material.diffuse", Selected_Material->diffuse);
        ourShader.setVec3("material.specular", Selected_Material->specular);
        ourShader.setFloat("material.shininess", Selected_Material->shininess);

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

        // Updating the material for the finger.
        UpdateMaterial(currentFrame, true);
        ourShader.setVec3("material.ambient", Selected_Material->ambient);
        ourShader.setVec3("material.diffuse", Selected_Material->diffuse);
        ourShader.setVec3("material.specular", Selected_Material->specular);
        ourShader.setFloat("material.shininess", Selected_Material->shininess);

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        lightCubeShader.use();
        lightCubeShader.setMat4("projection", projection);
        lightCubeShader.setMat4("view", view);
        model = mat4(1.0f);
        model = translate(model, lightPos);
        model = scale(model, vec3(0.2f)); // a smaller cube
        lightCubeShader.setMat4("model", model);

        glBindVertexArray(lightCubeVAO);
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

    // Camera controls
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn / 100);
    float ypos = static_cast<float>(yposIn / 100);
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
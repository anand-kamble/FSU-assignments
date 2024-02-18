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




float LowerArmAngle = 20.0f;
float UpperArmAngle = 40.0f;
float FingerAngle = 0.0f;
float cameraSpeedFactor = 0.0f;

int main()
{
    
    
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    
    
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Anand", NULL, NULL);
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

    
    
    Shader ourShader("src/Shader/shader.vert", "src/Shader/shader.frag");

    
    
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

        
        mat4 view = mat4(1.0f); 
        float radius = 10.0f;

        float camX = static_cast<float>(sin(cameraSpeedFactor) * radius);
        float camZ = static_cast<float>(cos(cameraSpeedFactor) * radius);
        view = lookAt(vec3(camX, 0.0f, camZ), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
        ourShader.setMat4("view", view);

        glBindVertexArray(VAO);

        auto ArmScale = vec3(2.0f, 0.4f, 0.4f);
        vec3 armPosition = vec3(0.0f, 0.0f, 0.0f);
        
        
        auto model = mat4(1.0f);

        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = scale(model, ArmScale);
        model = translate(model, vec3(0.5f, 0.0f, 0.0f));

        armPosition.x = model[0][0];
        armPosition.y = model[1][1];
        armPosition.z = model[2][2];

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        
        model = mat4(1.0f);
        model = rotate(model, radians(LowerArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = rotate(model, radians(UpperArmAngle), vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, vec3(1.0f, 0.0f, 0.0f));
        model = scale(model, ArmScale);

        ourShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        
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
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        LowerArmAngle += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        UpperArmAngle += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
        FingerAngle += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        FingerAngle -= 0.03f;
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        cameraSpeedFactor += 0.01f;
}



void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    
    
    glViewport(0, 0, width, height);
}
#version 330 core

// Input attribute: position of the vertex in 3D space
layout(location = 0) in vec3 aPos;

// Input attribute: color of the vertex
layout(location = 1) in vec3 aColor;

// Output variable: color to be sent to the fragment shader
out vec3 ourColor;

// Uniform variable: transformation matrix provided by the application
uniform mat4 trans;

void main() {
    // Calculate the final position of the vertex after applying the transformation matrix
    gl_Position = trans * vec4(aPos, 1.0);

    // Pass the color of the vertex to the fragment shader
    ourColor = aColor;
}

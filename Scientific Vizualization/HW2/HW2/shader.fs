#version 330 core

// Output color variable that will be used to set the final color of the fragment
out vec4 FragColor;

// Input color variable received from the vertex shader
in vec3 ourColor;

// Uniform variable representing time, provided by the application
uniform float time;

void main() {
    // Calculate the final color for the fragment
    // Using sin and cos functions to create dynamic color changes over time
    // Applying absolute value and dividing by 2 for normalization
    FragColor = vec4(
        (ourColor.x + abs(sin(time)) / 2),
        (ourColor.y + abs(cos(time / 4)) / 2),
        (ourColor.z + abs(sin(time / 2)) / 2),
        1.0
    );
}

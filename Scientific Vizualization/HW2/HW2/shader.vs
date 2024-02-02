#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 ourColor;

uniform float time;
uniform mat4 trans;

void main() {
    gl_Position = trans * vec4(aPos, 1.0);
    ourColor = aColor;
}
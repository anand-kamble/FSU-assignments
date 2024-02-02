#version 330 core
out vec4 FragColor;

in vec3 ourColor;

uniform float time;

void main() {
    FragColor = vec4((ourColor.x + abs(sin(time))/2), 
    (ourColor.y + abs(cos(time/4))/2), 
    (ourColor.z + abs(sin(time / 2)))/2, 1.0);
}
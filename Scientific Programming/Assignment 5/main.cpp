#include "brain_mesh.cpp"

int main (){

    BrainMesh brain_Mesh("First",true);
    BrainMesh second = brain_Mesh;
    BrainMesh newBren("Secin");
    newBren = brain_Mesh;


    return 0;
}
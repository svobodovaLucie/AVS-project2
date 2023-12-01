/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Lucie Svobodová <xsvobo1x@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    01 December 2023
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

#define CUT_OFF 1   // grid size cut-off

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::octTreeDecompose(const ParametricScalarField &field, Vec3_t<float> &offset, unsigned gridSize)
{
    unsigned totalTriangles = 0;
    // vyloucit to, kde neni mozne, ze by byl povrch -> tam se nebude volat metoda buildCube
    // na kazde urovni se prostor deli na 8 potomku
    // pruchod stromem
    //    1. aktualni blok rozdelen na 8 potomku (na zacatku cela mrizka)
    // prvni cubeSize = mGridSize
    // dalsi cubeSize = tmp = cubeSize/2
    // rozdelit na 8 potomku
    //    2. pro kazdeho potomka overit, zda je mozne, aby jeho podprostorem prochazel hledany povrch (isosurface)
    //       podminka prazdnosti bloku je v rovnici 5.3.
    // podminka prazdnosti bloku
    float size = float(gridSize) * BaseMeshBuilder::mGridSize;
    float halfSize = (float(gridSize) * BaseMeshBuilder::mGridSize)/2.0;
    float midOffsetX = offset.x * BaseMeshBuilder::mGridResolution + halfSize;
    float midOffsetY = offset.y * BaseMeshBuilder::mGridResolution + halfSize;
    float midOffsetZ = offset.z * BaseMeshBuilder::mGridResolution + halfSize;
	const Vec3_t<float> mid(midOffsetX, midOffsetY, midOffsetZ);
    float l = BaseMeshBuilder::mIsoLevel;
    float sqrt32 = sqrtf(3.0)/2.0;
    // exp je vysledek rovnice 5.3 -> true je prazdny asi, false je neprazdny
    bool exp = evaluateFieldAt(mid, field) > l + sqrt32 * size;

    //    3. kazdy neprazdny potomek je rozdelen na dalsich 8 potomku az do cut-off hloubky nebo velikosti hrany a
    if (exp) {
        return 0;
    }

    //    4. na nejnizsi urovni jsou volanim buildCude vygenerovany samotne polygony pro vsechny krychle nalezejici do daneho podprostoru
    // cut-off overi, ze jsme na nejnizsi urovni 
    //  - tohle je asi mozne dat hned na zacatek,
    //  at se pripadne nepocitaji vsechny predesle veci, 
    //  kdyz mame uz cut-off proste 1
    if (gridSize <= CUT_OFF) {
        return buildCube(offset, field);
    }

// pole, kde budou jednotlive pricitane konstanty???
    unsigned y[8] = {0,0,1,1,0,0,1,1};
    // unsigned z[8] = {0,0,0,0,1,1,1,1};
    // 000  0
    // 001  1
    // 010  2
    // 011  3
    // 100  4
    // 101  5
    // 110  6
    // 111  7

    for (unsigned i = 0; i < 8; i++) {
        // #pragma omp task shared(totalTriangles)
        float newOffsetX = offset.x + (i%2)*(gridSize/2);
        float newOffsetY = offset.y + y[i]*(gridSize/2);
        float newOffsetZ = offset.z + float(i>=4)*(gridSize/2);
        Vec3_t<float> newOffset(newOffsetX, newOffsetY, newOffsetZ);
        unsigned trianglesCount = octTreeDecompose(field, newOffset, gridSize/2);
        // #pragma omp critical
        totalTriangles += trianglesCount;
    }

    // #pragma omp taskwait
    return totalTriangles;
}


unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    Vec3_t<float> initialOffset(0.0, 0.0, 0.0);
    unsigned totalTriangles;
    // #pragma omp parallel
    // #pragma omp single
    // totalTriangles = octTree(initialPosition, mGridSize, field);
    totalTriangles = octTreeDecompose(field, initialOffset, BaseMeshBuilder::mGridSize);
    
    return totalTriangles;
}



float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrtf(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}

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

unsigned TreeMeshBuilder::octTreeDecomposition(const ParametricScalarField &field, Vec3_t<float> &offset, unsigned gridSize)
{
    // if the cube is of a minimal size, end the decomposition
    if (gridSize <= CUT_OFF) {
        return buildCube(offset, field);
    }

    // check if the current block of cubes is empty
    float midOffsetX = offset.x * BaseMeshBuilder::mGridResolution + (float(gridSize) * BaseMeshBuilder::mGridSize)/2.0;
    float midOffsetY = offset.y * BaseMeshBuilder::mGridResolution + (float(gridSize) * BaseMeshBuilder::mGridSize)/2.0;
    float midOffsetZ = offset.z * BaseMeshBuilder::mGridResolution + (float(gridSize) * BaseMeshBuilder::mGridSize)/2.0;
	const Vec3_t<float> mid(midOffsetX, midOffsetY, midOffsetZ);
    bool isEmpty = evaluateFieldAt(mid, field) > BaseMeshBuilder::mIsoLevel + (sqrtf(3.0)/2.0) * (float(gridSize) * BaseMeshBuilder::mGridSize);

    // if the block is empty, end the decomposition
    if (isEmpty) {
        return 0;
    }
    
    // compute the total number of triangles
    unsigned totalTriangles = 0;

    // decompose the current block of cubes to 8 cubes 
    // and call the octTreeDecomposition function recursively
    unsigned y[8] = {0,0,1,1,0,0,1,1};
    for (unsigned i = 0; i < 8; i++) {
        #pragma omp task shared(totalTriangles)
        {
            float newOffsetX = offset.x + (i%2)*(gridSize/2.0);
            float newOffsetY = offset.y + y[i]*(gridSize/2.0);
            float newOffsetZ = offset.z + float(i>=4)*(gridSize/2.0);
            Vec3_t<float> newOffset(newOffsetX, newOffsetY, newOffsetZ);
            unsigned trianglesCount = octTreeDecomposition(field, newOffset, gridSize/2);
            #pragma omp atomic update
            totalTriangles += trianglesCount;
        }
    }

    // wait for the children tasks to finish
    #pragma omp taskwait
    return totalTriangles;
}


unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    Vec3_t<float> initialOffset(0.0, 0.0, 0.0);
    unsigned totalTriangles;
    #pragma omp parallel
    #pragma omp single
    totalTriangles = octTreeDecomposition(field, initialOffset, BaseMeshBuilder::mGridSize);
    
    return totalTriangles;
}


// method called from buildCube()
float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
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

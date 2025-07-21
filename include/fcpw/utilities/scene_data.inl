namespace fcpw {

template<size_t DIM>
inline SceneData<DIM>::SceneData():
aggregate(nullptr),
ignoreSilhouette({})
{

}

template<size_t DIM>
inline void SceneData<DIM>::clearObjectData()
{
    /*soups.clear();
    soupToObjectsMap.clear();
    pointObjects.clear();
    lineSegmentObjects.clear();
    triangleObjects.clear();
    silhouetteVertexObjects.clear();
    silhouetteEdgeObjects.clear();
    instanceTransforms.clear();
    csgTree.clear();
    ignoreSilhouette = {};*/
}

template<size_t DIM>
inline void SceneData<DIM>::clearAggregateData()
{
    /*pointObjectPtrs.clear();
    lineSegmentObjectPtrs.clear();
    triangleObjectPtrs.clear();
    silhouetteVertexObjectPtrs.clear();
    silhouetteEdgeObjectPtrs.clear();
    silhouetteObjectPtrStub.clear();
    aggregateInstancePtrs.clear();
    aggregateInstances.clear();
    aggregate = nullptr;*/
}

} // namespace fcpw

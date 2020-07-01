namespace fcpw {

template<size_t DIM>
inline SceneData<DIM>::SceneData():
aggregate(nullptr)
{

}

template<size_t DIM>
inline void SceneData<DIM>::clearObjectData()
{
	soups.clear();
	soupToObjectsMap.clear();
	lineSegmentObjects.clear();
	triangleObjects.clear();
	instanceTransforms.clear();
	csgTree.clear();
}

template<size_t DIM>
inline void SceneData<DIM>::clearAggregateData()
{
	lineSegmentObjectPtrs.clear();
	triangleObjectPtrs.clear();
	mixedObjectPtrs.clear();
	aggregateInstancePtrs.clear();
	aggregateInstances.clear();
	aggregate = nullptr;
}

} // namespace fcpw

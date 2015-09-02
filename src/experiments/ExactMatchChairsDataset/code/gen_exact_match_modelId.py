
modelIds = [x.rstrip() for x in open('../filelist_chair_6777.txt','r')]
modelId_modelIndex_map = {}
modelIndex_modelId_map = {}
for k in range(len(modelIds)):
  modelId = modelIds[k]
  modelId_modelIndex_map[modelId] = k+1
  modelIndex_modelId_map[k+1] = modelId


shape_models = [x.rstrip() for x in open('../filelist_exactmatch_chair_105.txt','r')]

fout = open('exact_match_chairs_shape_modelIds_0to6776.txt', 'w')
for k in xrange(len(shape_models)):
  print k
  modelId = shape_models[k]
  modelIndex = modelId_modelIndex_map[modelId]
  fout.write(str(modelIndex-1)+'\n')
fout.close()

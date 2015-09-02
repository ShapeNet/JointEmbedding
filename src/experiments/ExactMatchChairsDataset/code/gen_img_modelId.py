
modelIds = [x.rstrip() for x in open('../filelist_chair_6777.txt','r')]
modelId_modelIndex_map = {}
modelIndex_modelId_map = {}
for k in range(len(modelIds)):
  modelId = modelIds[k]
  modelId_modelIndex_map[modelId] = k+1
  modelIndex_modelId_map[k+1] = modelId


img_filenames = [x.rstrip().split('/')[-1] for x in open('../exact_match_chairs_img_filelist.txt','r')]

fout = open('exact_match_chairs_img_modelIds_0to6776.txt', 'w')
for k in xrange(len(img_filenames)):
  print k
  modelId = img_filenames[k].split('_')[0]
  modelIndex = modelId_modelIndex_map[modelId]
  fout.write(str(modelIndex-1)+'\n')
fout.close()

import os
import Image

all_modelIds = [x.rstrip() for x in open('filelist_chair_6777.txt','r')]
modelIds = [x.rstrip() for x in open('filelist_exactmatch_chair_105.txt','r')]
for modelId in modelIds:
  assert(modelId in all_modelIds)

for modelId in modelIds:
  img_names = os.listdir('exactmatch_selected_cropped/'+modelId)
  print img_names
  j = 1
  for k in range(len(img_names)):
    try:
      im = Image.open('exactmatch_selected_cropped/'+modelId+'/'+img_names[k])
      im.save('ExactMatchChairs/' + modelId+'_'+str(j)+'.jpg', 'JPEG')
      j += 1
    except:
      print "Cannot open: " + modelId + '/' + img_names[k]
    #os.rename(modelId+'/'+imgs[k], modelId+'/'+modelId+'_'+str(k)+

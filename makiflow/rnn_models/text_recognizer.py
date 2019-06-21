from __future__ import absolute_import
import tensorflow as tf

charList = 'abcdefg'

def toSparse(texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)

if __name__ == "__main__":
   a = toSparse(['ab', 'fd', 'aa'])
   print(a)
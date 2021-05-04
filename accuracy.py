#! /usr/bin/python
# -*- encoding: utf-8 -*-

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	# output是一个（batch，1）的分数张量
	# topk函数的返回值是  值向量， 位置向量（即原来的index）
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0/batch_size))
	return res

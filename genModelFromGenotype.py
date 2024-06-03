from genotypes import Genotype

normal=[(1, 0, 'max_pool_7x7', 'sep_conv_7x7', 'add'), (1, 1, 'max_pool_5x5', 'sep_conv_3x1', 'concat'), (1, 1, 'sep_conv_5x1', 'max_pool_7x7', 'concat'), (3, 2, 'dil_conv_5x5', 'dil_conv_7x7', 'add')]
normal_concat={4, 5}
reduce=[(0, 0, 'sep_conv_5x5', 'max_pool_5x5', 'add'), (2, 1, 'sep_conv_7x1', 'max_pool_5x5', 'add'), (1, 3, 'dil_conv_7x7', 'sep_conv_5x1', 'concat'), (0, 3, 'sep_conv_5x5', 'noisy', 'concat')]
reduce_concat={4, 5}

genotype = Genotype(normal = normal, normal_concat = normal_concat,
                        reduce = reduce, reduce_concat = reduce_concat)


from model import Network

# genmodel=Network(genotype,num_classes=20)
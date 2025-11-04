# super-ai-symbol
# arc-gai first


2025-11-04:
tpj_agi_arc_data_generator_solver OK.py   完整清晰的实现针对ARC-AGI的DSL定义状态计算和网格操作DSL
arc_size_predict_improssible OK.py  网格跨任务变长，任务内输入输出也是变长的，这是一个挑战，用fixed-length-pad和special-token是相对靠谱但很浪费计算的路线，提前预测长度也是逻辑上不现实的，应为输出长度本身只有在操作完毕之后属于输出网格的一个属性而已，如果能准确提前预测，说明操作本身已经知道了
tpj_agi_dynamic_shape-2 SO.py  尝试寻找有效的超动态处理办法，结论是没有太好的办法，需要步步动态输出长度，对目前的AI框架来说很不自然，而且对batch和train的时候就更麻烦
alpha-npi-for-grid-complex OK $$.py  尝试探索AlphaNPI这种AlphaZeroZero和代码合成的路线，这是相对靠谱的路线，但仍然需要解决Action-Space的合成（神经学习还是自底向上组合的问题）  这个方向是重点方向
graph-engine-simple OK $$.py  尝试探索动作空间的构造，从最底层的计算模型（MinskyRegisterMatchine/Brainfuck开始肯定太低级，直接用C/C++/Java/Python等高级的图灵完备的语言有太泛泛，这里尝试结合更通用的逻辑表达但又不需要高级语言那么自由的中间表示（简单版本）：recursive looped directed graph作为一种通用的表示，按照任务来设计其中的state/action等操作，所以不是全自动的。



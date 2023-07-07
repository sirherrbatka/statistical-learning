(cl:in-package #:statistical-learning.tree-protocol)


(defmacro switch-direction ((direction) left right &optional middle)
  `(switch (,direction)
     (sl.opt:left ,left)
     (sl.opt:right ,right)
     (sl.opt:middle ,middle)))

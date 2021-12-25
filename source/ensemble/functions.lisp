(cl:in-package #:statistical-learning.ensemble)


(defun trees-predict (ensemble trees data parallel
                      &optional state)
  (let* ((parameters (statistical-learning.mp:parameters ensemble))
         (tree-parameters (tree-parameters parameters))
         (state (contribute-trees ensemble tree-parameters trees
                                  data parallel state)))
    (values (statistical-learning.tp:extract-predictions state)
            state)))
